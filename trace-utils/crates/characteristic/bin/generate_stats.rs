use clap::Parser;
use clio_utils::path::remove_extension;
use globwalk::glob;
use indicatif::{MultiProgress, ParallelProgressIterator, ProgressBar, ProgressStyle};
use kernel_density_estimation::prelude::*;
use polars::prelude::*;
use rayon::prelude::*;
use std::collections::HashMap;
use std::error::Error;
use std::io::Write;
use std::path::Path;
use std::process;

#[derive(Parser, Debug)]
#[command(version, about, long_about = None)]
struct Args {
    // Path to data directory
    #[clap(short = 'i', long = "input")]
    input: String,

    // Path to output directory
    #[clap(short = 'o', long = "output")]
    output: String,

    // Duration to split the trace
    #[clap(short = 'w', long = "window", default_value = "1m")]
    window: String,
}

fn default_pbar_style() -> Result<ProgressStyle, Box<dyn Error>> {
    let pbar = ProgressStyle::default_bar()
        .template("[{elapsed_precise}] {wide_bar} {pos}/{len} {msg}")?
        .progress_chars("=> ");
    Ok(pbar)
}

const NORMALIZED_METRICS: [&str; 11] = [
    // iops
    "iops",
    "read_iops",
    "write_iops",
    // ratio
    "read_ratio",
    "write_ratio",
    // iat
    "iat_avg",
    "read_iat_avg",
    "write_iat_avg",
    // size
    "size_avg",
    "read_size_avg",
    "write_size_avg",
];

fn series_to_vec(series: &Series) -> Vec<f64> {
    series
        .f64()
        .unwrap()
        .into_iter()
        .map(|v| v.unwrap())
        .collect()
}

fn cdf_lazyframe(df: &LazyFrame, column: &str) -> LazyFrame {
    df.clone()
        .with_column(col(column).alias("x"))
        .sort(&["x"], SortMultipleOptions::default())
        .select(&[
            // col(column).alias("x"),
            col(column).alias("real"),
            col("x"),
            (col("x").rank(
                RankOptions {
                    method: RankMethod::Average,
                    ..Default::default()
                },
                None,
            ) / col("x").count())
            .alias("y"),
        ])
}

enum StatRecord {
    Raw(Vec<f64>),
    PDF((Vec<f64>, Vec<f64>)),
    CDF((Vec<f64>, Vec<f64>)),
    CDFFromPDF((Vec<f64>, Vec<f64>)),
}

struct StatTask<P: AsRef<Path>> {
    path: P,
    x: Vec<f64>,
    y: Option<Vec<f64>>,
}

fn main() -> Result<(), Box<dyn Error>> {
    let start = std::time::Instant::now();

    let args = Args::parse();
    let matcher = format!("{}/**/*.{{csv}}", args.input);
    let output_dir = std::path::Path::new(&args.output);
    let files = glob(&matcher)?;
    let mut files = files.map(|f| f.unwrap()).collect::<Vec<_>>();
    files.sort_by(|a, b| natord::compare(&a.path().to_string_lossy(), &b.path().to_string_lossy()));

    println!("Reading traces and normalizing metrics...");
    let pbar = ProgressBar::new(files.len() as u64);
    pbar.set_style(default_pbar_style()?);
    pbar.set_message("Reading traces");

    let dataframes = pbar
        .wrap_iter(files.into_iter())
        .map(|entry| {
            let path = entry.path().to_path_buf();
            let df = LazyCsvReader::new(path.clone())
                .with_infer_schema_length(None)
                .with_ignore_errors(true)
                .with_has_header(true)
                .finish();
            if let Ok(characteristic_df) = df {
                // lazy normalization
                let df = characteristic_df;
                let df = df.with_columns(NORMALIZED_METRICS.map(|metric| {
                    (col(metric) / col(metric).mean()).alias(&format!("{}_norm", metric))
                }));
                (path, df)
            } else {
                eprintln!("Error reading file: {:?}", path);
                eprintln!("Error message: {:?}", df.err().unwrap());
                process::exit(1);
            }
        })
        .collect::<HashMap<_, _>>();

    println!("Generating stats...");
    let multi_pbar = MultiProgress::new();
    let tasks = dataframes
        .par_iter()
        .map(|(path, df)| {
            let pbar = multi_pbar.add(ProgressBar::new(NORMALIZED_METRICS.len() as u64));
            pbar.set_style(default_pbar_style().unwrap());
            pbar.set_message(format!(
                "Processing {:?}",
                path.parent().unwrap().file_stem().unwrap()
            ));
            // let iter = pbar.wrap_iter(NORMALIZED_METRICS.iter());
            let stat = NORMALIZED_METRICS
                .par_iter()
                .progress_with(pbar)
                .map(|metric| {
                    let df = df
                        .clone()
                        .select(&[
                            col(metric),
                            col(&format!("{}_norm", metric)),
                            (col(metric) / col(metric).shift_and_fill(1, 0))
                                .alias(&format!("{}_diff", metric))
                                .fill_null(0.0)
                                .fill_nan(0.0),
                            (col(&format!("{}_norm", metric))
                                / col(&format!("{}_norm", metric)).shift_and_fill(1, 0))
                            .fill_null(0.0)
                            .fill_nan(0.0)
                            .alias(&format!("{}_diff_norm", metric)),
                        ])
                        .select(&[
                            col(metric),
                            col(&format!("{}_norm", metric)),
                            when(col(&format!("{}_diff", metric)).is_infinite())
                                .then(0.0)
                                .otherwise(col(&format!("{}_diff", metric)))
                                .alias(&format!("{}_diff", metric)),
                            when(col(&format!("{}_diff_norm", metric)).is_infinite())
                                .then(0.0)
                                .otherwise(col(&format!("{}_diff_norm", metric)))
                                .alias(&format!("{}_diff_norm", metric)),
                        ]);
                    // let names = ["real", "norm", "diff", "diff_norm"];
                    let names = [
                        format!("{}", metric),
                        format!("{}_norm", metric),
                        format!("{}_diff", metric),
                        format!("{}_diff_norm", metric),
                    ];

                    let stats = names
                        .into_iter()
                        .map(|name| {
                            let n = name.as_str();
                            let cdf_df = df.clone();
                            let cdf_df = cdf_lazyframe(&cdf_df, n);
                            let cdf_df = cdf_df.collect().unwrap();
                            let x = series_to_vec(&cdf_df.column("real").unwrap());
                            let bandwidth = Scott;
                            let kernel = SilvermanKernel;
                            let kde = KernelDensityEstimator::new(x.clone(), bandwidth, kernel);
                            let pdf_x = x.clone();
                            let pdf_y = kde.pdf(&pdf_x);
                            let cdf_from_pdf_x = x.clone();
                            let cdf_from_pdf_y = kde.cdf(&cdf_from_pdf_x);
                            let cdf_x = series_to_vec(&cdf_df.column("x").unwrap());
                            let cdf_y = series_to_vec(&cdf_df.column("y").unwrap());

                            // original
                            let original_df = df.clone().select(&[col(n)]).collect().unwrap();
                            let x = series_to_vec(&original_df.column(n).unwrap());
                            // if n.contains("diff") {
                            //     // set the first value to 0
                            //     x[0] = 0.0;
                            // }
                            // let y = (0..x.len()).map(|i| i as f64).collect::<Vec<_>>();

                            let name = if name == metric.to_string() {
                                "real"
                            } else if name == format!("{}_norm", metric) {
                                "norm"
                            } else if name == format!("{}_diff", metric) {
                                "diff"
                            } else {
                                "diff_norm"
                            };
                            (
                                name,
                                vec![
                                    StatRecord::Raw(x),
                                    StatRecord::PDF((pdf_x, pdf_y)),
                                    StatRecord::CDFFromPDF((cdf_from_pdf_x, cdf_from_pdf_y)),
                                    StatRecord::CDF((cdf_x, cdf_y)),
                                ],
                            )
                        })
                        .collect::<HashMap<_, _>>();

                    // {
                    //     // add raw
                    //     let names = [
                    //         format!("{}", metric),
                    //         format!("{}_norm", metric),
                    //         format!("{}_diff", metric),
                    //         format!("{}_diff_norm", metric),
                    //     ];
                    //     let original_df = df
                    //         .clone()
                    //         .select(&names.map(|name| col(&name)))
                    //         .collect()
                    //         .unwrap();
                    //     let stats = names.map(|name| {
                    //         let n = name.as_str();
                    //         if n == metric {
                    //             return StatRecord::Raw((x, y));
                    //         }
                    //         StatRecord::Raw((x, y))
                    //     });

                    //     let x = series_to_vec(&original_df.column(n).unwrap());
                    //     let y = (0..x.len()).map(|i| i as f64).collect::<Vec<_>>();
                    //     // let x = series_to_vec(&original_df.column(metric).unwrap());
                    //     // let y = (0..x.len()).map(|i| i as f64).collect::<Vec<_>>();
                    //     stats.insert("raw", vec![StatRecord::Raw((x, y))]);
                    // }

                    (metric, stats)
                })
                .collect::<HashMap<_, _>>();

            let pbar = multi_pbar.add(ProgressBar::new(NORMALIZED_METRICS.len() as u64));
            pbar.set_style(default_pbar_style().unwrap());
            pbar.set_message(format!(
                "Generate task {:?}",
                path.parent().unwrap().file_stem().unwrap()
            ));
            // let iter = pbar.wrap_iter(NORMALIZED_METRICS.into_iter());
            NORMALIZED_METRICS
                .par_iter()
                .progress_with(pbar)
                .map(move |metric| {
                    let stat_map = stat.get(&metric).unwrap();
                    stat_map
                        .into_iter()
                        .map(|(data_type, vec)| {
                            vec.into_iter()
                                .map(|v| {
                                    let (t, x, y) = match v {
                                        StatRecord::Raw(x) => ("raw", x, None),
                                        StatRecord::PDF((x, y)) => ("pdf", x, Some(y)),
                                        StatRecord::CDF((x, y)) => ("cdf", x, Some(y)),
                                        StatRecord::CDFFromPDF((x, y)) => {
                                            ("cdf_from_pdf", x, Some(y))
                                        }
                                    };
                                    let p = path.parent().unwrap_or(Path::new("")).to_path_buf();
                                    let name = remove_extension(p);
                                    let name = name.file_name().unwrap();

                                    let by_metric_output_dir_path =
                                        output_dir.join("by-metric").join(t);
                                    let by_metric_output_dir_path =
                                        by_metric_output_dir_path.join(data_type).join(metric);
                                    let by_metric_output_dir_path =
                                        remove_extension(by_metric_output_dir_path);
                                    let by_metric_output_dir_path = by_metric_output_dir_path
                                        .join(format!("{}.dat", name.to_string_lossy()));

                                    let by_window_output_dir_path =
                                        output_dir.join("by-window").join(t);
                                    let by_window_output_dir_path =
                                        by_window_output_dir_path.join(data_type).join(name);
                                    let by_window_output_dir_path =
                                        remove_extension(by_window_output_dir_path);
                                    let by_window_output_dir_path =
                                        by_window_output_dir_path.join(format!("{}.dat", metric));

                                    vec![
                                        StatTask {
                                            path: by_metric_output_dir_path,
                                            x: x.to_vec(),
                                            y: y.cloned(),
                                        },
                                        StatTask {
                                            path: by_window_output_dir_path,
                                            x: x.to_vec(),
                                            y: y.cloned(),
                                        },
                                    ]
                                })
                                .flatten()
                                .collect::<Vec<_>>()
                        })
                        .flatten()
                        .collect::<Vec<_>>()
                })
                .flatten()
                .collect::<Vec<_>>()
        })
        .flatten()
        .collect::<Vec<_>>();

    let pbar = ProgressBar::new(tasks.len() as u64);
    pbar.set_style(default_pbar_style()?);
    pbar.set_message("Writing stats");
    tasks.par_iter().progress_with(pbar).for_each(|task| {
        std::fs::create_dir_all(&task.path.parent().unwrap()).unwrap();
        let mut file = std::fs::File::create(&task.path).unwrap();
        if let Some(y) = &task.y {
            for (x, y) in task.x.iter().zip(y.iter()) {
                file.write(format!("{}\t{}\n", x, y).as_bytes()).unwrap();
            }
            return;
        }
        for x in task.x.iter() {
            file.write(format!("{}\n", x).as_bytes()).unwrap();
        }
    });

    let duration = start.elapsed();
    println!("Time elapsed in main() is: {:?}", duration);

    Ok(())
}
