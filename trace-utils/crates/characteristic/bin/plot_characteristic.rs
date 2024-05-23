use clap::Parser;
use clio_utils::path::remove_extension;
use globwalk::glob;
use gnuplot::{AxesCommon, Figure};
use indicatif::{ParallelProgressIterator, ProgressBar, ProgressStyle};
use polars::lazy::dsl::{col, concat_list};
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

// fn norm_mean(series: &Series) -> Series {
//     let mean = series.mean().unwrap();

//     series / mean
// }

const NORMALIZED_METRICS: [&str; 8] = [
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
];

// fn cdf(x: &[f64]) -> Vec<(f64, f64)> {
//     let ln = x.len() as f64;
//     let mut x_ord = x.to_vec();
//     x_ord.sort_by(|a, b| a.partial_cmp(b).unwrap());

//     if let Some(mut previous) = x_ord.get(0).map(|&f| f) {
//         let mut cdf = Vec::new();
//         for (i, f) in x_ord.into_iter().enumerate() {
//             if f != previous {
//                 cdf.push((previous, i as f64 / ln));
//                 previous = f;
//             }
//         }

//         cdf.push((previous, 1.0));
//         cdf
//     } else {
//         Vec::new()
//     }
// }

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

    let files_iter = pbar.wrap_iter(files.iter());

    let dataframes = files_iter
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
                // for metric in &NORMALIZED_METRICS {
                let df = df.with_columns(
                    NORMALIZED_METRICS.map(|metric| col(metric) / col(metric).mean()),
                );
                (path, df)
            } else {
                eprintln!("Error reading file: {:?}", path);
                eprintln!("Error message: {:?}", df.err().unwrap());
                process::exit(1);
            }
        })
        .collect::<HashMap<_, _>>();

    let cdfs = dataframes
        .par_iter()
        .map(|(path, df)| {
            // each lazyframe will output EACH CDF of the normalized metrics
            let cdf = df
                .clone()
                .select(NORMALIZED_METRICS.map(|metric| col(metric)))
                .with_columns(NORMALIZED_METRICS.map(|metric| {
                    (col(metric).sort(SortOptions::default()).rank(
                        RankOptions {
                            method: RankMethod::Average,
                            ..Default::default()
                        },
                        None,
                    ) / col(metric).count())
                    .alias(&format!("{}_cdf", metric))
                }));

            let cdf_metrics = NORMALIZED_METRICS.map(|metric| {
                let expr_concat_list =
                    vec![concat_list(&[col(metric), col(&format!("{}_cdf", metric))])
                        .unwrap()
                        .alias(metric)];

                let temp_cdf = cdf.clone();
                temp_cdf
                    .select(&expr_concat_list)
                    .collect()
                    .unwrap()
                    .iter()
                    .map(|s| {
                        let ss = s.clone();
                        ss.list()
                            .unwrap()
                            .into_iter()
                            .map(|v| {
                                let v = v.unwrap().clone();
                                (
                                    v.get(0)
                                        .unwrap()
                                        .cast(&DataType::Float64)
                                        .try_extract()
                                        .unwrap(),
                                    v.get(1)
                                        .unwrap()
                                        .cast(&DataType::Float64)
                                        .try_extract()
                                        .unwrap(),
                                )
                            })
                            .collect::<Vec<(f64, f64)>>()
                    })
                    .flatten()
                    .collect::<Vec<_>>()
            });

            (path, cdf_metrics)
        })
        .collect::<HashMap<_, _>>();

    // for (path, cdf_metrics) in cdfs.iter() {
    //     for (i, metric) in NORMALIZED_METRICS.iter().enumerate() {
    //         println!("{}: {:?}", metric, cdf_metrics[i]);
    //         process::exit(1);
    //     }
    // }

    println!("Writing CDFs...");
    let pbar = ProgressBar::new(cdfs.len() as u64);
    pbar.set_style(default_pbar_style()?);
    pbar.set_message("Writing CDFs");
    cdfs.par_iter()
        .progress_with(pbar)
        .for_each(|(path, cdf_metrics)| {
            for (i, metric) in NORMALIZED_METRICS.iter().enumerate() {
                let name = path.parent().unwrap_or(Path::new("")).file_name().unwrap();
                let output_path = output_dir.join(name);
                let output_path = remove_extension(output_path);
                std::fs::create_dir_all(&output_path).unwrap();
                let output_path = output_path.join(format!("{}.cdf", metric));
                let mut file = std::fs::File::create(output_path).unwrap();
                let cdf = cdf_metrics[i].clone();

                for (x, y) in cdf {
                    file.write(format!("{:4}\t{:4}\n", x, y).as_bytes())
                        .unwrap();
                }
            }
        });

    // plot the cdfs

    println!("Plotting CDFs...");
    let pbar = ProgressBar::new(cdfs.len() as u64);
    pbar.set_style(default_pbar_style()?);
    pbar.set_message("Plotting CDFs");

    cdfs.par_iter()
        .progress_with(pbar)
        .for_each(|(path, cdf_metrics)| {
            for (i, metric) in NORMALIZED_METRICS.iter().enumerate() {
                let name = path.parent().unwrap_or(Path::new("")).file_name().unwrap();
                let output_path = output_dir.join(&name);
                let output_path = remove_extension(output_path);
                std::fs::create_dir_all(&output_path).unwrap();
                let png_output_path = output_path.join(format!("{}.png", metric));
                let eps_output_path = output_path.join(format!("{}.eps", metric));

                let mut fg = Figure::new();
                fg.set_terminal("pngcairo", &png_output_path.to_str().unwrap())
                    .axes2d()
                    .set_title(
                        &format!("CDF of {} ({})", name.to_str().unwrap(), metric),
                        &[],
                    )
                    .set_x_label("Value", &[])
                    .set_y_label("CDF", &[])
                    .lines(
                        cdf_metrics[i].iter().map(|(x, _)| *x),
                        cdf_metrics[i].iter().map(|(_, y)| *y),
                        &[gnuplot::Caption(metric)],
                    );

                // fg.echo_to_file(&eps_output_path.to_str().unwrap());
                fg.save_to_png(png_output_path, 800, 600)
                    .expect("Error saving png figure");
                fg.save_to_eps(eps_output_path, 800.0, 600.0)
                    .expect("Error saving eps figure");
            }
        });

    let duration = start.elapsed();
    println!("Time elapsed in main() is: {:?}", duration);

    Ok(())
}
