use clap::Parser;
use clio_utils::dpi::Inch;
use clio_utils::dpi::Point;
use clio_utils::image::convert_eps_to_png;
use clio_utils::path::is_program_in_path;
use clio_utils::path::remove_extension;
use dashmap::DashMap;
use globwalk::glob;
use gnuplot::AutoOption;
use gnuplot::MarginSide;
use gnuplot::{AxesCommon, Figure};
use indicatif::{ParallelProgressIterator, ProgressBar, ProgressStyle};
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

const DPI: f32 = 1000.0;
const LINEWIDTH: f64 = 5.0;

const COLORS: [&str; 12] = [
    "blue", "red", "green", "purple", "orange", "cyan", "magenta", "gold", "black", "gray",
    "brown", "pink",
];

fn cdf_lazyframe(df: &LazyFrame, column: &str) -> LazyFrame {
    df.clone()
        .with_column(col(column).alias("x"))
        .sort(&["x"], SortMultipleOptions::default())
        .select(&[
            // col(column).alias("x"),
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

fn cdf_lazyframe_multiple(df: &LazyFrame, columns: &[&str]) -> Vec<LazyFrame> {
    columns
        .iter()
        .map(|column| cdf_lazyframe(df, column))
        .collect()
}

fn series_to_vec(series: &Series) -> Vec<f64> {
    series
        .f64()
        .unwrap()
        .into_iter()
        .map(|v| v.unwrap())
        .collect()
}

struct PlotCdfSingleOptions<'a, P: AsRef<Path>> {
    title: String,
    output_path: P,
    size: Point<Inch>,
    lines: Vec<(Vec<f64>, Vec<f64>)>,
    line_options: Vec<Vec<gnuplot::PlotOption<&'a str>>>,
    gs: bool,
    xrange: Option<(AutoOption<f64>, AutoOption<f64>)>,
    yrange: Option<(AutoOption<f64>, AutoOption<f64>)>,
    xlabel_options: Vec<gnuplot::LabelOption<&'a str>>,
    ylabel_options: Vec<gnuplot::LabelOption<&'a str>>,
    samples: Option<usize>,
    pre_commands: Option<String>,
    save_data: bool,
    margins: Option<&'a [MarginSide]>,
}

fn plot_cdf_single<'a, P: AsRef<Path>>(
    options: &'a PlotCdfSingleOptions<'a, P>,
) -> Result<(), Box<dyn Error>> {
    // let name = path.parent().unwrap_or(Path::new("")).file_name().unwrap();
    let PlotCdfSingleOptions {
        title,
        // name,
        // metric,
        output_path,
        size: Point {
            x: width,
            y: height,
        },
        // x,
        // y,
        lines,
        line_options,
        gs,
        xrange,
        yrange,
        xlabel_options,
        ylabel_options,
        samples,
        pre_commands,
        save_data,
        margins,
    } = options;

    let output_path = output_path.as_ref();

    std::fs::create_dir_all(output_path.parent().unwrap())?;

    let mut fig = Figure::new();
    fig.set_terminal("postscript eps enhanced color 20 font", "")
        .set_pre_commands(&format!(
            "set key autotitle columnheader
set samples {}
{}
",
            samples.unwrap_or(30000),
            pre_commands.clone().unwrap_or("".to_string())
        ));

    let mut ax = fig
        .axes2d()
        .set_title(&title, &[])
        .set_x_label("", xlabel_options)
        .set_y_label("Quantile", ylabel_options)
        .set_margins(margins.unwrap_or(&[]));
    // .set_aspect_ratio(AutoOption::Fix(1.0));
    // .set_x_ticks(Some((AutoOption::Auto, 1)), &[], xticks_options);

    for ((x, y), opt) in lines.into_iter().zip(line_options) {
        // ax.lines(x, y, &[gnuplot::Caption(&metric.replace('_', r"\_"))]);
        ax = ax.lines(x, y, opt);
    }

    if let Some((min, max)) = xrange {
        ax.set_x_range(*min, *max);
    }

    if let Some((min, max)) = yrange {
        ax.set_y_range(*min, *max);
    }

    fig.save_to_eps(&output_path, *height, *width)
        .expect("Error saving eps figure");
    fig.echo_to_file(&output_path.with_extension("plot").to_str().unwrap());

    if *gs {
        let png_output_path = output_path.with_extension("png");
        let png_output_path = png_output_path.as_path();
        convert_eps_to_png(&output_path, &png_output_path, &DPI)?;
        // let now = std::time::SystemTime::now();
        // std::fs::File::open(png_output_path)?.set_times(
        //     std::fs::FileTimes::new()
        //         .set_accessed(now)
        //         .set_modified(now),
        // )?;
        // touch the file
    }

    if *save_data {
        let data_output_dir_path = output_path.with_extension("").join("data");
        std::fs::create_dir_all(&data_output_dir_path)?;

        for (i, (x, y)) in lines.into_iter().enumerate() {
            let data_output_path = data_output_dir_path.clone().join(format!("{}.dat", i));
            let mut data = std::fs::File::create(data_output_path)?;
            for (x, y) in x.iter().zip(y.iter()) {
                data.write(format!("{}\t{}\n", x, y).as_bytes())?;
            }
        }
    }

    Ok(())
}

// enum CdfRecord {
//     Real(Vec<f64>, Vec<f64>),
//     Diff(Vec<f64>, Vec<f64>),
//     Norm(Vec<f64>, Vec<f64>),
//     DiffNorm(Vec<f64>, Vec<f64>),
// }

fn main() -> Result<(), Box<dyn Error>> {
    let start = std::time::Instant::now();

    let is_gs_exists = is_program_in_path("gs");

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

    let cdfs = dataframes
        .par_iter()
        .map(|(path, df)| {
            // each lazyframe will output EACH CDF of the normalized metrics

            let cdf_metrics = NORMALIZED_METRICS
                .par_iter()
                .map(|metric| {
                    let df = df.clone().select(&[
                        col(metric),
                        col(&format!("{}_norm", metric)),
                        (col(metric) / col(metric).shift_and_fill(1, 1))
                            .alias(&format!("{}_diff", metric)),
                        (col(&format!("{}_norm", metric))
                            / col(&format!("{}_norm", metric)).shift_and_fill(1, 1))
                        .alias(&format!("{}_diff_norm", metric)),
                    ]);
                    // let names = ["real", "norm", "diff", "diff_norm"];
                    let names = [
                        format!("{}", metric),
                        format!("{}_norm", metric),
                        format!("{}_diff", metric),
                        format!("{}_diff_norm", metric),
                    ];
                    let names_str = names.iter().map(|v| v.as_str()).collect::<Vec<_>>();
                    let dfs = cdf_lazyframe_multiple(&df, &names_str);
                    let dfs = dfs
                        .into_iter()
                        .map(|df| {
                            let df = df.collect().unwrap();
                            let x = series_to_vec(&df.column("x").unwrap());
                            let y = series_to_vec(&df.column("y").unwrap());
                            (x, y)
                        })
                        .collect::<Vec<_>>();
                    let dfs = names
                        .clone()
                        .iter()
                        .zip(dfs.into_iter())
                        .map(|(name, (x, y))| (name.to_string(), (x, y)))
                        .collect::<HashMap<_, _>>();

                    // let x = series_to_vec(&df.column(metric).unwrap());
                    // let y = series_to_vec(&df.column(&format!("{}_cdf", metric)).unwrap());
                    // let norm_x = series_to_vec(&df.column(&format!("{}_norm", metric)).unwrap());
                    // let norm_y =
                    //     series_to_vec(&df.column(&format!("{}_norm_cdf", metric)).unwrap());

                    // let diff_x = series_to_vec(&df.column("diff").unwrap());
                    // let diff_y = series_to_vec(&df.column("diff_cdf").unwrap());

                    // let diff_norm_x = series_to_vec(&df.column("diff_norm").unwrap());
                    // let diff_norm_y = series_to_vec(&df.column("diff_norm_cdf").unwrap());

                    (
                        metric,
                        dfs,
                        // vec![
                        //     CdfRecord::Real(x, y),
                        //     CdfRecord::Norm(norm_x, norm_y),
                        //     CdfRecord::Diff(diff_x, diff_y),
                        //     CdfRecord::DiffNorm(diff_norm_x, diff_norm_y),
                        // ],
                    )
                })
                .collect::<HashMap<_, _>>();

            (path, cdf_metrics)
        })
        .collect::<HashMap<_, _>>();

    println!("Generating CDF tasks");
    let pbar = ProgressBar::new(cdfs.len() as u64);
    pbar.set_style(default_pbar_style()?);
    pbar.set_message("Generating CDF tasks");
    let plot_size = Point::<f32>::new(2.0, 1.7);
    let tasks = cdfs
        .par_iter()
        .progress_with(pbar)
        .map(|(path, cdf_metrics)| {
            NORMALIZED_METRICS
                .iter()
                .fold(Vec::new(), |mut tasks, metric| {
                    let cdfs = cdf_metrics.get(metric).unwrap();

                    let name = path.parent().unwrap_or(Path::new("")).file_name().unwrap();

                    // let plot_types = vec!["real", "norm", "diff", "diff_norm"];
                    for (plot_type, (x, y)) in cdfs.iter() {
                        // let metric = metric.replace('_', r"\_");
                        let t = if plot_type.as_str().contains("diff_norm") {
                            "diff_norm"
                        } else if plot_type.as_str().contains("diff") {
                            "diff"
                        } else if plot_type.as_str().contains("norm") {
                            "norm"
                        } else {
                            "real"
                        };

                        let lines = vec![(x.clone(), y.clone())];
                        let line_options =
                            vec![vec![gnuplot::LineWidth(LINEWIDTH), gnuplot::Caption("")]];

                        let plot_dir_path = output_dir.join(&name);
                        let plot_dir_path = remove_extension(plot_dir_path);
                        let plot_dir_path = plot_dir_path.join(t);
                        {
                            let cut_plot_dir_path = plot_dir_path.clone();
                            let cut_plot_dir_path = cut_plot_dir_path.join("cut");
                            let max_x = x[(0.99 * x.len() as f64) as usize - 1];
                            std::fs::create_dir_all(&cut_plot_dir_path).unwrap();
                            let output_path = cut_plot_dir_path.join(format!("{}.eps", metric));
                            let plot_type = plot_type.replace('_', r"\_");
                            let plot_title = format!("Cut CDF of {}", plot_type);

                            let plot_cdf_opts = PlotCdfSingleOptions {
                                title: plot_title,
                                output_path: output_path.clone(),
                                size: plot_size.clone(),
                                lines: lines.clone(),
                                line_options: line_options.clone(),
                                gs: is_gs_exists,
                                xrange: Some((AutoOption::Auto, AutoOption::Fix(max_x))),
                                yrange: None,
                                xlabel_options: vec![],
                                ylabel_options: vec![],
                                samples: None,
                                margins: Some(&[
                                    MarginSide::MarginLeft(0.15),
                                    MarginSide::MarginRight(0.95),
                                    MarginSide::MarginTop(0.13),
                                    MarginSide::MarginBottom(0.87),
                                ]),
                                pre_commands: Some(
                                    "set key off
set xtics rotate by 45 right"
                                        .to_string(),
                                ),
                                save_data: true,
                            };
                            // plot_cdf_single(&plot_cdf_opts).unwrap();
                            tasks.push(plot_cdf_opts);
                        }

                        {
                            let full_plot_dir_path = plot_dir_path.clone();
                            let full_plot_dir_path = full_plot_dir_path.join("full");

                            std::fs::create_dir_all(&full_plot_dir_path).unwrap();
                            let output_path = full_plot_dir_path.join(format!("{}.eps", metric));

                            let plot_type = plot_type.replace('_', r"\_");
                            let plot_title = format!("CDF of {}", plot_type);

                            let plot_cdf_opts = PlotCdfSingleOptions {
                                title: plot_title,
                                output_path: output_path.clone(),
                                size: plot_size.clone(),
                                lines: lines,
                                line_options: line_options,
                                gs: is_gs_exists,
                                xrange: None,
                                yrange: None,
                                xlabel_options: vec![],
                                ylabel_options: vec![],
                                samples: None,
                                margins: Some(&[
                                    MarginSide::MarginLeft(0.15),
                                    MarginSide::MarginRight(0.95),
                                    MarginSide::MarginTop(0.13),
                                    MarginSide::MarginBottom(0.87),
                                ]),
                                pre_commands: Some(
                                    "set key off
set xtics rotate by 45 right"
                                        .to_string(),
                                ),
                                save_data: true,
                            };

                            // plot_cdf_single(&plot_cdf_opts).unwrap();
                            tasks.push(plot_cdf_opts);
                        }
                    }

                    tasks
                })
        })
        .flatten()
        .collect::<Vec<_>>();

    println!("Plotting CDFs");
    let pbar = ProgressBar::new(tasks.len() as u64);
    pbar.set_style(default_pbar_style()?);
    pbar.set_message("Plotting CDFs");
    tasks
        .par_iter()
        .progress_with(pbar)
        .for_each(|task| plot_cdf_single(task).unwrap());

    println!("CDF metric combination");
    let pbar = ProgressBar::new(cdfs.len() as u64);
    pbar.set_style(default_pbar_style()?);
    pbar.set_message("Plotting CDFs");
    let hashmap = DashMap::new();
    cdfs.par_iter().for_each(|(path, cdf_metrics)| {
        cdf_metrics.iter().for_each(|(name, cdf)| {
            let mut cdf_map = hashmap.entry(name).or_insert(HashMap::new());
            cdf_map.entry(path).or_insert(cdf);
        });
    });

    let hashmap = hashmap
        .into_iter()
        .map(|entry| {
            let (metric, cdf_map) = entry;
            let mut cdf_map = cdf_map.clone().into_iter().collect::<Vec<_>>();

            // sort based on file name
            // cdf_map.sort_by(|a, b| natord::compare(&a.0.to_string_lossy(), &b.0.to_string_lossy()));
            cdf_map.sort_by(|a, b| {
                let a = &a.0.parent().unwrap().file_stem().unwrap().to_string_lossy();
                let b = &b.0.parent().unwrap().file_stem().unwrap().to_string_lossy();
                fundu::parse_duration(a)
                    .unwrap()
                    .cmp(&fundu::parse_duration(b).unwrap())
            });

            (metric, cdf_map)
        })
        .collect::<HashMap<_, _>>();

    println!("Generating combined CDF tasks");
    let pbar = ProgressBar::new(hashmap.len() as u64);
    pbar.set_style(default_pbar_style()?);
    pbar.set_message("Generating combined CDF tasks");
    let combined_plot_dir_path = output_dir.join("combined");
    let combined_plot_size = Point::new(2.0, 3.0);
    let tasks = hashmap
        .into_par_iter()
        .progress_with(pbar)
        .map(|(metric, cdf_map)| {
            let mut tasks = Vec::new();
            let mut hashmap_max_x = HashMap::new();
            let mut hashmap_plot_type = HashMap::new();
            let mut hashmap_cut_plot_type = HashMap::new();
            if cdf_map.len() == 0 {
                return vec![];
            }
            // let max_x = first_elem
            //     .1
            //     .iter()
            //     .map(|(x, _)| x.last().unwrap())
            //     .max()
            //     .unwrap();

            // let first_elem = cdf_map.iter().next().unwrap().1.iter().next().unwrap().1 .0;

            let cdf_len = cdf_map.len();
            let margins = if cdf_len > 7 {
                &[
                    MarginSide::MarginLeft(0.1),
                    MarginSide::MarginRight(0.70),
                    MarginSide::MarginTop(0.13),
                    MarginSide::MarginBottom(0.87),
                ]
            } else {
                &[
                    MarginSide::MarginLeft(0.15),
                    MarginSide::MarginRight(0.95),
                    MarginSide::MarginTop(0.13),
                    MarginSide::MarginBottom(0.87),
                ]
            };
            // let combined_plot_size = if cdf_len > 7 {
            //     Point::<f32>::new(2.0, 5.0)
            // } else {
            //     Point::<f32>::new(2.0, 2.0)
            // };

            let default_pre_commands = if cdf_len > 7 { "set key out" } else { "" };
            // println!("default_pre_commands: {}", default_pre_commands);

            for (_, map) in cdf_map.iter() {
                for (plot_type, (x, _)) in map.iter() {
                    let max_x = hashmap_max_x.entry(plot_type).or_insert(f64::MAX);
                    let p = x[(0.95 * x.len() as f64) as usize - 1];
                    if p < *max_x {
                        *max_x = p;
                    }
                }
            }

            // for k in hashmap_max_x.keys() {
            //     println!("{}: {}", k, hashmap_max_x.get(k).unwrap());
            // }

            for (i, (path, map)) in cdf_map.iter().enumerate() {
                for (plot_type, (x, y)) in map.iter() {
                    let t = if plot_type.as_str().contains("diff_norm") {
                        "diff_norm"
                    } else if plot_type.as_str().contains("diff") {
                        "diff"
                    } else if plot_type.as_str().contains("norm") {
                        "norm"
                    } else {
                        "real"
                    };

                    let plot_type_norm = plot_type.replace('_', r"\_");

                    {
                        let pre_commands =
                            format!("set xtics rotate by 45 right\n{}", default_pre_commands);
                        let plot_type_entry = hashmap_plot_type
                            .entry(plot_type.to_string())
                            .or_insert(PlotCdfSingleOptions {
                                title: format!("CDF of {}", plot_type_norm),
                                output_path: combined_plot_dir_path
                                    .join("full")
                                    .join(t)
                                    .join(format!("{}.eps", metric)),
                                size: combined_plot_size.clone(),
                                lines: vec![],
                                line_options: vec![],
                                gs: is_gs_exists,
                                xrange: None,
                                yrange: None,
                                xlabel_options: vec![],
                                ylabel_options: vec![],
                                samples: None,
                                margins: Some(margins),
                                pre_commands: Some(pre_commands),
                                save_data: false,
                            });

                        plot_type_entry.line_options.push(vec![
                            gnuplot::LineWidth(LINEWIDTH),
                            gnuplot::Caption(
                                path.parent()
                                    .unwrap()
                                    .file_name()
                                    .unwrap()
                                    .to_str()
                                    .unwrap(),
                            ),
                        ]);
                        plot_type_entry.lines.push((x.clone(), y.clone()));
                    }

                    {
                        let pre_commands = if metric.contains("size") {
                            "
set xtics rotate by 45 right"
                        } else {
                            "
set xtics rotate by 45 right"
                        };
                        let pre_commands = format!("{}\n{}", pre_commands, default_pre_commands);

                        let max_x = hashmap_max_x.get(plot_type).unwrap();

                        let cut_plot_type_entry = hashmap_cut_plot_type
                            .entry(plot_type.to_string())
                            .or_insert(PlotCdfSingleOptions {
                                title: format!("CDF of {}", plot_type_norm),
                                output_path: combined_plot_dir_path
                                    .join("cut")
                                    .join(t)
                                    .join(format!("{}.eps", metric)),
                                size: combined_plot_size.clone(),
                                lines: vec![],
                                line_options: vec![],
                                gs: is_gs_exists,
                                xrange: Some((AutoOption::Auto, AutoOption::Fix(*max_x))),
                                yrange: None,
                                xlabel_options: vec![],
                                ylabel_options: vec![],
                                samples: None,
                                margins: Some(margins),
                                pre_commands: Some(pre_commands),
                                save_data: false,
                            });

                        cut_plot_type_entry.line_options.push(vec![
                            gnuplot::LineWidth(LINEWIDTH),
                            gnuplot::Caption(
                                path.parent()
                                    .unwrap()
                                    .file_name()
                                    .unwrap()
                                    .to_str()
                                    .unwrap(),
                            ),
                            gnuplot::Color(COLORS[i % COLORS.len()]),
                        ]);
                        cut_plot_type_entry.lines.push((x.clone(), y.clone()));
                    }
                }
            }

            for (_, plot_type_entry) in hashmap_plot_type.into_iter() {
                tasks.push(plot_type_entry);
            }

            for (_, cut_plot_type_entry) in hashmap_cut_plot_type.into_iter() {
                tasks.push(cut_plot_type_entry);
            }

            tasks
        })
        .flatten()
        .collect::<Vec<_>>();

    println!("Plotting Combined CDFs");
    let pbar = ProgressBar::new(tasks.len() as u64);
    pbar.set_style(default_pbar_style()?);
    pbar.set_message("Plotting Combined CDFs");

    tasks
        .par_iter()
        .progress_with(pbar)
        .for_each(|task| plot_cdf_single(task).unwrap());
    let duration = start.elapsed();
    println!("Time elapsed in main() is: {:?}", duration);

    Ok(())
}
