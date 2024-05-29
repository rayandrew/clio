use clap::Parser;
use clio_utils::cdf::calc_cdf;
use clio_utils::file::BufReader;
use clio_utils::pbar::default_pbar_style;
use indicatif::{ParallelProgressIterator, ProgressBar};
use rayon::prelude::*;
use std::error::Error;
use std::io::Write;
use std::path::Path;
use std::{fs, process};

#[derive(Parser, Debug)]
#[command(version, about, long_about = None)]
struct Args {
    // Path to data directory
    #[clap(short = 'i', long = "input")]
    input: String,

    // Path to output directory
    #[clap(short = 'o', long = "output")]
    output: String,

    // Diff threshold
    #[clap(short = 'd', long = "diff-threshold", default_value = "0.05")]
    diff_threshold: f64,

    // Stationary Drift Threshold
    #[clap(short = 's', long = "stationary-threshold", default_value = "30")]
    stationary_threshold: usize,

    // Grouping threshold
    #[clap(short = 'g', long = "group-threshold", default_value = "100")]
    group_threshold: f64,

    // Group offset
    #[clap(short = 'f', long = "group-offset", default_value = "50.0")]
    group_offset: f64,

    // Drift end start threshold
    #[clap(short = 't', long = "drift-threshold", default_value = "80")]
    drift_threshold: usize,

    // Rolling window
    #[clap(short = 'w', long = "window", default_value = "10")]
    window: usize,
}

#[derive(Debug)]
struct Row {
    idx: usize,
    value: f64,
    diff: f64, // diff between current and previous value
    increasing_idx: i64,
    decreasing_idx: i64,
    stability: usize,
    group: i32,
}

fn count_drifts(rows: &[Row]) -> (usize, usize) {
    let mut increasing_drifts = 0;
    let mut decreasing_drifts = 0;

    for (row, prev_row) in rows.iter().skip(1).zip(rows.iter()) {
        if row.increasing_idx != prev_row.increasing_idx {
            increasing_drifts += 1;
        } else if row.decreasing_idx != prev_row.decreasing_idx {
            decreasing_drifts += 1;
        }
    }

    (increasing_drifts, decreasing_drifts)
}

fn determine_group(value: f64, threshold: f64, offset: f64) -> i32 {
    if value < 0.0 {
        return 0;
    }

    if value <= offset {
        return 1;
    }

    ((value - offset) / threshold).floor() as i32 + 2
}

fn main() -> Result<(), Box<dyn Error>> {
    let start = std::time::Instant::now();

    let args = Args::parse();
    let input = Path::new(&args.input);
    let output_dir = Path::new(&args.output);
    let diff_threshold = args.diff_threshold;
    let stationary_threshold = args.stationary_threshold - 1;
    let group_threshold = args.group_threshold;
    let group_offset = args.group_offset;
    let drift_threshold = args.drift_threshold;

    if !input.is_file() {
        eprintln!("Input file does not exist");
        process::exit(1);
    }

    fs::create_dir_all(output_dir)?;

    // read the input file, will be one column float points
    let mut data = Vec::new();
    for line in BufReader::open(input)? {
        let line = line?;
        let value: f64 = line.trim().parse()?;
        data.push(value);
    }

    let avg_data = data.iter().sum::<f64>() / data.len() as f64;
    let norm_data = data.iter().map(|x| x / avg_data).collect::<Vec<f64>>();
    let diff_data = norm_data
        .windows(2)
        .map(|x| if x[0] > 0.0 { x[1] / x[0] } else { 0.0 })
        .collect::<Vec<f64>>();

    // for (idx, &value) in diff_data.iter().enumerate() {
    //     println!("IDX: {}, REAL: {}, VALUE: {}", idx, data[idx], value);
    //     if idx == 10 {
    //         process::exit(0);
    //     }
    // }

    println!("Data length: {}", data.len());

    let mut rows = Vec::<Row>::with_capacity(data.len());

    // create 2 pointers for idx and value. current value and next value
    rows.push(Row {
        idx: 0,
        value: data[0],
        diff: 0.0,
        // value: norm_data[0],
        increasing_idx: -1,
        decreasing_idx: -1,
        stability: 0,
        group: determine_group(data[0], group_threshold, group_offset),
    });

    let increasing_bound = 1.0 + diff_threshold;
    let decreasing_bound = 1.0 - diff_threshold;

    for (current_idx, &current_value) in data.iter().skip(1).enumerate() {
        // current_value and prev_value is DIFF between 2 windows
        let current_idx = current_idx + 1;
        let prev_idx = current_idx - 1;
        // let current_norm_value = norm_data[current_idx];
        // let prev_norm_value = norm_data[prev_idx];
        // println!("Current idx: {}, Prev idx: {}", current_idx, prev_idx);
        let prev_row = &rows[prev_idx];
        let prev_group = prev_row.group;
        let mut row = Row {
            idx: current_idx,
            value: current_value,
            diff: diff_data[prev_idx],
            // value: current_norm_value,
            increasing_idx: prev_row.increasing_idx,
            decreasing_idx: prev_row.decreasing_idx,
            stability: prev_row.stability,
            group: determine_group(current_value, group_threshold, group_offset),
        };
        // let diff = current_norm_value / prev_norm_value;
        // let diff = diff_data[prev_idx];
        if prev_group < row.group {
            row.increasing_idx = prev_idx as i64;
            row.stability = 0;
        } else if prev_group > row.group {
            row.decreasing_idx = prev_idx as i64;
            row.stability = 0;
        } else {
            row.stability += 1;
        }
        rows.push(row);
        // if current_idx == 10 {
        //     break;
        // }
    }

    let (num_increase_drifts, num_decrease_drifts) = count_drifts(&rows);
    println!("Number of increasing drifts: {}", num_increase_drifts);
    println!("Number of decreasing drifts: {}", num_decrease_drifts);

    // for (row, prev_row) in rows.iter().skip(1).zip(rows.iter()) {
    //     if row.increasing_idx != prev_row.increasing_idx {
    //         num_increasing_drift += 1;
    //     } else if row.decreasing_idx != prev_row.decreasing_idx {
    //         num_decreasing_drift += 1;
    //     }
    // }

    // println!("Number of increasing drift: {}", num_increasing_drift);
    // println!("Number of decreasing drift: {}", num_decreasing_drift);

    // find index where stability is equal to stationary_threshold
    let mut stationary_drifts = Vec::<usize>::new();
    for (idx, row) in rows.iter().enumerate() {
        if row.stability == stationary_threshold {
            stationary_drifts.push(idx);
        }
    }

    let mut file = fs::File::create(output_dir.join("drifts.csv"))?;
    file.write(b"start,end,start_group,end_group\n")?;
    // generate possible drift start and index
    // assume we have stationary drifts as follow [1, 5, 10, 15, 20]
    // mean we will generate [(1, 5), (1, 10), (1, 15), (1, 20), (5, 10), (5, 15), ... ] as possible drifts
    // cartesian product of stationary drifts
    let mut possible_drifts = Vec::<(usize, usize)>::new();
    for (idx, &drift_idx) in stationary_drifts.iter().enumerate() {
        for &next_drift_idx in stationary_drifts.iter().skip(idx + 1) {
            if (next_drift_idx - drift_idx) < drift_threshold {
                continue;
            }
            let start_current_drift_idx = drift_idx - stationary_threshold;
            let end_next_drift_idx = next_drift_idx;
            let data = &rows[start_current_drift_idx..=end_next_drift_idx];
            let data_norm = data.iter().map(|x| x.value).collect::<Vec<f64>>();

            let cdf = calc_cdf(data_norm.as_slice());
            let q = 0.97;
            let ps = cdf[(q * cdf.len() as f64) as usize];
            let median = cdf[(0.5 * cdf.len() as f64) as usize].0;
            let q1 = cdf[(0.25 * cdf.len() as f64) as usize].0;
            let q3 = cdf[(0.75 * cdf.len() as f64) as usize].0;
            let p90 = cdf[(0.9 * cdf.len() as f64) as usize].0;
            let p99 = cdf[(0.99 * cdf.len() as f64) as usize].0;
            let iqr = q3 - q1;
            println!(
                "current idx: {}, next idx: {}, median: {}, quantile {}: {:?}, q1: {}, q3: {}, iqr: {}, p90: {}, p99: {}",
                start_current_drift_idx, end_next_drift_idx, median, q, ps.0, q1, q3, iqr, p90, p99
            );

            let start_group = rows[start_current_drift_idx].group;
            let end_group = rows[end_next_drift_idx].group;
            file.write(
                format!(
                    "{},{},{},{}\n",
                    start_current_drift_idx, end_next_drift_idx, start_group, end_group
                )
                .as_bytes(),
            )
            .unwrap();
            possible_drifts.push((drift_idx, next_drift_idx));

            if (next_drift_idx > 1000) {
                // process::exit(0);
                break;
            }
        }
        // process::exit(0);
        break;
    }
    file.flush()?;

    // process::exit(0);

    println!("Stationary drifts: {:?}", stationary_drifts.len());

    let drifts_dir = output_dir.join("drifts");
    fs::create_dir_all(&drifts_dir)?;
    let pbar = ProgressBar::new(possible_drifts.len() as u64);
    pbar.set_style(default_pbar_style()?);
    pbar.set_message("Writing drifts to file");
    possible_drifts
        .par_iter_mut()
        .progress_with(pbar)
        .for_each(|(drift_idx, next_drift_idx)| {
            // println!(
            //     "DRIFT IDX: {}, NEXT DRIFT IDX: {}",
            //     drift_idx, next_drift_idx
            // );

            let start_current_drift_idx = *drift_idx - stationary_threshold;
            let end_current_drift_idx = *drift_idx;
            let start_next_drift_idx = *next_drift_idx - stationary_threshold;
            let end_next_drift_idx = *next_drift_idx;

            // if (end_next_drift_idx - start_current_drift_idx) < drift_threshold {
            //     continue;
            // }

            // let current_drift_rows = &rows[start_current_drift_idx..=end_current_drift_idx];
            // let next_drift_rows = &rows[start_next_drift_idx..=end_next_drift_idx];

            let all_rows = &rows[start_current_drift_idx..=end_next_drift_idx];

            let drift_file_path = drifts_dir.join(format!(
                "{}_{}.dat",
                start_current_drift_idx, end_next_drift_idx
            ));
            let mut drift_file = fs::File::create(drift_file_path).unwrap();

            for row in all_rows.iter() {
                drift_file
                    .write(format!("{}\t{}\n", row.idx, row.value).as_bytes())
                    .unwrap();
            }
        });

    let duration = start.elapsed();
    println!("Time elapsed in main() is: {:?}", duration);

    Ok(())
}
