use clap::Parser;
// use clio_utils::cdf::calc_cdf;
use clio_utils::file::BufReader;
use clio_utils::pbar::default_pbar_style;
use indicatif::{ParallelProgressIterator, ProgressBar};
use rayon::prelude::*;
use std::collections::BTreeMap;
use std::error::Error;
use std::hash::Hash;
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
    #[clap(short = 'w', long = "rolling-window", default_value = "10")]
    rolling_window: usize,
}

#[allow(dead_code)]
#[derive(Debug)]
enum DriftType {
    Sudden(usize, usize),
    Incremental(usize, usize),
    Recurring(usize, usize),
    Gradual(usize, usize),
}

#[derive(Debug)]
struct Row {
    idx: usize,
    value: f64,
    diff: f64, // diff between current and previous value
    increasing_idx: i64,
    decreasing_idx: i64,
    stability_group: usize,
    stability: usize,
    group: i32,
    mode: i32,      // rolling window
    next_mode: i32, // rolling window
    max_group: i32, // rolling window
    min_group: i32, // rolling window
}

impl Row {
    pub fn header() -> String {
        "idx\tvalue\tdiff\tincreasing_idx\tdecreasing_idx\tstability\tstability_group\tgroup\tmode\tnext_mode\tmin_group\tmax_group\n".to_string()
    }

    pub fn to_string(&self) -> String {
        format!(
            "{}\t{}\t{}\t{}\t{}\t{}\t{}\t{}\t{}\t{}\t{}\t{}\n",
            self.idx,
            self.value,
            self.diff,
            self.increasing_idx,
            self.decreasing_idx,
            self.stability,
            self.stability_group,
            self.group,
            self.mode,
            self.next_mode,
            self.min_group,
            self.max_group
        )
    }
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

fn mode<T: Hash + Eq + Default + Copy + Ord>(values: &[T]) -> T {
    let mut counts = BTreeMap::<T, usize>::new();
    for &value in values {
        *counts.entry(value).or_insert(0) += 1;
    }

    let mut max_count = 0;
    let mut mode = T::default();
    for (value, count) in counts {
        if count > max_count {
            max_count = count;
            mode = value;
        }
    }

    mode
}

fn main() -> Result<(), Box<dyn Error>> {
    let start = std::time::Instant::now();

    let args = Args::parse();
    let input = Path::new(&args.input);
    let output_dir = Path::new(&args.output);
    let stationary_threshold = args.stationary_threshold - 1;
    let group_threshold = args.group_threshold;
    let group_offset = args.group_offset;
    let drift_threshold = args.drift_threshold;
    let rolling_window = args.rolling_window;

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

    // let avg_data = data.iter().sum::<f64>() / data.len() as f64;
    // let norm_data = data.iter().map(|x| x / avg_data).collect::<Vec<f64>>();
    let diff_data = data
        .windows(2)
        .map(|x| if x[0] > 0.0 { x[1] / x[0] } else { 0.0 })
        .collect::<Vec<f64>>();

    println!("Data length: {}", data.len());

    let mut raw_rows = Vec::<Row>::with_capacity(data.len());
    let g = determine_group(data[0], group_threshold, group_offset);
    raw_rows.push(Row {
        idx: 0,
        value: data[0],
        diff: 0.0,
        // value: norm_data[0],
        increasing_idx: -1,
        decreasing_idx: -1,
        stability: 0,
        stability_group: 0,
        group: g,
        mode: g,
        next_mode: -1,
        min_group: g,
        max_group: g,
    });

    for (current_idx, &current_value) in data.iter().skip(1).enumerate() {
        let current_idx = current_idx + 1;
        let prev_idx = current_idx - 1;
        let prev_row = &raw_rows[prev_idx];
        let g = determine_group(current_value, group_threshold, group_offset);
        let mut row = Row {
            idx: current_idx,
            value: current_value,
            diff: diff_data[prev_idx],
            increasing_idx: prev_row.increasing_idx,
            decreasing_idx: prev_row.decreasing_idx,
            stability: prev_row.stability,
            stability_group: 0,
            group: g,
            mode: g,
            next_mode: -1,
            max_group: g,
            min_group: g,
        };
        let prev_windows =
            &raw_rows[(current_idx as i64 - rolling_window as i64).max(0) as usize..=prev_idx];
        let m = mode(
            &prev_windows
                .into_iter()
                .map(|x| x.group)
                .collect::<Vec<i32>>(),
        );
        row.mode = m;

        if m < prev_row.mode {
            row.increasing_idx = prev_idx as i64;
            row.stability = 0;
        } else if m > prev_row.mode {
            row.decreasing_idx = prev_idx as i64;
            row.stability = 0;
        } else {
            row.stability = prev_row.stability + 1;
        }

        // println!(
        //     "Current group {}, prev group {} ({})",
        //     row.group, prev_row.group, prev_row.stability_group
        // );
        if row.group != prev_row.group {
            row.stability_group = 0;
        } else {
            row.stability_group = prev_row.stability_group + 1;
        }

        raw_rows.push(row);
    }

    let mut rows_file = fs::File::create(output_dir.join("raw.dat"))?;
    rows_file.write(Row::header().as_bytes())?;

    let mut rows = Vec::<Row>::with_capacity(data.len());
    let g = determine_group(data[0], group_threshold, group_offset);
    let row = Row {
        idx: 0,
        value: data[0],
        diff: 0.0,
        // value: norm_data[0],
        increasing_idx: -1,
        decreasing_idx: -1,
        stability: 0,
        stability_group: 0,
        group: g,
        mode: g,
        next_mode: -1,
        min_group: g,
        max_group: g,
    };
    rows_file.write(row.to_string().as_bytes())?;
    rows.push(row);

    for (current_idx, &current_value) in data.iter().skip(1).enumerate() {
        let current_idx = current_idx + 1;
        let prev_idx = current_idx - 1;
        let prev_windows =
            &raw_rows[(current_idx as i64 - rolling_window as i64).max(0) as usize..=prev_idx];
        let prev_row = &raw_rows[prev_idx];
        let next_windows = &raw_rows[current_idx as usize
            ..=(current_idx as usize + rolling_window as usize).min(data.len() - 1) as usize];
        let g = determine_group(current_value, group_threshold, group_offset);
        let mut row = Row {
            idx: current_idx,
            value: current_value,
            diff: diff_data[prev_idx],
            increasing_idx: prev_row.increasing_idx,
            decreasing_idx: prev_row.decreasing_idx,
            stability: prev_row.stability,
            stability_group: 0,
            group: g,
            mode: 0,
            next_mode: -1,
            max_group: g,
            min_group: g,
        };
        let mut prev_groups = prev_windows
            .into_iter()
            .map(|x| x.group)
            .collect::<Vec<i32>>();
        let next_groups = next_windows
            .into_iter()
            .map(|x| x.group)
            .collect::<Vec<i32>>();
        prev_groups.push(row.group);

        let m = mode(&prev_groups);
        let next_m = mode(&next_groups);
        row.mode = m;
        row.next_mode = next_m;
        row.min_group = prev_groups.iter().min().unwrap().clone();
        row.max_group = prev_groups.iter().max().unwrap().clone();

        if m < prev_row.mode {
            row.increasing_idx = prev_idx as i64;
            row.stability = 0;
        } else if m > prev_row.mode {
            row.decreasing_idx = prev_idx as i64;
            row.stability = 0;
        } else {
            row.stability = prev_row.stability + 1;
        }

        // println!(
        //     "Current group {}, prev group {} ({})",
        //     row.group, prev_row.group, prev_row.stability_group
        // );
        if row.group != prev_row.group {
            row.stability_group = 0;
        } else {
            row.stability_group = prev_row.stability_group + 1;
        }

        rows_file.write(row.to_string().as_bytes())?;
        rows.push(row);
    }

    let (num_increase_drifts, num_decrease_drifts) = count_drifts(&rows);
    println!("Number of increasing drifts: {}", num_increase_drifts);
    println!("Number of decreasing drifts: {}", num_decrease_drifts);

    // find index where stability is equal to stationary_threshold
    let mut stationary_drifts_file = fs::File::create(output_dir.join("stationary_drifts.csv"))?;
    stationary_drifts_file.write(b"idx,stability,start\n")?;
    let mut stationary_drifts = Vec::<usize>::new();
    for (row, next_row) in rows.iter().zip(rows.iter().skip(1)) {
        if row.stability >= stationary_threshold && next_row.stability == 0 {
            stationary_drifts.push(row.idx);
            stationary_drifts_file.write(format!("{},{}\n", row.idx, row.stability).as_bytes())?;
        }
    }
    stationary_drifts_file.flush()?;

    // generate possible drift start and index
    // assume we have stationary drifts as follow [1, 5, 10, 15, 20]
    // mean we will generate [(1, 5), (1, 10), (1, 15), (1, 20), (5, 10), (5, 15), ... ] as possible drifts
    // cartesian product of stationary drifts
    let mut recurring_map = BTreeMap::<usize, usize>::new();
    let mut possible_drifts = Vec::<DriftType>::new();
    let mut possible_drifts_file = fs::File::create(output_dir.join("possible_drifts.csv"))?;
    possible_drifts_file.write(b"start,end,type\n")?;
    for (idx, &drift_idx) in stationary_drifts.iter().enumerate() {
        for &next_drift_idx in stationary_drifts.iter().skip(idx + 1) {
            let start_current_drift_idx = drift_idx;
            let end_next_drift_idx = next_drift_idx;
            let start_row = &rows[start_current_drift_idx];
            let mut start_current_drift_idx = start_row.idx - start_row.stability;
            let start_row = &rows[start_current_drift_idx];

            for (i, next_i) in (0..=start_current_drift_idx)
                .rev()
                .zip((0..start_current_drift_idx).rev())
            {
                // println!("I: {}, Next I: {}", i, next_i);
                if rows[i].group == start_row.group && rows[next_i].group != start_row.group {
                    start_current_drift_idx = i;
                    break;
                }
            }

            let start_row = &rows[start_current_drift_idx];
            let end_row = &rows[end_next_drift_idx];
            let end_next_drift_idx = end_row.idx;

            if (end_next_drift_idx - start_current_drift_idx) < drift_threshold {
                continue;
            }

            let start_mode = start_row.mode;
            let end_mode = end_row.mode;
            let start_group = start_row.group;
            let end_group = end_row.group;

            let data = &rows[start_current_drift_idx..=end_next_drift_idx];

            println!(
                "Start current drift idx: {}({}), End next drift idx: {}({})",
                start_current_drift_idx, start_mode, end_next_drift_idx, end_mode
            );

            // if next_drift_idx > 300 {
            //     process::exit(0);
            //     // break;
            // }

            // if start_current_drift_idx == 164 && end_next_drift_idx > 400 {
            //     process::exit(0);
            // }

            if start_mode == end_mode {
                // TODO: potentially recurring drift and gradual drift
                // treat everything as recurring drift for now
                {
                    if recurring_map.contains_key(&start_current_drift_idx) {
                        continue;
                    }

                    let mut new_end_current_drift_idx = end_next_drift_idx;
                    if start_group != end_group {
                        for i in (start_current_drift_idx..end_next_drift_idx).rev() {
                            // find the last drift that has stability > 0 and mode == start_mode
                            if rows[i].stability > 0
                                && rows[i].group == start_group
                                && (rows[i].stability_group as usize) > stationary_threshold
                            {
                                new_end_current_drift_idx = i;
                                println!(
                                    "New end current drift idx: {}",
                                    new_end_current_drift_idx
                                );
                                break;
                            }
                        }
                    }

                    println!("A Is recurring: {} == {}", start_group, end_group);
                    let end_group = rows[new_end_current_drift_idx].group;
                    println!("B Is recurring: {} == {}", start_group, end_group);
                    if start_group == end_group {
                        possible_drifts.push(DriftType::Recurring(
                            start_current_drift_idx,
                            new_end_current_drift_idx,
                        ));
                        recurring_map.insert(start_current_drift_idx, new_end_current_drift_idx);
                        possible_drifts_file.write(
                            format!(
                                "{},{},{}\n",
                                start_current_drift_idx, new_end_current_drift_idx, "recurring"
                            )
                            .as_bytes(),
                        )?;
                    }
                }
            } else {
                // potentially sudden drift, incremental drift, or gradual drift

                // here we check the intermediate modes between the start and end drift
                let mut is_increasing = true; // assumption
                let mut is_decreasing = true; // assumption
                #[allow(unused_variables)]
                let mut number_of_stationary_between = 0;
                let mut modes = Vec::new();
                let mut diffs = Vec::new();

                for (row, next_row) in data.iter().zip(data.iter().skip(1)) {
                    modes.push(row.mode);
                    diffs.push(next_row.diff - row.diff);
                    if row.mode == start_mode {
                        number_of_stationary_between += 1;
                    }

                    let diff = (next_row.mode - row.mode).abs();

                    if is_increasing && next_row.mode > row.mode && diff >= 1 {
                        is_increasing = true;
                        continue;
                    } else if is_increasing && next_row.mode <= row.mode {
                        is_increasing = false;
                        continue;
                    }

                    if is_decreasing && next_row.mode < row.mode && diff >= 1 {
                        is_decreasing = true;
                        continue;
                    } else if is_decreasing && next_row.mode >= row.mode {
                        is_decreasing = false;
                        continue;
                    }
                }

                if is_increasing || is_decreasing {
                    // println!(
                    //     "Incremental, start: {}, end: {}",
                    //     start_current_drift_idx, end_next_drift_idx
                    // );
                    possible_drifts.push(DriftType::Incremental(
                        start_current_drift_idx,
                        end_next_drift_idx,
                    ));
                    possible_drifts_file.write(
                        format!(
                            "{},{},{}\n",
                            start_current_drift_idx, end_next_drift_idx, "incremental"
                        )
                        .as_bytes(),
                    )?;
                } else {
                    if recurring_map.contains_key(&start_current_drift_idx) {
                        continue;
                    }

                    let mut new_end_current_drift_idx = end_next_drift_idx;
                    let end_group = rows[end_next_drift_idx].group;
                    // println!(
                    //     "End next drift idx: {}, len data: {}",
                    //     end_next_drift_idx,
                    //     data.len()
                    // );
                    for (i, next_i) in (new_end_current_drift_idx..rows.len())
                        .zip(new_end_current_drift_idx + 1..rows.len())
                    {
                        // println!("I: {}, Next I: {}", i, next_i);
                        // find the last drift that has stability > 0 and mode == start_mode
                        if rows[i].group != end_group {
                            break;
                        }
                        if rows[i].group == end_group && rows[next_i].group != end_group {
                            new_end_current_drift_idx = i;
                            break;
                        }
                    }

                    let end_group = rows[new_end_current_drift_idx].group;

                    if (end_group - start_group).abs() < 2 {
                        continue;
                    }

                    println!(
                        "Sudden, start: {}, end: {}, new end: {}",
                        start_current_drift_idx, end_next_drift_idx, new_end_current_drift_idx
                    );
                    // if number_of_stationary_between < 1 {
                    possible_drifts.push(DriftType::Sudden(
                        start_current_drift_idx,
                        new_end_current_drift_idx,
                    ));
                    possible_drifts_file.write(
                        format!(
                            "{},{},{}\n",
                            start_current_drift_idx, new_end_current_drift_idx, "sudden"
                        )
                        .as_bytes(),
                    )?;
                }
            }
        }
    }

    possible_drifts_file.flush()?;

    // sort the possible drifts by both start and end index
    possible_drifts.sort_by(|a, b| match (a, b) {
        (DriftType::Sudden(a_start, a_end), DriftType::Sudden(b_start, b_end)) => {
            a_start.cmp(b_start).then(a_end.cmp(b_end))
        }
        (DriftType::Incremental(a_start, a_end), DriftType::Incremental(b_start, b_end)) => {
            a_start.cmp(b_start).then(a_end.cmp(b_end))
        }
        (DriftType::Recurring(a_start, a_end), DriftType::Recurring(b_start, b_end)) => {
            a_start.cmp(b_start).then(a_end.cmp(b_end))
        }
        (DriftType::Gradual(a_start, a_end), DriftType::Gradual(b_start, b_end)) => {
            a_start.cmp(b_start).then(a_end.cmp(b_end))
        }
        (DriftType::Sudden(a_start, a_end), DriftType::Incremental(b_start, b_end)) => {
            a_start.cmp(b_start).then(a_end.cmp(b_end))
        }
        (DriftType::Sudden(a_start, a_end), DriftType::Recurring(b_start, b_end)) => {
            a_start.cmp(b_start).then(a_end.cmp(b_end))
        }
        (DriftType::Sudden(a_start, a_end), DriftType::Gradual(b_start, b_end)) => {
            a_start.cmp(b_start).then(a_end.cmp(b_end))
        }
        (DriftType::Incremental(a_start, a_end), DriftType::Sudden(b_start, b_end)) => {
            a_start.cmp(b_start).then(a_end.cmp(b_end))
        }
        (DriftType::Incremental(a_start, a_end), DriftType::Recurring(b_start, b_end)) => {
            a_start.cmp(b_start).then(a_end.cmp(b_end))
        }
        (DriftType::Incremental(a_start, a_end), DriftType::Gradual(b_start, b_end)) => {
            a_start.cmp(b_start).then(a_end.cmp(b_end))
        }
        (DriftType::Recurring(a_start, a_end), DriftType::Sudden(b_start, b_end)) => {
            a_start.cmp(b_start).then(a_end.cmp(b_end))
        }
        (DriftType::Recurring(a_start, a_end), DriftType::Incremental(b_start, b_end)) => {
            a_start.cmp(b_start).then(a_end.cmp(b_end))
        }
        (DriftType::Recurring(a_start, a_end), DriftType::Gradual(b_start, b_end)) => {
            a_start.cmp(b_start).then(a_end.cmp(b_end))
        }
        (DriftType::Gradual(a_start, a_end), DriftType::Sudden(b_start, b_end)) => {
            a_start.cmp(b_start).then(a_end.cmp(b_end))
        }
        (DriftType::Gradual(a_start, a_end), DriftType::Incremental(b_start, b_end)) => {
            a_start.cmp(b_start).then(a_end.cmp(b_end))
        }
        (DriftType::Gradual(a_start, a_end), DriftType::Recurring(b_start, b_end)) => {
            a_start.cmp(b_start).then(a_end.cmp(b_end))
        } // _ => std::cmp::Ordering::Equal,
    });

    // process::exit(0);

    let mut file = fs::File::create(output_dir.join("drifts.csv"))?;
    file.write(b"start,end,type\n")?;

    let mut cleaned_drifts = Vec::new();
    let mut temp_current_start = 0;
    for (current, next) in possible_drifts.iter().zip(possible_drifts.iter().skip(1)) {
        // println!(
        //     "Temp: {}, Current: {:?}, Next: {:?}",
        //     temp_current_start, current, next
        // );
        let (current_name, &current_start, &current_end) = match current {
            DriftType::Sudden(start, end) => ("sudden", start, end),
            DriftType::Incremental(start, end) => ("incremental", start, end),
            DriftType::Recurring(start, end) => ("recurring", start, end),
            DriftType::Gradual(start, end) => ("gradual", start, end),
        };

        let (next_name, &next_start, &_next_end) = match next {
            DriftType::Sudden(start, end) => ("sudden", start, end),
            DriftType::Incremental(start, end) => ("incremental", start, end),
            DriftType::Recurring(start, end) => ("recurring", start, end),
            DriftType::Gradual(start, end) => ("gradual", start, end),
        };

        if temp_current_start == current_start
            && current_start == next_start
            && current_name == next_name
        {
            continue;
        }

        if temp_current_start != next_start {
            temp_current_start = next_start;
        }

        // println!("==== Adding drift: {:?}", current);
        cleaned_drifts.push(current);

        file.write(format!("{},{},{}\n", current_start, current_end, current_name).as_bytes())?;
        // if current_start > 300 {
        //     break;
        // }
    }

    file.flush()?;

    // process::exit(0);

    println!("Stationary drifts: {:?}", stationary_drifts.len());

    let drifts_dir = output_dir.join("drifts");
    fs::create_dir_all(&drifts_dir)?;
    let pbar = ProgressBar::new(cleaned_drifts.len() as u64);
    pbar.set_style(default_pbar_style()?);
    pbar.set_message("Writing drifts to file");
    cleaned_drifts
        .par_iter()
        .progress_with(pbar)
        .for_each(|drift| {
            let (drift_name, &start_index, &end_index) = match drift {
                DriftType::Sudden(start, end) => ("sudden", start, end),
                DriftType::Incremental(start, end) => ("incremental", start, end),
                DriftType::Recurring(start, end) => ("recurring", start, end),
                DriftType::Gradual(start, end) => ("gradual", start, end),
            };

            let drift_output_dir = drifts_dir.join(drift_name);
            fs::create_dir_all(&drift_output_dir).unwrap();

            // println!(
            //     "Drift: {}, Start: {}, End: {}",
            //     drift_name, start_index, end_index
            // );

            let all_rows = &rows[start_index..=end_index];

            let drift_file_path =
                drift_output_dir.join(format!("{}_{}.dat", start_index, end_index));
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
