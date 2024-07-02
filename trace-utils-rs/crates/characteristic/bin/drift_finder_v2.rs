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

    // Stability Drift Threshold
    #[clap(short = 's', long = "stability-threshold", default_value = "30")]
    stability_threshold: usize,

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
    stability_group: usize,
    stability: usize,
    group: usize,
    prev_group: usize,
    mode: BTreeMap<usize, usize>,      // rolling window
    next_mode: BTreeMap<usize, usize>, // rolling window
}

impl Row {
    pub fn header() -> String {
        "idx\tvalue\tstability_group\tstability\tgroup\tprev_group\tmode\tnext_mode\n".to_string()
    }

    pub fn to_string(&self) -> String {
        format!(
            "{}\t{}\t{}\t{}\t{}\t{}\t{:?}\t{:?}\n",
            self.idx,
            self.value,
            self.stability_group,
            self.stability,
            self.group,
            self.prev_group,
            self.mode,
            self.next_mode
        )
    }
}

fn determine_group(value: f64, threshold: f64, offset: f64) -> usize {
    match value {
        v if v < 0.0 => 0,
        v if v <= offset => 1,
        _ => ((value - offset) / threshold).floor() as usize + 2,
    }
}

fn mode<T: Hash + Eq + Default + Copy + Ord>(values: &[T]) -> BTreeMap<T, usize> {
    values.iter().fold(BTreeMap::new(), |mut acc, &x| {
        *acc.entry(x).or_default() += 1;
        acc
    })
    // let mut counts = BTreeMap::<T, usize>::new();
    // for &value in values {
    //     *counts.entry(value).or_insert(0) += 1;
    // }
    // counts
}

fn get_similar_modes(
    mode: &BTreeMap<usize, usize>,
    criteria: usize,
    diff: usize,
    include_self: bool,
) -> BTreeMap<usize, usize> {
    let mut similar_modes = BTreeMap::<usize, usize>::new();
    let mut diffs: Vec<i32> = Vec::new();
    if include_self {
        diffs.push(0);
    }
    for i in 1..=diff {
        diffs.push(i as i32);
        diffs.push(-(i as i32));
    }

    for &d in diffs.iter() {
        let key = if criteria as i32 + d < 0 {
            0
        } else {
            (criteria as i32 + d) as usize
        };
        let value = mode.get(&key).unwrap_or(&0);
        similar_modes.insert(key, *value);
    }

    similar_modes
}

fn lookback_consecutive_drift_idx(
    rows: &[Row],
    drift_idx: usize,
    key: &str,
    criteria: usize,
) -> usize {
    for i in (0..=drift_idx).rev() {
        if i == 0 {
            break;
        }

        let (attr, prev_attr) = match key {
            "group" => (rows[i].group, rows[i - 1].group),
            "stability_group" => (rows[i].stability_group, rows[i - 1].stability_group),
            "stability" => (rows[i].stability, rows[i - 1].stability),
            _ => panic!("Invalid key"),
        };

        if attr != criteria && prev_attr == criteria {
            continue;
        }
        if attr == criteria && prev_attr != criteria {
            return i;
        }
    }

    drift_idx
}

fn lookback_nearest_drift_idx(rows: &[Row], drift_idx: usize, key: &str, criteria: usize) -> usize {
    for i in (0..=drift_idx).rev() {
        if i == 0 {
            break;
        }

        if match key {
            "group" => rows[i].group,
            "stability_group" => rows[i].stability_group,
            "stability" => rows[i].stability,
            _ => panic!("Invalid key"),
        } == criteria
        {
            return i;
        }
    }

    drift_idx
}

// def lookahead_nearest_drift_idx(rows: List[Row], drift_idx: int, key: str, criteria: int) -> int:
//     new_drift_idx = drift_idx
//     for i in range(new_drift_idx + 1, len(rows)):
//         if getattr(rows[i], key) == criteria:
//             new_drift_idx = i
//             break

//     return new_drift_idx

fn lookahead_nearest_drift_idx(
    rows: &[Row],
    drift_idx: usize,
    key: &str,
    criteria: usize,
) -> usize {
    for i in drift_idx + 1..rows.len() {
        if match key {
            "group" => rows[i].group,
            "stability_group" => rows[i].stability_group,
            "stability" => rows[i].stability,
            _ => panic!("Invalid key"),
        } == criteria
        {
            return i;
        }
    }

    drift_idx
}

fn lookahead_consecutive_drift_idx(
    rows: &[Row],
    drift_idx: usize,
    key: &str,
    criteria: usize,
) -> usize {
    for i in drift_idx..rows.len() {
        if i == rows.len() - 1 {
            break;
        }

        let (attr, next_attr) = match key {
            "group" => (rows[i].group, rows[i + 1].group),
            "stability_group" => (rows[i].stability_group, rows[i + 1].stability_group),
            "stability" => (rows[i].stability, rows[i + 1].stability),
            _ => panic!("Invalid key"),
        };

        if attr != criteria && next_attr == criteria {
            continue;
        }

        if attr == criteria && next_attr != criteria {
            return i;
        }
    }

    drift_idx
}

fn get_highest_mode(mode: &BTreeMap<usize, usize>) -> i32 {
    let val = mode.into_iter().max_by_key(|x| x.1).unwrap_or((&0, &0));
    if *val.1 == 0 {
        -1
    } else {
        *val.0 as i32
    }
}

fn collapse_intervals(intervals: &[(usize, usize)], threshold: usize) -> Vec<(usize, usize)> {
    if intervals.is_empty() {
        return Vec::new();
    }

    let mut intervals = intervals.to_vec();
    intervals.sort_by_key(|x| x.0);
    let mut collapsed = vec![intervals[0]];

    for &(start, end) in intervals.iter().skip(1) {
        let last_end = collapsed.last().unwrap().1;

        if start - last_end <= threshold {
            collapsed.last_mut().unwrap().1 = end;
        } else {
            collapsed.push((start, end));
        }
    }

    collapsed
}

fn process<P: AsRef<Path>>(rows: &[Row], output_dir: P) {
    let potential_end_concepts: Vec<usize> = rows
        .windows(2)
        .enumerate()
        .filter_map(|(i, values)| {
            let (c, n) = (&values[0], &values[1]);

            if c.stability >= 30 && n.stability == 0 {
                Some(c.idx)
            } else {
                None
            }
        })
        .collect();

    let mut possible_recurring_drifts = Vec::<(usize, usize)>::new();
    let mut possible_other_drifts = Vec::<(usize, usize)>::new();

    println!("{:?}", potential_end_concepts);
}

fn main() -> Result<(), Box<dyn Error>> {
    let start = std::time::Instant::now();

    let args = Args::parse();
    let input = Path::new(&args.input);
    let output_dir = Path::new(&args.output);
    let stability_threshold = args.stability_threshold;
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

    let num_data = data.len();

    let mut raw_rows = Vec::<Row>::with_capacity(num_data);
    let g = determine_group(data[0], group_threshold, group_offset);
    raw_rows.push(Row {
        idx: 0,
        value: data[0],
        stability: 0,
        stability_group: 0,
        group: g,
        prev_group: g,
        mode: BTreeMap::new(),
        next_mode: BTreeMap::new(),
    });
    for (current_idx, &current_value) in data.iter().skip(1).enumerate() {
        let current_idx = current_idx + 1;
        let prev_idx = current_idx - 1;
        let prev_row = &raw_rows[prev_idx];
        let g = determine_group(current_value, group_threshold, group_offset);
        let mut row = Row {
            idx: current_idx,
            value: current_value,
            stability: prev_row.stability,
            stability_group: 0,
            group: g,
            prev_group: prev_row.group,
            mode: BTreeMap::new(),
            next_mode: BTreeMap::new(),
        };
        let prev_windows =
            &raw_rows[(current_idx as i64 - rolling_window as i64).max(0) as usize..=prev_idx];
        let m = mode(
            &prev_windows
                .into_iter()
                .map(|x| x.group)
                .collect::<Vec<_>>(),
        );
        row.mode = m;
        raw_rows.push(row);
    }

    let rows: &mut [Row] = &mut raw_rows;

    for i in 1..num_data {
        let (prev_windows, next_windows) = rows.split_at_mut(i);
        let next_rolling_windows = &next_windows[..rolling_window.min(next_windows.len())];
        let next_groups = next_rolling_windows
            .iter()
            .map(|x| x.group)
            .collect::<Vec<_>>();
        let row = &mut next_windows[0];
        let prev_row = &prev_windows.last().unwrap();
        let m = mode(&next_groups);
        row.next_mode = m;

        let curr_total_mode = row.mode.values().sum::<usize>();
        let &curr_mode_group = row.mode.get(&row.group).unwrap_or(&0);
        let &next_mode_group = row.next_mode.get(&row.group).unwrap_or(&0);

        if curr_total_mode < rolling_window {
            row.stability = prev_row.stability + 1
        } else {
            let half_rolling_window = rolling_window / 2;
            if curr_mode_group > half_rolling_window && next_mode_group > half_rolling_window {
                row.stability = prev_row.stability + 1;
            } else {
                let &prev_group_mode = prev_row.mode.get(&prev_row.group).unwrap_or(&0);
                let potential_modes = get_similar_modes(&row.mode, row.group, 2, false);
                let sum_potential_modes = potential_modes.values().sum::<usize>();
                if prev_group_mode > half_rolling_window
                    || sum_potential_modes > half_rolling_window
                {
                    row.stability = prev_row.stability + 1;
                } else {
                    row.stability = 0;
                }
            }
        }

        if row.group != prev_row.group {
            row.stability_group = 0;
        } else {
            row.stability_group = prev_row.stability_group + 1;
        }
    }

    let mut rows_file = fs::File::create(output_dir.join("row.dat"))?;
    rows_file.write(Row::header().as_bytes())?;
    for i in 0..num_data {
        rows_file.write(rows[i].to_string().as_bytes())?;
    }

    process(rows, output_dir);

    let duration = start.elapsed();
    println!("Time elapsed in main() is: {:?}", duration);

    Ok(())
}
