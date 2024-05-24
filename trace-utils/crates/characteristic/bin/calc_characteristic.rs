use clap::Parser;
use clio_characteristic::characteristic::RawTraceCharacteristic;
use clio_utils::msft::{msft_csv_reader_builder, MsftTrace};
use clio_utils::trace_reader::{TraceReaderBuilder, TraceReaderTrait};
use dashmap::DashMap;
use globwalk::glob;
use indicatif::{ParallelProgressIterator, ProgressBar, ProgressStyle};

use rayon::prelude::*;
use std::error::Error;
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

// OLD CODE without duration
// fn main() -> Result<(), Box<dyn Error>> {
//     let start = std::time::Instant::now();

//     let args = Args::parse();
//     let matcher = format!("{}/**/*.{{tar.gz,gz,csv,tgz}}", args.input);
//     let output_dir = std::path::Path::new(&args.output);
//     // let window = fundu::parse_duration(&args.window)?;

//     let files = glob(&matcher)?;
//     let mut files = files.map(|f| f.unwrap()).collect::<Vec<_>>();
//     files.sort_by(|a, b| natord::compare(&a.path().to_string_lossy(), &b.path().to_string_lossy()));

//     let mut characteristics: Vec<RawTraceCharacteristic> = files
//         .into_par_iter()
//         .map(|entry| {
//             let path = entry.path().to_path_buf();
//             let mut tencent_trace = TraceReaderBuilder::new(&path).unwrap();
//             tencent_trace.with_csv_builder(|csv_builder| msft_csv_reader_builder(csv_builder));
//             let mut traces = Vec::new();
//             if let Err(err) = tencent_trace.read(|record: &csv::StringRecord| {
//                 let record: MsftTrace = record.try_into()?;
//                 traces.push(record);
//                 Ok(())
//             }) {
//                 println!("error: {}", err);
//                 process::exit(1);
//             }

//             traces.try_into().unwrap()
//         })
//         .collect();

//     characteristics.sort_by(|a, b| a.start_time.partial_cmp(&b.start_time).unwrap());

//     println!("Writing characteristic csv file");
//     let output_file = output_dir.join("characteristic.csv");
//     let output_file = std::fs::File::create(output_file)?;
//     let mut writer = csv::Writer::from_writer(output_file);
//     writer.write_byte_record(&RawTraceCharacteristic::csv_header())?;
//     for characteristic in &characteristics {
//         let record: csv::ByteRecord = characteristic.try_into()?;
//         writer.write_byte_record(&record)?;
//     }

//     // println!("Writing characteristic json file");
//     // let output_file = output_dir.join("characteristic.json");
//     // let output_file = std::fs::File::create(output_file)?;
//     // serde_json::to_writer(output_file, &characteristics)?;

//     let duration = start.elapsed();
//     println!("Time elapsed in main() is: {:?}", duration);

//     Ok(())
// }

fn default_pbar_style() -> Result<ProgressStyle, Box<dyn Error>> {
    let pbar = ProgressStyle::default_bar()
        .template("[{elapsed_precise}] {wide_bar} {pos}/{len} {msg}")?
        .progress_chars("=> ");
    Ok(pbar)
}

fn main() -> Result<(), Box<dyn Error>> {
    let start = std::time::Instant::now();

    let args = Args::parse();
    let matcher = format!("{}/**/*.{{tar.gz,gz,csv,tgz}}", args.input);
    let output_dir = std::path::Path::new(&args.output);
    let window = fundu::parse_duration(&args.window)?;
    let window_duration = window.as_millis() as f64;
    let files = glob(&matcher)?;
    let mut files = files.map(|f| f.unwrap()).collect::<Vec<_>>();
    files.sort_by(|a, b| natord::compare(&a.path().to_string_lossy(), &b.path().to_string_lossy()));

    // let progress_bar = MultiProgress::new();
    let window_map: DashMap<u64, Vec<MsftTrace>> = DashMap::new();
    println!("Reading traces");
    let pbar = ProgressBar::new(files.len() as u64);
    pbar.set_style(default_pbar_style()?);
    pbar.set_message("Reading traces");
    files
        .par_iter()
        .progress_with(pbar)
        // .with_message("Reading traces")
        .for_each(|entry| {
            let path = entry.path().to_path_buf();
            let mut tencent_trace = TraceReaderBuilder::new(&path).unwrap();
            tencent_trace.with_csv_builder(|csv_builder| msft_csv_reader_builder(csv_builder));

            if let Err(err) = tencent_trace.read(|record: &csv::StringRecord| {
                let record: MsftTrace = record.try_into()?;
                let time = record.timestamp; // in ms

                let chunk = (time as f64 / window_duration).floor() as u64;
                let mut entry = window_map.entry(chunk).or_insert_with(Vec::new);
                entry.push(record);

                Ok(())
            }) {
                println!("error: {}", err);
                process::exit(1);
            }
        });

    let window_len = window_map.len();
    println!("Found {} of windows", window_len);

    println!("Sorting traces");
    let pbar = ProgressBar::new(window_len as u64);
    pbar.set_style(default_pbar_style()?);
    pbar.set_message("Sorting traces");

    window_map
        .iter_mut()
        .par_bridge()
        .progress_with(pbar)
        .for_each(|mut entry| {
            entry
                .value_mut()
                .sort_by(|a, b| a.timestamp.partial_cmp(&b.timestamp).unwrap());
        });

    println!("Calculating characteristics");
    let pbar = ProgressBar::new(window_len as u64);
    pbar.set_style(default_pbar_style()?);
    pbar.set_message("Calculating characteristics");

    let characteristics: Vec<RawTraceCharacteristic> = window_map
        .iter()
        .par_bridge()
        .progress_with(pbar)
        .map(|entry| {
            let traces = entry.value();
            let characteristic: RawTraceCharacteristic = traces
                .try_into()
                .map_err(|e| {
                    println!("error: {}", e);
                    process::exit(1);
                })
                .unwrap();
            characteristic
        })
        .collect();

    {
        println!("Writing characteristic csv file");
        let output_file = output_dir.join("characteristic.csv");
        let output_file = std::fs::File::create(output_file)?;
        let mut writer = csv::Writer::from_writer(output_file);
        writer.write_byte_record(&RawTraceCharacteristic::csv_header())?;
        let pbar = ProgressBar::new(characteristics.len() as u64);
        pbar.set_style(default_pbar_style()?);
        pbar.set_message("Writing characteristic.csv");
        for characteristic in &characteristics {
            let record: csv::ByteRecord = characteristic.try_into()?;
            writer.write_byte_record(&record)?;
            pbar.inc(1);
        }
        pbar.finish();
        let duration = start.elapsed();
        println!(
            "Time elapsed in writing characteristic.csv is: {:?}",
            duration
        );
    }

    // println!("Writing characteristic json file");
    // let output_file = output_dir.join("characteristic.json");
    // let output_file = std::fs::File::create(output_file)?;
    // serde_json::to_writer(output_file, &characteristics)?;

    let duration = start.elapsed();
    println!("Time elapsed in main() is: {:?}", duration);

    Ok(())
}