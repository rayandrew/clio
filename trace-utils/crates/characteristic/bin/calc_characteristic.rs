use clap::Parser;
use clio_characteristic::characteristic::RawTraceCharacteristic;
use clio_utils::msft::{msft_csv_reader_builder, msft_csv_writer_builder, MsftTrace};
use clio_utils::pbar::default_pbar_style;
use clio_utils::trace_reader::{TraceReaderBuilder, TraceReaderTrait};
use dashmap::DashMap;
use globwalk::glob;
use indicatif::{ParallelProgressIterator, ProgressBar};
use rayon::prelude::*;
use std::collections::HashMap;
use std::error::Error;
use std::process;
use tempfile::tempdir;

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
    let temp_dir = tempdir()?;
    println!("Creating temporary directory: {:?}", temp_dir.path());
    let window_path_map = DashMap::new();
    let window_writer_map = DashMap::new();
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
                let path = window_path_map
                    .entry(chunk)
                    .or_insert_with(|| {
                        let output_path = temp_dir.path().join(format!("{}.unsorted", chunk));
                        output_path
                    })
                    .value()
                    .clone();

                let _ = window_writer_map
                    .entry(chunk)
                    .or_insert_with(|| {
                        let file = std::fs::File::create(&path).unwrap();
                        let mut builder = csv::WriterBuilder::new();
                        let builder = msft_csv_writer_builder(&mut builder);
                        builder.from_writer(file)
                    })
                    .value_mut()
                    .write_byte_record(&record.to_byte_record());
                // *writer.write_byte_record(&record.to_byte_record()).unwrap();
                Ok(())
            }) {
                println!("error: {}", err);
                process::exit(1);
            }
        });

    assert!(
        window_path_map.len() == window_writer_map.len(),
        "window_path_map.len() != window_writer_map.len()"
    );

    window_writer_map
        .iter_mut()
        .par_bridge()
        .for_each(|mut entry| {
            let writer = entry.value_mut();
            writer.flush().unwrap();
        });

    let window_len = window_path_map.len();
    println!("Found {} of windows", window_len);

    println!("Sorting traces");
    let pbar = ProgressBar::new(window_len as u64);
    pbar.set_style(default_pbar_style()?);
    pbar.set_message("Sorting traces");

    let window_path_map = window_path_map
        .into_iter()
        .par_bridge()
        .progress_with(pbar)
        .map(|(chunk, path)| {
            let mut builder = csv::ReaderBuilder::new();
            let builder = msft_csv_reader_builder(&mut builder);
            let mut rdr = builder.from_path(&path).unwrap();
            let mut records = rdr.records().map(|r| r.unwrap()).collect::<Vec<_>>();
            records.sort_by(|a, b| {
                let a = a[0].parse::<f64>().unwrap();
                let b = b[0].parse::<f64>().unwrap();
                a.partial_cmp(&b).unwrap()
            });

            let output_path = path.with_extension("csv");
            let output_file = std::fs::File::create(&output_path).unwrap();
            let mut builder = csv::WriterBuilder::new();
            let builder = msft_csv_writer_builder(&mut builder);
            let mut writer = builder.from_writer(output_file);
            for record in records {
                writer.write_record(&record).unwrap();
            }
            writer.flush().unwrap();
            std::fs::remove_file(path).unwrap();

            // // tar.gz the file
            // let tar_path = path.with_extension("tar.gz");
            // let tar_file = std::fs::File::create(&tar_path).unwrap();
            // let enc = flate2::write::GzEncoder::new(tar_file, flate2::Compression::default());
            // let mut tar = tar::Builder::new(enc);
            // tar.append_path_with_name(&output_path, output_path.file_name().unwrap())
            //     .unwrap();
            // tar.finish().unwrap();

            // std::fs::remove_file(output_path).unwrap();

            (chunk, output_path)
        })
        .collect::<HashMap<_, _>>();

    println!("Calculating characteristics");
    let pbar = ProgressBar::new(window_len as u64);
    pbar.set_style(default_pbar_style()?);
    pbar.set_message("Calculating characteristics");
    let mut characteristics: Vec<(u64, RawTraceCharacteristic)> = window_path_map
        .into_par_iter()
        .progress_with(pbar)
        .map(|(chunk, path)| {
            let mut builder = csv::ReaderBuilder::new();
            let builder = clio_utils::msft::msft_csv_reader_builder(&mut builder);
            let mut rdr = builder.from_path(path).unwrap();
            let traces: Vec<MsftTrace> = rdr
                .byte_records()
                .map(|r| r.unwrap().try_into().unwrap())
                .collect::<Vec<_>>();
            // println!("Trace length: {}", traces.len());
            let characteristic: RawTraceCharacteristic = traces
                .try_into()
                .map_err(|e| {
                    println!("error: {}", e);
                    process::exit(1);
                })
                .unwrap();
            (chunk, characteristic)
        })
        .collect();

    // sort characteristics by chunk
    characteristics.sort_by(|a, b| a.0.partial_cmp(&b.0).unwrap());
    let characteristics: Vec<RawTraceCharacteristic> =
        characteristics.into_iter().map(|(_, c)| c).collect();

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
    temp_dir.close()?;

    let duration = start.elapsed();
    println!("Time elapsed in main() is: {:?}", duration);

    Ok(())
}
