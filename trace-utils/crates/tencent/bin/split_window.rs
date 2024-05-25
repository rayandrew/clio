use clap::Parser;
use clio_utils::pbar::default_pbar_style;
use clio_utils::trace_reader::{TraceReaderBuilder, TraceReaderTrait};
use dashmap::DashMap;
use globwalk::glob;
use indicatif::{ParallelProgressIterator, ProgressBar};
use rand::Rng;
use rayon::prelude::*;

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

fn main() -> Result<(), Box<dyn std::error::Error>> {
    let start = std::time::Instant::now();

    let args = Args::parse();
    let matcher = format!("{}/**/*.{{tar.gz,gz,csv,tgz}}", args.input);
    let output_dir = std::path::Path::new(&args.output);
    let canonical_output_dir =
        std::fs::canonicalize(&output_dir).unwrap_or(output_dir.to_path_buf());
    let window = fundu::parse_duration(&args.window)?;
    let files = glob(&matcher)?;
    let mut files = files.map(|f| f.unwrap()).collect::<Vec<_>>();

    if files.len() == 0 {
        println!("No files found in the data directory");
        std::process::exit(1);
    }

    files.sort_by(|a, b| natord::compare(&a.path().to_string_lossy(), &b.path().to_string_lossy()));

    // create output directory
    std::fs::create_dir_all(output_dir)?;

    // rayon find the start time of the trace
    println!("Finding start time of the trace ...");
    let temp_files = files.clone();
    let pbar = ProgressBar::new(files.len() as u64);
    pbar.set_style(default_pbar_style()?);
    pbar.set_message("Finding start time");
    let trace_start_time = temp_files
        .into_par_iter()
        .progress_with(pbar)
        .map(|entry| {
            let path = entry.path().to_path_buf();
            if path.is_file() {
                let trace = TraceReaderBuilder::new(path).unwrap();
                let mut min_ts = f64::MAX;
                if let Err(err) = trace.read(|record| {
                    let ori_time = record[0].parse::<f64>().unwrap();
                    if ori_time < min_ts {
                        min_ts = ori_time;
                    }
                    Ok(())
                }) {
                    println!("error running trace.read: {}", err);
                    std::process::exit(1);
                }
                min_ts
            } else {
                f64::MAX
            }
        })
        .reduce(|| f64::MAX, |acc, time| if time < acc { time } else { acc });
    let trace_start_time = (trace_start_time * 1e3) as f64; // convert to milliseconds

    println!("Start time of the trace: {:?}", trace_start_time);
    let hashmap_writer: DashMap<u128, csv::Writer<std::fs::File>> = DashMap::new();
    let files_len = files.len();

    // -------------------------------------------------------------------------------
    println!("Splitting trace into windows ...");
    let pbar = ProgressBar::new(files_len as u64);
    pbar.set_style(default_pbar_style()?);
    pbar.set_message("Splitting trace");
    files.into_par_iter().progress_with(pbar).for_each(|entry| {
        let mut prev_jitter = 0.0;
        let mut prev_time = 0.0;
        let mut rng = rand::thread_rng();
        let path = entry.path().to_path_buf();
        if path.is_file() {
            let trace = TraceReaderBuilder::new(path).unwrap();
            if let Err(err) = trace.read(|record| {
                let mut msft_trace =
                    clio_tencent::normalize::normalize_to_msft_from_csv_record(&record)?;
                let time = msft_trace.timestamp - trace_start_time; // in seconds
                let mut ms_time = time;
                if prev_time == time {
                    let jitter = rng.gen_range(prev_jitter + 1.0..prev_jitter + 5.0);
                    ms_time += jitter;
                    prev_jitter = jitter;
                } else {
                    prev_jitter = 0.0;
                }
                prev_time = time;
                msft_trace.timestamp = ms_time;
                let ms_time_r = ((ms_time / 1e3) as u128) * 1e3 as u128;
                let current_chunk = ms_time_r / window.as_millis();

                if let Some(mut writer) = hashmap_writer.get_mut(&current_chunk) {
                    writer.write_byte_record(&msft_trace.to_byte_record())?;
                } else {
                    let chunk_path =
                        canonical_output_dir.join(format!("chunk_{}.temp", current_chunk));
                    let output_file = std::fs::File::create(chunk_path).unwrap();
                    let mut builder = csv::WriterBuilder::new();
                    let builder = clio_utils::msft::msft_csv_writer_builder(&mut builder);
                    let mut writer = builder.from_writer(output_file);
                    writer.write_byte_record(&msft_trace.to_byte_record())?;
                    hashmap_writer.insert(current_chunk, writer);
                }
                Ok(())
            }) {
                println!("error running trace.read: {}", err);
                std::process::exit(1);
            }
        }
    });

    // flush all writers
    hashmap_writer
        .iter_mut()
        .par_bridge()
        .for_each(|mut entry| {
            entry.value_mut().flush().unwrap();
        });
    // -------------------------------------------------------------------------------

    let canonical_output_dir = std::fs::canonicalize(&output_dir)?;
    let output_path = canonical_output_dir.join("chunk_*.temp");
    println!("Chunk path: {:?}", output_path.to_string_lossy());
    let files = glob(output_path.to_string_lossy());
    let mut files = files?.map(|f| f.unwrap()).collect::<Vec<_>>();
    files.sort_by(|a, b| natord::compare(&a.path().to_string_lossy(), &b.path().to_string_lossy()));

    println!("Sorting timestamp in csv files ...");

    let files_len = files.len();
    let pbar = ProgressBar::new(files_len as u64);
    pbar.set_style(default_pbar_style()?);
    pbar.set_message("Sorting timestamp");

    files.into_par_iter().progress_with(pbar).for_each(|entry| {
        let path = entry.path();
        let output_path = path.to_path_buf().with_extension("");
        let tar_output_path = path.to_path_buf().with_extension("tar.gz");
        if path.is_file() {
            let mut builder = csv::ReaderBuilder::new();
            let builder = clio_utils::msft::msft_csv_reader_builder(&mut builder);
            let mut rdr = builder.from_path(&path).unwrap();
            let mut records = rdr.records().map(|r| r.unwrap()).collect::<Vec<_>>();
            records.sort_by(|a, b| {
                let a = a[0].parse::<f64>().unwrap();
                let b = b[0].parse::<f64>().unwrap();
                a.partial_cmp(&b).unwrap()
            });

            let output_file = std::fs::File::create(&output_path).unwrap();
            let mut builder = csv::WriterBuilder::new();
            let builder = clio_utils::msft::msft_csv_writer_builder(&mut builder);
            let mut writer = builder.from_writer(output_file);
            for record in records {
                writer.write_record(&record).unwrap();
            }
            writer.flush().unwrap();
            std::fs::remove_file(&path).unwrap();

            // tar.gz the csv file
            let tar_path = std::fs::File::create(&tar_output_path).unwrap();
            let enc = flate2::write::GzEncoder::new(tar_path, flate2::Compression::default());
            let mut tar = tar::Builder::new(enc);
            tar.append_path_with_name(&output_path, output_path.file_name().unwrap())
                .unwrap();
            tar.finish().unwrap();
            std::fs::remove_file(&output_path).unwrap();
        }
    });

    let duration = start.elapsed();
    println!("Time elapsed in main: {:?}", duration);
    Ok(())
}
