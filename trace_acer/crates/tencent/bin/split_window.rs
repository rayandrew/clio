use clap::Parser;
use dashmap::DashMap;
use duration_str::parse;
use globwalk::glob;
use rand::Rng;
use rayon::prelude::*;
use tencent::{TencentTraceTarGz, TencentTraceTrait};

#[derive(Parser, Debug)]
#[command(version, about, long_about = None)]
struct Args {
    // Path to data directory
    #[clap(short = 'd', long = "data-dir")]
    data_dir: String,

    // Path to output directory
    #[clap(short = 'o', long = "output-dir")]
    output_dir: String,

    // Duration to split the trace
    #[clap(short = 'w', long = "window", default_value = "1m")]
    window: String,
}

fn main() -> Result<(), Box<dyn std::error::Error>> {
    let start = std::time::Instant::now();

    let args = Args::parse();
    let matcher = format!("{}/**/*.{{tar.gz,.gz,.csv}}", args.data_dir);
    let output_dir = std::path::Path::new(&args.output_dir);
    let canonical_output_dir =
        std::fs::canonicalize(&output_dir).unwrap_or(output_dir.to_path_buf());
    let window = parse(&args.window)?;
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
    let temp_files = files.clone();
    let trace_start_time = temp_files
        .into_par_iter()
        .map(|entry| {
            let path = entry.path().to_path_buf();
            if path.is_file() {
                let trace = TencentTraceTarGz::new();
                let mut min_ts = u128::MAX;
                if let Err(err) = trace.read(path.clone(), |record| {
                    let ori_time = record[0].parse::<u128>().unwrap();
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
                u128::MAX
            }
        })
        .reduce(
            || u128::MAX,
            |acc, time| if time < acc { time } else { acc },
        );

    println!("Start time of the trace: {:?}", trace_start_time);
    let hashmap_writer: DashMap<u128, csv::Writer<std::fs::File>> = DashMap::new();
    let files_len = files.len();

    // -------------------------------------------------------------------------------
    println!("Splitting trace into windows ...");
    files.into_par_iter().enumerate().for_each(|(i, entry)| {
        let mut prev_jitter = 0;
        let mut prev_time = 0;
        let mut rng = rand::thread_rng();
        let path = entry.path().to_path_buf();
        if path.is_file() {
            if i % 50 == 0 && i > 0 {
                println!("Processing in progress: {} files out of {}", i, files_len);
            }
            let trace = TencentTraceTarGz::new();
            if let Err(err) = trace.read(path, |record| {
                let ori_time = record[0].parse::<u128>().unwrap();
                let ori_time = ori_time - trace_start_time; // in seconds
                let time = ori_time * 1000; // in milliseconds
                let mut ms_time = time;
                if prev_time == time {
                    let jitter = rng.gen_range(prev_jitter + 1..prev_jitter + 5);
                    ms_time += jitter;
                    prev_jitter = jitter;
                } else {
                    prev_jitter = 0;
                }
                prev_time = time;
                let mut temp_record: Vec<String> = vec![];
                for (i, field) in record.iter().enumerate() {
                    if i == 0 {
                        temp_record.push(ms_time.to_string());
                    } else if i == 3 {
                        // io_type, revert 1 to 0 and 0 to 1
                        // check if the field contains 1 or 0
                        if field.contains("1") {
                            temp_record.push("0".to_string());
                        } else {
                            temp_record.push("1".to_string());
                        }
                    } else {
                        temp_record.push(field.to_string());
                    }
                }
                let current_chunk = ms_time / window.as_millis();

                if let Some(mut writer) = hashmap_writer.get_mut(&current_chunk) {
                    writer.write_record(&temp_record)?;
                } else {
                    let chunk_path =
                        canonical_output_dir.join(format!("chunk_{}.temp", current_chunk));
                    let output_file = std::fs::File::create(chunk_path).unwrap();
                    let mut writer = csv::WriterBuilder::new().from_writer(output_file);
                    writer.write_record(&temp_record)?;
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

    files.into_par_iter().enumerate().for_each(|(i, entry)| {
        let path = entry.path();
        let output_path = path.to_path_buf().with_extension("");
        let tar_output_path = path.to_path_buf().with_extension("tar.gz");

        if i % 500 == 0 && i > 0 {
            println!("Sorting in progress: {} files out of {}", i, files_len);
        }
        if path.is_file() {
            let mut rdr = csv::ReaderBuilder::new()
                .has_headers(false)
                .from_path(&path)
                .unwrap();
            let mut records = rdr.records().map(|r| r.unwrap()).collect::<Vec<_>>();
            records.sort_by(|a, b| {
                let a = a[0].parse::<u128>().unwrap();
                let b = b[0].parse::<u128>().unwrap();
                a.cmp(&b)
            });

            let output_file = std::fs::File::create(&output_path).unwrap();
            let mut writer = csv::WriterBuilder::new().from_writer(output_file);
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
