use clap::Parser;
use clio_tencent::{TencentTraceBuilder, TencentTraceTrait};
use clio_utils::path::remove_extension;
use globwalk::glob;
use rayon::iter::{IntoParallelIterator, ParallelIterator};
use std::path::PathBuf;
use std::time::Instant;
use std::{error::Error, process};

#[derive(Parser, Debug)]
#[command(version, about, long_about = None)]
struct Args {
    // Path to input directory
    #[clap(short = 'i', long = "input")]
    input: String,

    // Path to output file
    #[clap(short = 'o', long = "output")]
    output: String,

    // Volume to filter
    #[clap(short = 'v', long = "volume")]
    volume: u128,
}

// fn filter_device<T: std::io::Write>(
//     p: PathBuf,
//     mut writer: csv::Writer<T>,
//     volume: i32,
// ) -> Result<(), Box<dyn Error>> {
//     let mut rdr = csv::ReaderBuilder::new().has_headers(false).from_path(p)?;
//     for (i, result) in rdr.records().enumerate() {
//         let record = result?;
//         let vol = record[4].parse::<i32>()?;

//         if vol != volume {
//             continue;
//         }
//         // writer.write_record(&record)?;
//         writer.write_record(&record).expect("write record failed");
//         if i % 10_000_000 == 0 && i > 0 {
//             println!("Reaching record: {}", i);
//         }
//     }
//     writer.flush()?;
//     Ok(())
// }

// fn filter_device_gz<T: std::io::Write>(
//     mut decoder: flate2::read::GzDecoder<std::fs::File>,
//     mut writer: csv::Writer<T>,
//     volume: i32,
// ) -> Result<(), Box<dyn Error>> {
//     let mut rdr = csv::ReaderBuilder::new()
//         .has_headers(false)
//         .flexible(false)
//         .from_reader(&mut decoder);
//     for (i, result) in rdr.records().enumerate() {
//         let record = result?;
//         // if record.len() < 5 { // skip invalid records
//         //     continue;
//         // }
//         let vol = record[4].parse::<i32>()?;

//         if vol != volume {
//             continue;
//         }
//         writer.write_record(&record)?;
//         if i % 10_000_000 == 0 && i > 0 {
//             println!("Reaching record: {}", i);
//         }
//     }
//     writer.flush()?;
//     Ok(())
// }

// fn filter_device_tar_gz<T: std::io::Write>(
//     decoder: flate2::read::GzDecoder<std::fs::File>,
//     mut writer: csv::Writer<T>,
//     volume: i32,
// ) -> Result<(), Box<dyn Error>> {
//     let mut tar = tar::Archive::new(decoder);
//     let mut files = tar.entries()?;
//     let entry = files.next();
//     if entry.is_none() {
//         return Ok(());
//     }

//     let entry = entry.unwrap()?;
//     let mut rdr = csv::ReaderBuilder::new()
//         .has_headers(false)
//         .flexible(false)
//         .from_reader(entry);
//     for (i, result) in rdr.records().enumerate() {
//         let record = result?;
//         // if record.len() < 5 { // enable this if flexible = true
//         //     continue;
//         // }
//         let vol = record[4].parse::<i32>()?;

//         if vol != volume {
//             continue;
//         }
//         writer.write_record(&record)?;
//         if i % 10_000_000 == 0 && i > 0 {
//             println!("Reaching record: {}", i);
//             // println!("Record: {:?}", record);
//         }
//     }

//     writer.flush()?;

//     Ok(())
// }

////////////////////////
// OLD IMPLEMENTATION
////////////////////////
// fn main() -> Result<(), Box<dyn Error>> {
//     let start = Instant::now();

//     let args = Args::parse();
//     let input = PathBuf::from(&args.input);
//     let output = PathBuf::from(&args.output);
//     let volume = args.volume;

//     // mkdir output parent directory
//     std::fs::create_dir_all(&output.parent().unwrap())?;

//     if !input.exists() {
//         println!("Input file does not exist");
//         process::exit(1);
//     }

//     if input.is_file() {
//         let temp_file_path = if is_csv_from_path(&output) {
//             format!(
//                 "{}_temp",
//                 output
//                     .with_extension("")
//                     .with_extension("")
//                     .to_str()
//                     .unwrap()
//             )
//         } else if is_gzip_from_path(&output) {
//             format!(
//                 "{}_temp",
//                 output
//                     .with_extension("")
//                     .with_extension("")
//                     .to_str()
//                     .unwrap()
//             )
//             .into()
//         } else {
//             println!("Output file must be csv or tar.gz or tgz or gz");
//             process::exit(1);
//         };

//         let mut tencent_trace = TencentTraceBuilder::new(&input)?;
//         tencent_trace.with_filter(|record| {
//             let vol = record[4].parse::<i32>().unwrap();
//             vol == volume
//         });

//         if let Err(err) = tencent_trace.read(|record| {
//             let temp_output_file = std::fs::File::create(&temp_file_path)?;
//             let mut writer = csv::WriterBuilder::new().from_writer(temp_output_file);
//             writer.write_record(record)?;
//             Ok(())
//         }) {
//             println!("error running example: {}", err);
//             process::exit(1);
//         }

//         // SORTING based on timestamp
//         println!("Sorting based on timestamp");
//         let temp_file_path_sorted = if is_csv_from_path(&output) {
//             output.clone()
//         } else if is_gzip_from_path(&output) {
//             output.with_extension("").with_extension("")
//         } else {
//             println!("Output file must be csv or tar.gz or tgz or gz");
//             process::exit(1);
//         };
//         let temp_file = std::fs::File::open(&temp_file_path)?;
//         let mut rdr = csv::ReaderBuilder::new()
//             .has_headers(false)
//             .flexible(false)
//             .from_reader(temp_file);

//         let mut records = rdr.records().map(|r| r.unwrap()).collect::<Vec<_>>();
//         records.sort_by(|a, b| {
//             let a = a[0].parse::<u128>().unwrap();
//             let b = b[0].parse::<u128>().unwrap();
//             a.cmp(&b)
//         });
//         let output_file = std::fs::File::create(&temp_file_path_sorted)?;
//         let mut writer = csv::WriterBuilder::new().from_writer(output_file);
//         for record in records {
//             writer.write_record(&record)?;
//         }
//         writer.flush()?;
//         std::fs::remove_file(&temp_file_path)?;

//         // POST PROCESSING
//         if is_tgz_from_path(&output) {
//             let output_file = std::fs::File::create(&output)?;
//             let encoder =
//                 flate2::write::GzEncoder::new(output_file, flate2::Compression::default());
//             let mut tar = tar::Builder::new(encoder);
//             tar.append_file(
//                 temp_file_path_sorted.file_name().unwrap().to_str().unwrap(),
//                 &mut std::fs::File::open(temp_file_path_sorted.clone())?,
//             )?;
//             std::fs::remove_file(temp_file_path_sorted)?;
//             tar.finish()?;
//         } else if is_gzip_from_path(&output) {
//             let output_file = std::fs::File::create(&output)?;
//             let encoder =
//                 flate2::write::GzEncoder::new(output_file, flate2::Compression::default());
//             std::fs::copy(temp_file_path_sorted.clone(), output)?;
//             std::fs::remove_file(temp_file_path_sorted)?;
//             encoder.finish()?;
//         }
//     }

//     let duration = start.elapsed();
//     println!("Time elapsed in main: {:?}", duration);
//     Ok(())
// }

fn main() -> Result<(), Box<dyn Error>> {
    let start = Instant::now();

    let args = Args::parse();
    let input = PathBuf::from(&args.input);
    let output = PathBuf::from(&args.output);
    let volume = args.volume;

    println!("Input: {:?}", input);

    if !input.is_dir() {
        println!("Input directory does not exist");
        process::exit(1);
    }

    let matcher = format!("{}/**/*.{{tar.gz,gz,csv,tgz}}", input.to_string_lossy());
    println!("Matcher: {:?}", matcher);
    let files = glob(&matcher)?;

    let mut files = files.map(|f| f.unwrap()).collect::<Vec<_>>();
    files.sort_by(|a, b| natord::compare(&a.path().to_string_lossy(), &b.path().to_string_lossy()));

    if files.len() == 0 {
        println!("No files found in the input directory");
        process::exit(1);
    }

    // create output directory
    std::fs::create_dir_all(&output)?;

    files.into_par_iter().for_each(|entry| {
        println!("Processing file: {:?}", entry.path());
        let path = entry.path().to_path_buf();
        let temp_file_path = format!(
            "{}_temp",
            output
                .join(remove_extension(path.clone()).file_name().unwrap())
                .to_string_lossy()
        );

        {
            let mut tencent_trace = TencentTraceBuilder::new(&path).unwrap();
            tencent_trace.with_filter(|record| {
                let vol = record[4].parse::<u128>().unwrap();

                vol == volume
            });
            let temp_output_file = std::fs::File::create(&temp_file_path).unwrap();
            let mut writer = csv::WriterBuilder::new().from_writer(temp_output_file);

            if let Err(err) = tencent_trace.read(|record| {
                writer.write_record(record)?;
                Ok(())
            }) {
                println!("error running example: {}", err);
                process::exit(1);
            }
            writer.flush().unwrap();
        }

        let temp_file_path_sorted =
            output.join(remove_extension(path.clone()).file_name().unwrap());
        {
            // SORTING based on timestamp
            // println!("Sorting based on timestamp");
            let temp_file = std::fs::File::open(&temp_file_path).unwrap();
            let mut rdr = csv::ReaderBuilder::new()
                .has_headers(false)
                .flexible(false)
                .from_reader(temp_file);

            let mut records = rdr.records().map(|r| r.unwrap()).collect::<Vec<_>>();
            records.sort_by(|a, b| {
                let a = a[0].parse::<u128>().unwrap();
                let b = b[0].parse::<u128>().unwrap();
                a.cmp(&b)
            });
            let output_file = std::fs::File::create(&temp_file_path_sorted).unwrap();
            let mut writer = csv::WriterBuilder::new().from_writer(output_file);
            for record in records {
                writer.write_record(&record).unwrap();
            }
            writer.flush().unwrap();
            std::fs::remove_file(&temp_file_path).unwrap();
        }

        // POST PROCESSING
        {
            let output_path = output.join(
                remove_extension(path.clone())
                    .with_extension("tar.gz")
                    .file_name()
                    .unwrap(),
            );
            let output_file = std::fs::File::create(output_path).unwrap();
            let encoder =
                flate2::write::GzEncoder::new(output_file, flate2::Compression::default());
            let mut tar = tar::Builder::new(encoder);
            tar.append_file(
                temp_file_path_sorted.file_name().unwrap().to_str().unwrap(),
                &mut std::fs::File::open(temp_file_path_sorted.clone()).unwrap(),
            )
            .unwrap();
            std::fs::remove_file(temp_file_path_sorted).unwrap();
            tar.finish().unwrap();
        }
    });

    let duration = start.elapsed();
    println!("Time elapsed in main: {:?}", duration);
    Ok(())
}
