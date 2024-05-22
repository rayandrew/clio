// use clio_characteristic::RawTraceCharacteristic;
use std::error::Error;
use std::process;

use clio_characteristic::characteristic::RawTraceCharacteristic;
use clio_utils::msft::{msft_csv_reader_builder, MsftTrace};
use clio_utils::trace_reader::{TraceReaderBuilder, TraceReaderTrait};

use clap::Parser;
use globwalk::glob;
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
    // #[clap(short = 'w', long = "window", default_value = "1m")]
    // window: String,
}

fn main() -> Result<(), Box<dyn Error>> {
    let start = std::time::Instant::now();

    let args = Args::parse();
    let matcher = format!("{}/**/*.{{tar.gz,gz,csv,tgz}}", args.input);
    let output_dir = std::path::Path::new(&args.output);
    // let window = duration_str::parse(&args.window).unwrap();

    let files = glob(&matcher)?;
    let mut files = files.map(|f| f.unwrap()).collect::<Vec<_>>();
    files.sort_by(|a, b| natord::compare(&a.path().to_string_lossy(), &b.path().to_string_lossy()));

    let mut characteristics: Vec<RawTraceCharacteristic> = files
        .into_par_iter()
        .map(|entry| {
            let path = entry.path().to_path_buf();
            let mut tencent_trace = TraceReaderBuilder::new(&path).unwrap();
            tencent_trace.with_csv_builder(|csv_builder| msft_csv_reader_builder(csv_builder));
            let mut traces = Vec::new();
            if let Err(err) = tencent_trace.read(|record: &csv::StringRecord| {
                let record: MsftTrace = record.try_into()?;
                traces.push(record);
                Ok(())
            }) {
                println!("error: {}", err);
                process::exit(1);
            }

            traces.try_into().unwrap()
        })
        .collect();

    characteristics.sort_by(|a, b| a.start_time.partial_cmp(&b.start_time).unwrap());

    let output_file = output_dir.join("characteristic.csv");
    let output_file = std::fs::File::create(output_file)?;
    let mut writer = csv::Writer::from_writer(output_file);
    writer.write_byte_record(&RawTraceCharacteristic::csv_header())?;
    for characteristic in &characteristics {
        let record: csv::ByteRecord = characteristic.try_into()?;
        writer.write_byte_record(&record)?;
    }

    let output_file = output_dir.join("characteristic.json");
    let output_file = std::fs::File::create(output_file)?;
    serde_json::to_writer(output_file, &characteristics)?;

    let duration = start.elapsed();
    println!("Time elapsed in main() is: {:?}", duration);

    Ok(())
}
