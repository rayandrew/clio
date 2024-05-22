use clap::Parser;
use clio_tencent::{TencentTraceBuilder, TencentTraceTrait};
use dashmap::DashMap;
use globwalk::glob;
use rayon::prelude::*;
use std::path::PathBuf;
use std::time::Instant;
use std::{error::Error, process};

// fn map_devices(p: PathBuf, volumes: &mut HashMap<i32, i32>) -> Result<(), Box<dyn Error>> {
//     let mut rdr = csv::ReaderBuilder::new().has_headers(true).from_path(p)?;
//     for result in rdr.records() {
//         let record = result?;
//         let volume = record[0].parse::<i32>()?;
//         let count = record[1].parse::<i32>()?;
//         volumes
//             .entry(volume)
//             .and_modify(|v| *v += count)
//             .or_insert(count);
//     }
//     Ok(())
// }

#[derive(Parser, Debug)]
#[command(version, about, long_about = None)]
struct Args {
    // Path to input file
    #[clap(short = 'i', long = "input")]
    input: String,

    // Path to output file
    #[clap(short = 'o', long = "output")]
    output: String,
}

fn main() -> Result<(), Box<dyn Error>> {
    let start = Instant::now();
    let args = Args::parse();
    let data_dir = PathBuf::from(args.input);
    let output = PathBuf::from(args.output);

    output.parent().map(|p| std::fs::create_dir_all(p));

    let csv_files = format!("{}/**/*.csv", data_dir.to_str().unwrap());
    let files = glob(&csv_files)?;
    let mut files = files.map(|f| f.unwrap()).collect::<Vec<_>>();
    files.sort_by(|a, b| natord::compare(&a.path().to_string_lossy(), &b.path().to_string_lossy()));
    let volumes: DashMap<i32, i32> = DashMap::new();

    files.into_par_iter().for_each(|entry| {
        let path = entry.path().to_path_buf();
        if path.is_file() {
            let mut tencent_trace = TencentTraceBuilder::new(&path).unwrap();
            tencent_trace.with_csv_builder(|csv_builder| csv_builder.has_headers(true));
            if let Err(err) = tencent_trace.read(|record: &csv::StringRecord| {
                let volume = record[0].parse::<i32>().unwrap();
                let count = record[1].parse::<i32>().unwrap();

                volumes
                    .entry(volume)
                    .and_modify(|v| *v += count)
                    .or_insert(1);
                Ok(())
            }) {
                println!("error running example: {}", err);
                process::exit(1);
            }
        }
    });

    // for entry in files {
    //     let entry = entry.path();
    //     let path = entry.to_path_buf();
    //     println!("Processing file: {:?}", path);
    //     if path.is_file() {
    //         if let Err(err) = map_devices(path, &mut volumes) {
    //             println!("error running example: {}", err);
    //             process::exit(1);
    //         }
    //     }
    // }

    let mut volumes: Vec<_> = volumes.clone().into_iter().collect();
    volumes.sort_by(|a, b| a.1.cmp(&b.1));
    volumes.reverse();

    let mut wtr = csv::Writer::from_path(output)?;
    wtr.serialize(["volume", "count"])?;
    for (k, v) in volumes.iter() {
        wtr.serialize((k, v))?;
    }
    wtr.flush()?;

    let duration = start.elapsed();
    println!("Time elapsed in main: {:?}", duration);

    Ok(())
}
