use globwalk::glob;
use std::collections::HashMap;
use std::path::PathBuf;
use std::time::Instant;
use std::{error::Error, process};

use clap::Parser;
use clio_tencent::{TencentTraceBuilder, TencentTraceTrait};
// use dashmap::DashMap;
use rayon::prelude::*;

#[derive(Parser, Debug)]
#[command(version, about, long_about = None)]
struct Args {
    // Path to input file
    #[clap(short = 'i', long = "input")]
    input: String,

    // Path to output file
    #[clap(short = 'o', long = "output")]
    output: String,

    // Pattern to filter
    #[clap(short = 'p', long = "pattern")]
    pattern: Option<String>,
}

//////////////////////
// OLD CODE (SINGLE THREAD, parallelize using processes (gnu parallel))
//////////////////////
// fn single_main() -> Result<(), Box<dyn Error>> {
//     let start = Instant::now();

//     let args = Args::parse();
//     let input = PathBuf::from(&args.input);
//     let output = PathBuf::from(&args.output);

//     // mkdir output parent directory
//     std::fs::create_dir_all(&output.parent().unwrap())?;

//     if !input.exists() {
//         println!("Input directory does not exist");
//         process::exit(1);
//     }

//     let mut volumes = HashMap::new();

//     if input.is_file() {
//         let tencent_trace = TencentTraceBuilder::new(&input)?;
//         if let Err(err) = tencent_trace.read(|record: &csv::StringRecord| {
//             volumes
//                 .entry(record[4].parse::<i32>().unwrap())
//                 .and_modify(|v| *v += 1)
//                 .or_insert(1);
//             Ok(())
//         }) {
//             println!("error running example: {}", err);
//             process::exit(1);
//         }
//     }

//     // save to file
//     let mut wtr = csv::WriterBuilder::new().from_path(output)?;
//     wtr.serialize(["volume", "count"])?;
//     for (k, v) in volumes.iter() {
//         wtr.serialize((k, v))?;
//     }
//     wtr.flush()?;

//     let duration = start.elapsed();
//     println!("Time elapsed in main: {:?}", duration);
//     Ok(())
// }
//////////////////////

fn main() -> Result<(), Box<dyn Error>> {
    let start = Instant::now();

    let args = Args::parse();
    let input = PathBuf::from(&args.input);
    let output = PathBuf::from(&args.output);
    let pattern = args.pattern;

    if !input.is_dir() {
        println!("Input must be a directory");
        process::exit(1);
    }

    let files = if let Some(pattern) = pattern {
        glob(&format!("{}/{}", input.to_string_lossy(), pattern))?
    } else {
        glob(&format!("{}/**/*.csv", input.to_string_lossy()))?
    };
    let files = files.map(|f| f.unwrap()).collect::<Vec<_>>();

    let files_len = files.len();
    println!(
        "Reading {} files in directory {:?} in parallel",
        files_len, input
    );
    files.into_par_iter().for_each(|file| {
        let mut volumes = HashMap::new();
        let path = file.path();
        println!("Reading file: {:?}", path);
        let tencent_trace = TencentTraceBuilder::new(&path).unwrap();
        tencent_trace
            .read(|record| {
                volumes
                    .entry(record[4].parse::<i32>().unwrap())
                    .and_modify(|v| *v += 1)
                    .or_insert(1);
                Ok(())
            })
            .unwrap();
        let mut wtr = csv::WriterBuilder::new()
            .from_path(output.join(path.with_extension("csv").file_name().unwrap()))
            .unwrap();
        wtr.serialize(["volume", "count"]).unwrap();
        for (k, v) in volumes.iter() {
            wtr.serialize((k, v)).unwrap();
        }
        wtr.flush().unwrap();
    });

    std::fs::create_dir_all(&output.parent().unwrap())?;

    if !input.exists() {
        println!("Input directory does not exist");
        process::exit(1);
    }

    // let mut sorted_volumes: Vec<_> = volumes.iter().collect();
    // sorted_volumes.sort_by(|a, b| a.value().cmp(b.value()));

    // save to file

    let duration = start.elapsed();
    println!("Time elapsed in main: {:?}", duration);
    Ok(())
}
