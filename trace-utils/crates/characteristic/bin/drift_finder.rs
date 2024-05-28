use clap::Parser;
use clio_utils::file::BufReader;
use clio_utils::path::remove_extension;
use clio_utils::pbar::default_pbar_style;
use globwalk::glob;
use indicatif::{MultiProgress, ParallelProgressIterator, ProgressBar};
use kernel_density_estimation::prelude::*;
use polars::prelude::*;
use rayon::prelude::*;
use std::collections::HashMap;
use std::error::Error;
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
}

fn main() -> Result<(), Box<dyn Error>> {
    let start = std::time::Instant::now();

    let args = Args::parse();
    let input = Path::new(&args.input);
    let output_dir = Path::new(&args.output);

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

    println!("Data length: {}", data.len());

    let duration = start.elapsed();
    println!("Time elapsed in main() is: {:?}", duration);

    Ok(())
}
