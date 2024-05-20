use flate2::read::GzDecoder;
use std::path::PathBuf;
use std::time::Instant;
use std::{error::Error, process};
use trace_utils::is_tar_gz;

fn filter_device(p: PathBuf, output_path: PathBuf, volume: i32) -> Result<(), Box<dyn Error>> {
    let mut rdr = csv::ReaderBuilder::new().has_headers(false).from_path(p)?;
    let mut wrt = csv::WriterBuilder::new().from_path(output_path)?;
    for (i, result) in rdr.records().enumerate() {
        let record = result?;
        let vol = record[4].parse::<i32>()?;

        if vol != volume {
            continue;
        }
        wrt.write_record(&record)?;
        if i % 10_000_000 == 0 && i > 0 {
            println!("Reaching record: {}", i);
        }
    }
    wrt.flush()?;
    Ok(())
}

fn filter_device_gz(
    mut decoder: GzDecoder<std::fs::File>,
    output_path: PathBuf,
    volume: i32,
) -> Result<(), Box<dyn Error>> {
    let mut rdr = csv::ReaderBuilder::new()
        .has_headers(false)
        .from_reader(&mut decoder);
    let mut wrt = csv::WriterBuilder::new().from_path(output_path)?;
    for (i, result) in rdr.records().enumerate() {
        let record = result?;
        let vol = record[4].parse::<i32>()?;

        if vol != volume {
            continue;
        }
        wrt.write_record(&record)?;
        if i % 10_000_000 == 0 && i > 0 {
            println!("Reaching record: {}", i);
        }
    }
    wrt.flush()?;
    Ok(())
}

fn main() -> Result<(), Box<dyn Error>> {
    let start = Instant::now();

    let args: Vec<String> = std::env::args().collect();
    let input = PathBuf::from(&args[1]);
    let output = PathBuf::from(&args[2]);
    let volume = args[3].parse::<i32>()?;

    // mkdir output parent directory
    std::fs::create_dir_all(&output.parent().unwrap())?;

    if !input.exists() {
        println!("Input file does not exist");
        process::exit(1);
    }

    if input.is_file() {
        let is_targz = is_tar_gz(&input)?;
        
        if !is_targz {
            if let Err(err) = filter_device(input, output, volume) {
                println!("error running filter_device: {}", err);
                process::exit(1);
            }
        } else {
            let file = std::fs::File::open(&input)?;
            let decoder = GzDecoder::new(file);
            if let Err(err) = filter_device_gz(decoder, output, volume) {
                println!("error running filter_device_gz: {}", err);
                process::exit(1);
            }
        }
    }

    let duration = start.elapsed();
    println!("Time elapsed in main: {:?}", duration);
    Ok(())
}
