use flate2::read::GzDecoder;
use std::collections::HashMap;
use std::path::PathBuf;
use std::time::Instant;
use std::{error::Error, process};
use trace_utils::path::is_tar_gz;

fn map_devices(p: PathBuf, volumes: &mut HashMap<i32, i32>) -> Result<(), Box<dyn Error>> {
    let mut rdr = csv::ReaderBuilder::new().has_headers(false).from_path(p)?;
    for (i, result) in rdr.records().enumerate() {
        let record = result?;
        volumes
            .entry(record[4].parse::<i32>()?)
            .and_modify(|v| *v += 1)
            .or_insert(1);

        if i % 10_000_000 == 0 && i > 0 {
            println!("Reaching record: {}", i);
        }
    }
    Ok(())
}

fn map_devices_gz(
    mut decoder: GzDecoder<std::fs::File>,
    volumes: &mut HashMap<i32, i32>,
) -> Result<(), Box<dyn Error>> {
    let mut rdr = csv::ReaderBuilder::new()
        .has_headers(false)
        .from_reader(&mut decoder);
    for (i, result) in rdr.records().enumerate() {
        let record = result?;
        volumes
            .entry(record[4].parse::<i32>()?)
            .and_modify(|v| *v += 1)
            .or_insert(1);

        if i % 10_000_000 == 0 && i > 0 {
            println!("Reaching record: {}", i);
        }
    }
    Ok(())
}

fn main() -> Result<(), Box<dyn Error>> {
    let start = Instant::now();

    let args: Vec<String> = std::env::args().collect();
    let input = PathBuf::from(&args[1]);
    let output = PathBuf::from(&args[2]);

    // mkdir output parent directory
    std::fs::create_dir_all(&output.parent().unwrap())?;

    if !input.exists() {
        println!("Input directory does not exist");
        process::exit(1);
    }

    let mut volumes = HashMap::new();

    if input.is_file() {
        let is_targz = is_tar_gz(&input)?;
        if !is_targz {
            if let Err(err) = map_devices(input, &mut volumes) {
                println!("error running example: {}", err);
                process::exit(1);
            }
        } else {
            // if targz is gzip file, then use map_devices_stream
            let targz = std::fs::File::open(&input)?;
            let decoder = GzDecoder::new(targz);
            if let Err(err) = map_devices_gz(decoder, &mut volumes) {
                println!("error running example: {}", err);
                process::exit(1);
            }
        }
    }

    // save to file
    let mut wtr = csv::WriterBuilder::new().from_path(output)?;
    wtr.serialize(["volume", "count"])?;
    for (k, v) in volumes.iter() {
        wtr.serialize((k, v))?;
    }
    wtr.flush()?;

    let duration = start.elapsed();
    println!("Time elapsed in main: {:?}", duration);
    Ok(())
}
