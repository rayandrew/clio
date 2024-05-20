use glob::glob;
use std::collections::HashMap;
use std::path::PathBuf;
use std::time::Instant;
use std::{error::Error, process};

fn map_devices(p: PathBuf, volumes: &mut HashMap<i32, i32>) -> Result<(), Box<dyn Error>> {
    let mut rdr = csv::ReaderBuilder::new().has_headers(true).from_path(p)?;
    for result in rdr.records() {
        let record = result?;
        let volume = record[0].parse::<i32>()?;
        let count = record[1].parse::<i32>()?;
        volumes
            .entry(volume)
            .and_modify(|v| *v += count)
            .or_insert(count);
    }
    Ok(())
}

fn main() -> Result<(), Box<dyn Error>> {
    let start = Instant::now();
    let args: Vec<String> = std::env::args().collect();
    let data_dir = PathBuf::from(&args[1]);
    let csv_files = format!("{}/**/*.csv", data_dir.to_str().unwrap());
    let files = glob(&csv_files)?;
    let mut files = files.map(|f| f.unwrap()).collect::<Vec<_>>();
    files.sort_by(|a, b| natord::compare(&a.to_string_lossy(), &b.to_string_lossy()));
    let mut volumes: HashMap<i32, i32> = HashMap::new();
    for entry in files {
        let entry = entry.as_path();
        let path = entry.to_path_buf();
        println!("Processing file: {:?}", path);
        if path.is_file() {
            if let Err(err) = map_devices(path, &mut volumes) {
                println!("error running example: {}", err);
                process::exit(1);
            }
        }
    }

    let mut volumes: Vec<_> = volumes.clone().into_iter().collect();
    volumes.sort_by(|a, b| a.1.cmp(&b.1));
    volumes.reverse();

    let mut wtr = csv::Writer::from_path("volumes.csv")?;
    wtr.serialize(["volume", "count"])?;
    for (k, v) in volumes.iter() {
        wtr.serialize((k, v))?;
    }
    wtr.flush()?;

    let duration = start.elapsed();
    println!("Time elapsed in main: {:?}", duration);

    Ok(())
}
