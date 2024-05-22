use std::path::PathBuf;
use std::time::Instant;
use std::{error::Error, process};
use trace_utils::path::{
    is_csv_from_path, is_gzip, is_gzip_from_path, is_tar_gz, is_tgz_from_path,
};

fn filter_device<T: std::io::Write>(
    p: PathBuf,
    mut writer: csv::Writer<T>,
    volume: i32,
) -> Result<(), Box<dyn Error>> {
    let mut rdr = csv::ReaderBuilder::new().has_headers(false).from_path(p)?;
    for (i, result) in rdr.records().enumerate() {
        let record = result?;
        let vol = record[4].parse::<i32>()?;

        if vol != volume {
            continue;
        }
        // writer.write_record(&record)?;
        writer.write_record(&record).expect("write record failed");
        if i % 10_000_000 == 0 && i > 0 {
            println!("Reaching record: {}", i);
        }
    }
    writer.flush()?;
    Ok(())
}

fn filter_device_gz<T: std::io::Write>(
    mut decoder: flate2::read::GzDecoder<std::fs::File>,
    mut writer: csv::Writer<T>,
    volume: i32,
) -> Result<(), Box<dyn Error>> {
    let mut rdr = csv::ReaderBuilder::new()
        .has_headers(false)
        .flexible(false)
        .from_reader(&mut decoder);
    for (i, result) in rdr.records().enumerate() {
        let record = result?;
        // if record.len() < 5 { // skip invalid records
        //     continue;
        // }
        let vol = record[4].parse::<i32>()?;

        if vol != volume {
            continue;
        }
        writer.write_record(&record)?;
        if i % 10_000_000 == 0 && i > 0 {
            println!("Reaching record: {}", i);
        }
    }
    writer.flush()?;
    Ok(())
}

fn filter_device_tar_gz<T: std::io::Write>(
    decoder: flate2::read::GzDecoder<std::fs::File>,
    mut writer: csv::Writer<T>,
    volume: i32,
) -> Result<(), Box<dyn Error>> {
    let mut tar = tar::Archive::new(decoder);
    let mut files = tar.entries()?;
    let entry = files.next();
    if entry.is_none() {
        return Ok(());
    }

    let entry = entry.unwrap()?;
    let mut rdr = csv::ReaderBuilder::new()
        .has_headers(false)
        .flexible(false)
        .from_reader(entry);
    for (i, result) in rdr.records().enumerate() {
        let record = result?;
        // if record.len() < 5 { // enable this if flexible = true
        //     continue;
        // }
        let vol = record[4].parse::<i32>()?;

        if vol != volume {
            continue;
        }
        writer.write_record(&record)?;
        if i % 10_000_000 == 0 && i > 0 {
            println!("Reaching record: {}", i);
            // println!("Record: {:?}", record);
        }
    }

    writer.flush()?;

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
        let temp_file_path = if is_csv_from_path(&output) {
            format!(
                "{}_temp",
                output
                    .with_extension("")
                    .with_extension("")
                    .to_str()
                    .unwrap()
            )
        } else if is_gzip_from_path(&output) {
            format!(
                "{}_temp",
                output
                    .with_extension("")
                    .with_extension("")
                    .to_str()
                    .unwrap()
            )
            .into()
        } else {
            println!("Output file must be csv or tar.gz or tgz or gz");
            process::exit(1);
        };

        let temp_output_file = std::fs::File::create(&temp_file_path)?;
        let writer = csv::WriterBuilder::new().from_writer(temp_output_file);

        if is_tar_gz(&input)? {
            let file = std::fs::File::open(&input)?;
            let decoder = flate2::read::GzDecoder::new(file);
            if let Err(err) = filter_device_tar_gz(decoder, writer, volume) {
                println!("error running filter_device_tar_gz: {}", err);
                process::exit(1);
            }
        } else if is_gzip(&input)? {
            let file = std::fs::File::open(&input)?;
            let decoder = flate2::read::GzDecoder::new(file);
            if let Err(err) = filter_device_gz(decoder, writer, volume) {
                println!("error running filter_device_gz: {}", err);
                process::exit(1);
            }
        } else {
            if let Err(err) = filter_device(input, writer, volume) {
                println!("error running filter_device: {}", err);
                process::exit(1);
            }
        }

        // SORTING based on timestamp
        println!("Sorting based on timestamp");
        let temp_file_path_sorted = if is_csv_from_path(&output) {
            output.clone()
        } else if is_gzip_from_path(&output) {
            output.with_extension("").with_extension("")
        } else {
            println!("Output file must be csv or tar.gz or tgz or gz");
            process::exit(1);
        };
        let temp_file = std::fs::File::open(&temp_file_path)?;
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
        let output_file = std::fs::File::create(&temp_file_path_sorted)?;
        let mut writer = csv::WriterBuilder::new().from_writer(output_file);
        for record in records {
            writer.write_record(&record)?;
        }
        writer.flush()?;
        std::fs::remove_file(&temp_file_path)?;

        // POST PROCESSING
        if is_tgz_from_path(&output) {
            let output_file = std::fs::File::create(&output)?;
            let encoder =
                flate2::write::GzEncoder::new(output_file, flate2::Compression::default());
            let mut tar = tar::Builder::new(encoder);
            tar.append_file(
                temp_file_path_sorted.file_name().unwrap().to_str().unwrap(),
                &mut std::fs::File::open(temp_file_path_sorted.clone())?,
            )?;
            std::fs::remove_file(temp_file_path_sorted)?;
            tar.finish()?;
        } else if is_gzip_from_path(&output) {
            let output_file = std::fs::File::create(&output)?;
            let encoder =
                flate2::write::GzEncoder::new(output_file, flate2::Compression::default());
            std::fs::copy(temp_file_path_sorted.clone(), output)?;
            std::fs::remove_file(temp_file_path_sorted)?;
            encoder.finish()?;
        }
    }

    let duration = start.elapsed();
    println!("Time elapsed in main: {:?}", duration);
    Ok(())
}
