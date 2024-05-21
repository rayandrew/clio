use clap::Parser;
use globwalk::glob;
use std::borrow::{Borrow, BorrowMut, Cow};
use std::cell::RefCell;
use std::rc::Rc;
use tencent::{TencentTraceTarGz, TencentTraceTrait};
use trace_utils::is_tar_gz;

#[derive(Parser, Debug)]
#[command(version, about, long_about = None)]
struct Args {
    // Path to data directory
    #[clap(short = 'd', long = "data-dir")]
    data_dir: String,

    // Path to output directory
    #[clap(short = 'o', long = "output-dir")]
    output_dir: String,
}

fn main() -> Result<(), Box<dyn std::error::Error>> {
    let start = std::time::Instant::now();

    let args = Args::parse();
    let matcher = format!("{}/**/*.{{tar.gz,.gz,.csv}}", args.data_dir);
    let files = glob(&matcher)?;
    let mut files = files.map(|f| f.unwrap()).collect::<Vec<_>>();
    files.sort_by(|a, b| natord::compare(&a.path().to_string_lossy(), &b.path().to_string_lossy()));

    // let mut trace_start_time = RefCell::new(u32::MAX);
    // let mut trace_start_time = Cow::Borrowed(&i32::MAX);
    let mut trace_start_time = std::i32::MAX;
    for entry in files {
        let entry = entry.path();
        let path = entry.to_path_buf();
        println!("Processing file: {:?}", path);

        if path.is_file() {
            let mut trace = TencentTraceTarGz::new();
            // trace.with_process(|record| {
            //     // let _trace_start_time = trace_start_time.borrow();
            //     let time = record[0].parse::<i32>().unwrap();
            //     if time < trace_start_time {
            //         trace_start_time = time;
            //         println!("New trace start time: {}", trace_start_time);
            //     }
            // });

            let output = path.file_name();
            let output = args.output_dir.clone() + "/" + output.unwrap().to_str().unwrap();
            println!("Output file: {:?}", output);
            // let output_file = std::fs::File::create(&output)?;

            if let Err(err) = trace.read(path, |record| {
                // let _trace_start_time = trace_start_time.borrow();
                let time = record[0].parse::<i32>().unwrap();
                if time < trace_start_time {
                    trace_start_time = time;
                    println!("New trace start time: {}", trace_start_time);
                }
            }) {
                println!("error running trace.read: {}", err);
                std::process::exit(1);
            }

            println!("Trace start time: {:?}", trace_start_time);

            // writer.flush()?;
        }
    }

    let duration = start.elapsed();
    println!("Time elapsed in main: {:?}", duration);
    Ok(())
}
