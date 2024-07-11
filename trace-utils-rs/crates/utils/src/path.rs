use std::env;
use std::error::Error;
use std::io::Read;
use std::path::Path;

pub fn is_gzip<P: AsRef<Path>>(p: P) -> Result<bool, Box<dyn Error>> {
    let mut file = std::fs::File::open(p)?;
    let mut first_line = [0; 2];
    file.read_exact(&mut first_line)?;
    Ok(first_line == [0x1f, 0x8b])
}

pub fn is_tar_gz<P: AsRef<Path>>(p: P) -> Result<bool, Box<dyn Error>> {
    let mut file = std::fs::File::open(p)?;
    let mut first_line = [0; 4];
    file.read_exact(&mut first_line)?;
    Ok(first_line == [0x1f, 0x8b, 0x08, 0x00])
}

pub fn is_tar<P: AsRef<Path>>(p: P) -> Result<bool, Box<dyn Error>> {
    let mut file = std::fs::File::open(p)?;
    let mut first_line = [0; 4];
    file.read_exact(&mut first_line)?;
    Ok(first_line == [0x75, 0x73, 0x74, 0x61])
}

pub fn is_csv_from_path<P: AsRef<Path>>(p: P) -> bool {
    p.as_ref().extension().map_or(false, |e| e == "csv")
}

pub fn is_tgz_from_path<P: AsRef<Path>>(p: P) -> bool {
    let pp = p.as_ref();
    // check if tgz in path
    if let Some(ext) = pp.extension() {
        if ext == "tgz" {
            return true;
        }
    }

    // check if tar.gz in path
    if let Some(file_name) = pp.file_name() {
        if let Some(file_name) = file_name.to_str() {
            if file_name.ends_with(".tar.gz") {
                return true;
            }
        }
    }

    false
}

pub fn is_gzip_from_path<P: AsRef<Path>>(p: P) -> bool {
    p.as_ref().extension().map_or(false, |e| e == "gz")
}

pub fn remove_extension<P: AsRef<Path> + std::convert::From<std::path::PathBuf>>(p: P) -> P {
    let mut p = p.as_ref().to_path_buf();
    while let Some(ext) = p.extension() {
        if ext == "" {
            break;
        }
        p = p.with_extension("");
    }
    p.into()
}

// CC BY-SA 3.0
// Adapted from https://stackoverflow.com/a/35046243
pub fn is_program_in_path<P: AsRef<Path>>(program: P) -> bool {
    let program = program.as_ref().to_str().unwrap();
    if let Ok(path) = env::var("PATH") {
        for p in path.split(":") {
            let p_str = format!("{}/{}", p, program);
            if std::fs::metadata(p_str).is_ok() {
                return true;
            }
        }
    }
    false
}
