use std::error::Error;
use std::io::Read;
use std::path::PathBuf;

pub fn is_gzip(p: &PathBuf) -> Result<bool, Box<dyn Error>> {
    let mut file = std::fs::File::open(p)?;
    let mut first_line = [0; 2];
    file.read_exact(&mut first_line)?;
    Ok(first_line == [0x1f, 0x8b])
}

pub fn is_tar_gz(p: &PathBuf) -> Result<bool, Box<dyn Error>> {
    let mut file = std::fs::File::open(p)?;
    let mut first_line = [0; 4];
    file.read_exact(&mut first_line)?;
    Ok(first_line == [0x1f, 0x8b, 0x08, 0x00])
}

pub fn is_tar(p: &PathBuf) -> Result<bool, Box<dyn Error>> {
    let mut file = std::fs::File::open(p)?;
    let mut first_line = [0; 4];
    file.read_exact(&mut first_line)?;
    Ok(first_line == [0x75, 0x73, 0x74, 0x61])
}

pub fn is_csv_from_path(p: &PathBuf) -> bool {
    p.extension().map_or(false, |e| e == "csv")
}

pub fn is_tgz_from_path(p: &PathBuf) -> bool {
    // check if tgz in path
    if let Some(ext) = p.extension() {
        if ext == "tgz" {
            return true;
        }
    }

    // check if tar.gz in path
    if let Some(file_name) = p.file_name() {
        if let Some(file_name) = file_name.to_str() {
            if file_name.ends_with(".tar.gz") {
                return true;
            }
        }
    }

    false
}

pub fn is_gzip_from_path(p: &PathBuf) -> bool {
    p.extension().map_or(false, |e| e == "gz")
}
