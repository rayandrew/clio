use std::error::Error;
use std::io::Read;
use std::path::PathBuf;

pub fn is_tar_gz(p: &PathBuf) -> Result<bool, Box<dyn Error>> {
    let mut file = std::fs::File::open(p)?;
    let mut first_line = [0; 2];
    file.read_exact(&mut first_line)?;
    Ok(first_line == [0x1f, 0x8b])
}
