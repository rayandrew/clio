use crate::dpi::Dpi;
use std::path::Path;
use std::process::Command;

pub fn convert_eps_to_png<P: AsRef<Path>>(
    p: P,
    output: P,
    dpi: &Dpi,
) -> Result<(), Box<dyn std::error::Error>> {
    let p = p.as_ref();
    let o = output.as_ref();
    let proc = Command::new("gs")
        .args(&[
            "-dSAFER",
            "-dBATCH",
            "-dNOPAUSE",
            "-dEPSCrop",
            "-sDEVICE=png16m",
            &format!("-r{}", dpi),
            &format!("-sOutputFile={}", &o.to_str().unwrap()),
            &format!("{}", &p.to_str().unwrap()),
        ])
        .output()?;

    if !proc.status.success() {
        return Err(format!(
            "Failed to convert EPS to PNG: {}",
            String::from_utf8_lossy(&proc.stderr)
        )
        .into());
    }

    Ok(())
}
