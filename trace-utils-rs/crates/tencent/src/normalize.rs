use clio_utils::msft::MsftTrace;
use std::error::Error;

pub fn normalize_to_msft(v: &Vec<String>) -> Result<MsftTrace, Box<dyn Error>> {
    if v.len() != 5 {
        return Err("Invalid number of fields".into());
    }

    let timestamp = v[0].parse::<f64>()?;
    let timestamp = timestamp * 1e3; // convert to milliseconds
    let offset = u128::from_str_radix(&v[1], 10)?;
    let offset = offset * 512; // 1 sector = 512B
    let size = u128::from_str_radix(&v[2], 10)?;
    let size = size * 512; // 1 sector = 512B
    let disk_id = u128::from_str_radix(&v[4], 10)?;

    let read = match v[3].as_str() {
        "0" => true,
        "1" => false,
        _ => return Err("Invalid read/write field".into()),
    };
    let res = MsftTrace::new(timestamp, disk_id, offset, size, read);
    Ok(res)
}

pub fn normalize_to_msft_from_csv_record(
    record: &csv::StringRecord,
) -> Result<MsftTrace, Box<dyn Error>> {
    let v = record
        .iter()
        .map(|field| field.to_string())
        .collect::<Vec<String>>();
    normalize_to_msft(&v)
}
