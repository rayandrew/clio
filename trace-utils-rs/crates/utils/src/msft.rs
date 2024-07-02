use crate::serde::OrderedFloatDef;
use ordered_float::OrderedFloat;
use serde::{Deserialize, Serialize};
use std::fmt::Debug;

#[derive(Debug, Eq, PartialEq, Serialize, Deserialize)]
pub struct MsftTrace {
    #[serde(with = "OrderedFloatDef")]
    pub timestamp: OrderedFloat<f64>,
    pub disk_id: u128,
    pub offset: u128,
    pub size: u128,
    pub read: bool,
}

impl Default for MsftTrace {
    fn default() -> Self {
        Self {
            timestamp: OrderedFloat::from(0.0),
            disk_id: 0,
            offset: 0,
            size: 0,
            read: false,
        }
    }
}

impl PartialOrd for MsftTrace {
    fn partial_cmp(&self, other: &Self) -> Option<std::cmp::Ordering> {
        self.timestamp.partial_cmp(&other.timestamp)
    }
}

impl Ord for MsftTrace {
    fn cmp(&self, other: &Self) -> std::cmp::Ordering {
        self.timestamp
            .partial_cmp(&other.timestamp)
            .unwrap_or(std::cmp::Ordering::Equal)
    }
}

impl MsftTrace {
    pub fn new(timestamp: f64, disk_id: u128, offset: u128, size: u128, read: bool) -> Self {
        Self {
            timestamp: OrderedFloat::from(timestamp),
            disk_id,
            offset,
            size,
            read,
        }
    }

    pub fn is_read(&self) -> bool {
        self.read
    }

    pub fn is_write(&self) -> bool {
        !self.read
    }

    pub fn to_vec(&self) -> Vec<String> {
        vec![
            format!("{:.4}", self.timestamp),
            self.disk_id.to_string(),
            self.offset.to_string(),
            self.size.to_string(),
            (self.read as u8).to_string(),
        ]
    }

    pub fn to_bytes(&self) -> Vec<u8> {
        self.to_string().into_bytes()
    }

    pub fn to_byte_record(&self) -> csv::ByteRecord {
        let mut record = csv::ByteRecord::new();
        record.push_field(format!("{:.4}", self.timestamp).as_bytes());
        record.push_field(self.disk_id.to_string().as_bytes());
        record.push_field(self.offset.to_string().as_bytes());
        record.push_field(self.size.to_string().as_bytes());
        record.push_field((self.read as u8).to_string().as_bytes());
        record
    }
}

impl ToString for MsftTrace {
    fn to_string(&self) -> String {
        format!(
            "{:.4} {} {} {} {}",
            self.timestamp, self.disk_id, self.offset, self.size, self.read as u8
        )
    }
}

impl From<&str> for MsftTrace {
    fn from(s: &str) -> Self {
        let v: Vec<&str> = s.split_whitespace().collect();
        let timestamp = v[0].parse::<f64>().unwrap();
        let disk_id = v[1].parse::<u128>().unwrap();
        let offset = v[2].parse::<u128>().unwrap();
        let size = v[3].parse::<u128>().unwrap();
        let read = v[4].parse::<u8>().unwrap() == 1;
        Self {
            timestamp: OrderedFloat::from(timestamp),
            disk_id,
            offset,
            size,
            read,
        }
    }
}

impl From<String> for MsftTrace {
    fn from(s: String) -> Self {
        MsftTrace::from(s.as_str())
    }
}

impl TryFrom<&csv::StringRecord> for MsftTrace {
    type Error = Box<dyn std::error::Error>;

    fn try_from(record: &csv::StringRecord) -> Result<Self, Box<dyn std::error::Error>> {
        let timestamp = record[0].parse::<f64>()?;
        let disk_id = record[1].parse::<u128>()?;
        let offset = record[2].parse::<u128>()?;
        let size = record[3].parse::<u128>()?;
        let read = record[4].parse::<u8>()? == 1;
        Ok(Self {
            timestamp: OrderedFloat::from(timestamp),
            disk_id,
            offset,
            size,
            read,
        })
    }
}

impl TryFrom<&csv::ByteRecord> for MsftTrace {
    type Error = Box<dyn std::error::Error>;

    fn try_from(record: &csv::ByteRecord) -> Result<Self, Box<dyn std::error::Error>> {
        let timestamp = String::from_utf8_lossy(&record[0]).parse::<f64>()?;
        let disk_id = String::from_utf8_lossy(&record[1]).parse::<u128>()?;
        let offset = String::from_utf8_lossy(&record[2]).parse::<u128>()?;
        let size = String::from_utf8_lossy(&record[3]).parse::<u128>()?;
        let read = String::from_utf8_lossy(&record[4]).parse::<u8>()? == 1;
        Ok(Self {
            timestamp: OrderedFloat::from(timestamp),
            disk_id,
            offset,
            size,
            read,
        })
    }
}

impl TryInto<MsftTrace> for csv::StringRecord {
    type Error = Box<dyn std::error::Error>;

    fn try_into(self) -> Result<MsftTrace, Box<dyn std::error::Error>> {
        MsftTrace::try_from(&self)
    }
}

impl TryInto<MsftTrace> for csv::ByteRecord {
    type Error = Box<dyn std::error::Error>;

    fn try_into(self) -> Result<MsftTrace, Box<dyn std::error::Error>> {
        MsftTrace::try_from(&self)
    }
}

pub fn msft_csv_reader_builder<'a>(
    builder: &'a mut csv::ReaderBuilder,
) -> &'a mut csv::ReaderBuilder {
    builder.delimiter(b' ').has_headers(false)
}

pub fn msft_csv_writer_builder<'a>(
    builder: &'a mut csv::WriterBuilder,
) -> &'a mut csv::WriterBuilder {
    builder.delimiter(b' ').has_headers(false)
}
