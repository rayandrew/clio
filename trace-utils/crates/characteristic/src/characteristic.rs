use clio_utils::msft::MsftTrace;
use clio_utils::stats::{Statistic, ToStatistic, STATISTIC_FIELDS};
use ordered_float::OrderedFloat;
use serde::{Deserialize, Serialize};
use std::borrow::Borrow;
use std::convert::TryFrom;
use std::error::Error;

pub const RAW_CHARACTERISTIC_FIELDS_NO_STATISTIC: [&str; 7] = [
    "num_io",
    "start_time",
    "end_time",
    "ts_unit",
    "duration",
    "read_count",
    "write_count",
];

pub const RAW_CHARACTERISTIC_FIELDS_STATISTIC: [&str; 7] = [
    "read_size",
    "write_size",
    "size",
    "offset",
    "iat",
    "read_iat",
    "write_iat",
];

#[derive(Debug, Serialize, Deserialize)]
pub struct RawTraceCharacteristic {
    pub num_io: u64,
    pub start_time: f64,
    pub end_time: f64,
    pub ts_unit: String,
    pub duration: f64,

    // read/write
    pub read_count: u64,
    pub write_count: u64,

    // size
    pub size: Statistic,
    pub read_size: Statistic,
    pub write_size: Statistic,

    pub offset: Statistic,

    // iat
    pub iat: Statistic,
    pub read_iat: Statistic,
    pub write_iat: Statistic,
}

impl Default for RawTraceCharacteristic {
    fn default() -> Self {
        Self {
            num_io: 0,
            start_time: 0.0,
            end_time: 0.0,
            ts_unit: String::new(),
            duration: 0.0,
            read_count: 0,
            write_count: 0,
            size: Statistic::new(),
            read_size: Statistic::new(),
            write_size: Statistic::new(),
            offset: Statistic::new(),
            iat: Statistic::new(),
            read_iat: Statistic::new(),
            write_iat: Statistic::new(),
        }
    }
}

impl RawTraceCharacteristic {
    pub fn new() -> Self {
        Self::default()
    }

    pub fn csv_header() -> csv::ByteRecord {
        let mut record = csv::ByteRecord::new();

        for field in RAW_CHARACTERISTIC_FIELDS_NO_STATISTIC.iter() {
            record.push_field(field.as_bytes());
        }

        for field in RAW_CHARACTERISTIC_FIELDS_STATISTIC.iter() {
            for stat_field in STATISTIC_FIELDS.iter() {
                record.push_field(format!("{}_{}", field, stat_field).as_bytes());
            }
        }

        record
    }
}

impl<R: std::io::Read> TryFrom<&mut csv::Reader<R>> for RawTraceCharacteristic {
    type Error = Box<dyn Error>;

    fn try_from(reader: &mut csv::Reader<R>) -> Result<RawTraceCharacteristic, Box<dyn Error>> {
        let mut iat: Vec<OrderedFloat<f64>> = Vec::new();
        let mut read_iat: Vec<OrderedFloat<f64>> = Vec::new();
        let mut write_iat: Vec<OrderedFloat<f64>> = Vec::new();
        let mut offset: Vec<u128> = Vec::new();
        let mut size: Vec<u128> = Vec::new();
        let mut read_size: Vec<u128> = Vec::new();
        let mut write_size: Vec<u128> = Vec::new();
        let mut read_count = 0;
        let mut write_count = 0;
        let mut num_io = 0;
        let mut start_time = 0.0;
        let mut end_time = 0.0;
        let mut ts_unit = String::new();
        let mut duration = 0.0;

        let mut last_time = 0.0;

        for record in reader.byte_records() {
            let entry = record?;
            let msft = MsftTrace::try_from(&entry)?;

            if num_io == 0 {
                start_time = msft.timestamp;
            }

            num_io += 1;
            end_time = msft.timestamp;
            iat.push(OrderedFloat::from(msft.timestamp - last_time));
            offset.push(msft.offset);
            size.push(msft.size);

            if msft.read {
                read_count += 1;
                read_iat.push(OrderedFloat::from(msft.timestamp - last_time));
                read_size.push(msft.size);
            } else {
                write_count += 1;
                write_iat.push(OrderedFloat::from(msft.timestamp - last_time));
                write_size.push(msft.size);
            }

            last_time = msft.timestamp;
        }

        if num_io > 0 {
            duration = end_time - start_time;
            ts_unit = "ms".to_string();
        }

        let iat = iat.to_statistic()?;
        let read_iat = read_iat.to_statistic()?;
        let write_iat = write_iat.to_statistic()?;
        let offset = offset.to_statistic()?;
        let size = size.to_statistic()?;
        let read_size = read_size.to_statistic()?;
        let write_size = write_size.to_statistic()?;

        Ok(RawTraceCharacteristic {
            num_io,
            start_time,
            end_time,
            ts_unit,
            duration,
            read_count,
            write_count,
            size,
            read_size,
            write_size,
            offset,
            iat,
            read_iat,
            write_iat,
        })
    }
}

// impl TryFrom<RawTraceCharacteristic> for csv::ByteRecord {
//     type Error = Box<dyn Error>;

//     fn try_from(characteristic: RawTraceCharacteristic) -> Result<csv::ByteRecord, Box<dyn Error>> {
//         let mut record = csv::ByteRecord::new();
//         record.push_field(characteristic.num_io.to_string().as_bytes());
//         record.push_field(format!("{:.4}", characteristic.start_time).as_bytes());
//         record.push_field(format!("{:.4}", characteristic.end_time).as_bytes());
//         record.push_field(characteristic.ts_unit.as_bytes());
//         record.push_field(format!("{:.4}", characteristic.duration).as_bytes());
//         record.push_field(characteristic.read_count.to_string().as_bytes());
//         record.push_field(characteristic.write_count.to_string().as_bytes());

//         let read_size_record: csv::ByteRecord = characteristic.read_size.borrow().into();
//         record.extend(read_size_record.iter());

//         let write_size_record: csv::ByteRecord = characteristic.write_size.borrow().into();
//         record.extend(write_size_record.iter());

//         let size_record: csv::ByteRecord = characteristic.size.borrow().into();
//         record.extend(size_record.iter());

//         let offset_record: csv::ByteRecord = characteristic.offset.borrow().into();
//         record.extend(offset_record.iter());

//         let iat_record: csv::ByteRecord = characteristic.iat.borrow().into();
//         record.extend(iat_record.iter());

//         let read_iat_record: csv::ByteRecord = characteristic.read_iat.borrow().into();
//         record.extend(read_iat_record.iter());

//         let write_iat_record: csv::ByteRecord = characteristic.write_iat.borrow().into();
//         record.extend(write_iat_record.iter());

//         Ok(record)
//     }
// }

impl TryInto<csv::ByteRecord> for &RawTraceCharacteristic {
    type Error = Box<dyn Error>;

    fn try_into(self) -> Result<csv::ByteRecord, Box<dyn Error>> {
        let mut record = csv::ByteRecord::new();
        record.push_field(self.num_io.to_string().as_bytes());
        record.push_field(format!("{:.4}", self.start_time).as_bytes());
        record.push_field(format!("{:.4}", self.end_time).as_bytes());
        record.push_field(self.ts_unit.as_bytes());
        record.push_field(format!("{:.4}", self.duration).as_bytes());
        record.push_field(self.read_count.to_string().as_bytes());
        record.push_field(self.write_count.to_string().as_bytes());

        let read_size_record: csv::ByteRecord = self.read_size.borrow().into();
        record.extend(read_size_record.iter());

        let write_size_record: csv::ByteRecord = self.write_size.borrow().into();
        record.extend(write_size_record.iter());

        let size_record: csv::ByteRecord = self.size.borrow().into();
        record.extend(size_record.iter());

        let offset_record: csv::ByteRecord = self.offset.borrow().into();
        record.extend(offset_record.iter());

        let iat_record: csv::ByteRecord = self.iat.borrow().into();
        record.extend(iat_record.iter());

        let read_iat_record: csv::ByteRecord = self.read_iat.borrow().into();
        record.extend(read_iat_record.iter());

        let write_iat_record: csv::ByteRecord = self.write_iat.borrow().into();
        record.extend(write_iat_record.iter());

        Ok(record)
    }
}

// impl TryInto<csv::ByteRecord> for RawTraceCharacteristic {
//     type Error = Box<dyn Error>;

//     fn try_into(self) -> Result<csv::ByteRecord, Box<dyn Error>> {
//         let record = csv::ByteRecord::try_from(self)?;
//         Ok(record)
//     }
// }
