use crate::characteristic::RawTraceCharacteristic;
use clio_utils::msft::MsftTrace;
use clio_utils::stats::ToStatistic;
use ordered_float::OrderedFloat;
use std::convert::TryInto;
use std::error::Error;

impl TryInto<RawTraceCharacteristic> for Vec<MsftTrace> {
    type Error = Box<dyn Error>;

    fn try_into(self) -> Result<RawTraceCharacteristic, Box<dyn Error>> {
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

        for msft in self {
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
