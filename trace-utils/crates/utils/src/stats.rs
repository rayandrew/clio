use average::{Estimate, Quantile};
use core::hash::Hash;
use hashbrown::HashMap;
use num::{Num, ToPrimitive};
use ordered_float::OrderedFloat;
use serde::{Deserialize, Serialize};
use std::error::Error;

pub const STATISTIC_FIELDS: [&str; 25] = [
    "avg", "max", "min", "sum", "count", "mode", "median", "variance", "std_dev", "p10", "p20",
    "p25", "p30", "p40", "p50", "p60", "p70", "p75", "p80", "p90", "p95", "p99", "p999", "p9999",
    "p100",
];

#[derive(Debug, Serialize, Deserialize)]
pub struct Statistic {
    pub avg: f64,
    pub max: f64,
    pub min: f64,
    pub sum: f64,
    pub count: u64,
    pub mode: f64,
    pub median: f64,
    pub variance: f64,
    pub std_dev: f64,
    // percentile
    pub p10: f64,
    pub p20: f64,
    pub p25: f64,
    pub p30: f64,
    pub p40: f64,
    pub p50: f64,
    pub p60: f64,
    pub p70: f64,
    pub p75: f64,
    pub p80: f64,
    pub p90: f64,
    pub p95: f64,
    pub p99: f64,
    pub p999: f64,
    pub p9999: f64,
    pub p100: f64,
}

impl Default for Statistic {
    fn default() -> Self {
        Self {
            avg: 0.0,
            max: 0.0,
            min: 0.0,
            sum: 0.0,
            count: 0,
            mode: 0.0,
            median: 0.0,
            variance: 0.0,
            std_dev: 0.0,
            p10: 0.0,
            p20: 0.0,
            p25: 0.0,
            p30: 0.0,
            p40: 0.0,
            p50: 0.0,
            p60: 0.0,
            p70: 0.0,
            p75: 0.0,
            p80: 0.0,
            p90: 0.0,
            p95: 0.0,
            p99: 0.0,
            p999: 0.0,
            p9999: 0.0,
            p100: 0.0,
        }
    }
}

impl Statistic {
    pub fn new() -> Self {
        Self::default()
    }

    pub fn to_hashmap(&self) -> HashMap<String, String> {
        self.into()
    }
}

impl Into<HashMap<String, String>> for &Statistic {
    fn into(self) -> HashMap<String, String> {
        let mut row = HashMap::new();
        row.insert("avg".to_string(), self.avg.to_string());
        row.insert("max".to_string(), self.max.to_string());
        row.insert("min".to_string(), self.min.to_string());
        row.insert("sum".to_string(), self.sum.to_string());
        row.insert("count".to_string(), self.count.to_string());
        row.insert("mode".to_string(), self.mode.to_string());
        row.insert("median".to_string(), self.median.to_string());
        row.insert("variance".to_string(), self.variance.to_string());
        row.insert("std_dev".to_string(), self.std_dev.to_string());
        row.insert("p10".to_string(), self.p10.to_string());
        row.insert("p20".to_string(), self.p20.to_string());
        row.insert("p25".to_string(), self.p25.to_string());
        row.insert("p30".to_string(), self.p30.to_string());
        row.insert("p40".to_string(), self.p40.to_string());
        row.insert("p50".to_string(), self.p50.to_string());
        row.insert("p60".to_string(), self.p60.to_string());
        row.insert("p70".to_string(), self.p70.to_string());
        row.insert("p75".to_string(), self.p75.to_string());
        row.insert("p80".to_string(), self.p80.to_string());
        row.insert("p90".to_string(), self.p90.to_string());
        row.insert("p95".to_string(), self.p95.to_string());
        row.insert("p99".to_string(), self.p99.to_string());
        row.insert("p999".to_string(), self.p999.to_string());
        row.insert("p9999".to_string(), self.p9999.to_string());
        row.insert("p100".to_string(), self.p100.to_string());
        row
    }
}

impl Into<csv::ByteRecord> for &Statistic {
    fn into(self) -> csv::ByteRecord {
        let mut record = csv::ByteRecord::new();
        record.push_field(self.avg.to_string().as_bytes());
        record.push_field(self.max.to_string().as_bytes());
        record.push_field(self.min.to_string().as_bytes());
        record.push_field(self.sum.to_string().as_bytes());
        record.push_field(self.count.to_string().as_bytes());
        record.push_field(self.mode.to_string().as_bytes());
        record.push_field(self.median.to_string().as_bytes());
        record.push_field(self.variance.to_string().as_bytes());
        record.push_field(self.std_dev.to_string().as_bytes());
        record.push_field(self.p10.to_string().as_bytes());
        record.push_field(self.p20.to_string().as_bytes());
        record.push_field(self.p25.to_string().as_bytes());
        record.push_field(self.p30.to_string().as_bytes());
        record.push_field(self.p40.to_string().as_bytes());
        record.push_field(self.p50.to_string().as_bytes());
        record.push_field(self.p60.to_string().as_bytes());
        record.push_field(self.p70.to_string().as_bytes());
        record.push_field(self.p75.to_string().as_bytes());
        record.push_field(self.p80.to_string().as_bytes());
        record.push_field(self.p90.to_string().as_bytes());
        record.push_field(self.p95.to_string().as_bytes());
        record.push_field(self.p99.to_string().as_bytes());
        record.push_field(self.p999.to_string().as_bytes());
        record.push_field(self.p9999.to_string().as_bytes());
        record.push_field(self.p100.to_string().as_bytes());
        record
    }
}

pub trait ToStatistic<T> {
    fn to_statistic(self) -> Result<Statistic, Box<dyn Error>>;
}

impl<T> ToStatistic<T> for Vec<T>
where
    T: Num + Ord + ToPrimitive + Hash + ToString,
{
    fn to_statistic(self) -> Result<Statistic, Box<dyn Error>> {
        if self.is_empty() {
            return Ok(Statistic::default());
        }

        let mut data = self;
        data.sort_by(|a, b| a.partial_cmp(b).unwrap());

        let mut stats = Statistic::new();
        let mut map = HashMap::new();
        let mut p10 = Quantile::new(0.1);
        let mut p20 = Quantile::new(0.2);
        let mut p25 = Quantile::new(0.25);
        let mut p30 = Quantile::new(0.3);
        let mut p40 = Quantile::new(0.4);
        let mut p50 = Quantile::new(0.5);
        let mut p60 = Quantile::new(0.6);
        let mut p70 = Quantile::new(0.7);
        let mut p75 = Quantile::new(0.75);
        let mut p80 = Quantile::new(0.8);
        let mut p90 = Quantile::new(0.9);
        let mut p95 = Quantile::new(0.95);
        let mut p99 = Quantile::new(0.99);
        let mut p999 = Quantile::new(0.999);
        let mut p9999 = Quantile::new(0.9999);
        let mut p100 = Quantile::new(1.0);

        for (i, v) in data.iter().enumerate() {
            let v = v.to_f64().unwrap();
            stats.sum += v;
            stats.count += 1;
            if i == 0 {
                stats.min = v;
            }
            if i == data.len() - 1 {
                stats.max = v;
            }
            map.entry(OrderedFloat::from(v))
                .and_modify(|v| *v += 1)
                .or_insert(1);
            p10.add(v);
            p20.add(v);
            p25.add(v);
            p30.add(v);
            p40.add(v);
            p50.add(v);
            p60.add(v);
            p70.add(v);
            p75.add(v);
            p80.add(v);
            p90.add(v);
            p95.add(v);
            p99.add(v);
            p999.add(v);
            p9999.add(v);
            p100.add(v);
        }

        stats.avg = stats.sum / stats.count as f64;
        stats.median = if stats.count % 2 == 0 {
            (data[stats.count as usize / 2].to_f64().unwrap()
                + data[stats.count as usize / 2 - 1].to_f64().unwrap())
                / 2.0
        } else {
            data[stats.count as usize / 2].to_f64().unwrap()
        };
        stats.mode = map.iter().max_by_key(|&(_, v)| v).unwrap().0.into_inner();
        stats.variance = data
            .iter()
            .map(|v| (v.to_f64().unwrap() - stats.avg).powi(2))
            .sum::<f64>()
            / stats.count as f64;
        stats.std_dev = stats.variance.sqrt();

        stats.p10 = p10.quantile();
        stats.p20 = p20.quantile();
        stats.p25 = p25.quantile();
        stats.p30 = p30.quantile();
        stats.p40 = p40.quantile();
        stats.p50 = p50.quantile();
        stats.p60 = p60.quantile();
        stats.p70 = p70.quantile();
        stats.p75 = p75.quantile();
        stats.p80 = p80.quantile();
        stats.p90 = p90.quantile();
        stats.p95 = p95.quantile();
        stats.p99 = p99.quantile();
        stats.p999 = p999.quantile();
        stats.p9999 = p9999.quantile();
        stats.p100 = p100.quantile();

        // stats.p10 = data[(stats.count as f64 * 0.1) as usize].to_f64().unwrap();
        // stats.p20 = data[(stats.count as f64 * 0.2) as usize].to_f64().unwrap();
        // stats.p25 = data[(stats.count as f64 * 0.25) as usize].to_f64().unwrap();
        // stats.p30 = data[(stats.count as f64 * 0.3) as usize].to_f64().unwrap();
        // stats.p40 = data[(stats.count as f64 * 0.4) as usize].to_f64().unwrap();
        // stats.p50 = data[(stats.count as f64 * 0.5) as usize].to_f64().unwrap();
        // stats.p60 = data[(stats.count as f64 * 0.6) as usize].to_f64().unwrap();
        // stats.p70 = data[(stats.count as f64 * 0.7) as usize].to_f64().unwrap();
        // stats.p75 = data[(stats.count as f64 * 0.75) as usize].to_f64().unwrap();
        // stats.p80 = data[(stats.count as f64 * 0.8) as usize].to_f64().unwrap();
        // stats.p90 = data[(stats.count as f64 * 0.9) as usize].to_f64().unwrap();
        // stats.p95 = data[(stats.count as f64 * 0.95) as usize].to_f64().unwrap();
        // stats.p99 = data[(stats.count as f64 * 0.99) as usize].to_f64().unwrap();
        // stats.p999 = data[(stats.count as f64 * 0.999) as usize]
        //     .to_f64()
        //     .unwrap();
        // stats.p9999 = data[(stats.count as f64 * 0.9999) as usize]
        //     .to_f64()
        //     .unwrap();
        // stats.p100 = data[(stats.count as f64 * 1.0) as usize].to_f64().unwrap();

        Ok(stats)
    }
}
