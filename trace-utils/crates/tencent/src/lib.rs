use std::error::Error;
use std::path::PathBuf;
// use trace_utils::{is_csv_from_path, is_gzip, is_gzip_from_path, is_tar_gz, is_tgz_from_path};

pub trait TencentTraceTrait {
    fn read(
        &self,
        p: PathBuf,
        process: impl FnMut(&csv::StringRecord) -> Result<(), Box<dyn Error>>,
        // writer: &mut csv::Writer<T>,
    ) -> Result<(), Box<dyn Error>>;
}

// pub struct TencentTraceDefault {
//     filter: fn(&csv::StringRecord) -> bool,
// }

// impl Default for TencentTraceDefault {
//     fn default() -> Self {
//         Self::new()
//     }
// }

// impl TencentTraceDefault {
//     pub fn new() -> Self {
//         Self { filter: |_| true }
//     }

//     pub fn with_filter(filter: fn(&csv::StringRecord) -> bool) -> Self {
//         Self { filter: filter }
//     }
// }

// impl TencentTraceTrait for TencentTraceDefault {
//     fn read<T: std::io::Write>(
//         &self,
//         p: PathBuf,
//         writer: &mut csv::Writer<T>,
//     ) -> Result<(), Box<dyn Error>> {
//         let mut rdr = csv::ReaderBuilder::new().has_headers(false).from_path(p)?;
//         for (i, result) in rdr.records().enumerate() {
//             let record = result?;

//             if !(self.filter)(&record) {
//                 continue;
//             }
//             writer.write_record(&record)?;
//             if i % 10_000_000 == 0 && i > 0 {
//                 println!("Reaching record: {}", i);
//             }
//         }
//         writer.flush()?;
//         Ok(())
//     }
// }

// pub struct TencentTraceGz {
//     filter: fn(&csv::StringRecord) -> bool,
// }

// impl Default for TencentTraceGz {
//     fn default() -> Self {
//         Self::new()
//     }
// }

// impl TencentTraceGz {
//     pub fn new() -> Self {
//         Self { filter: |_| true }
//     }

//     pub fn with_filter(filter: fn(&csv::StringRecord) -> bool) -> Self {
//         Self { filter: filter }
//     }
// }

// impl TencentTraceTrait for TencentTraceGz {
//     fn read<T: std::io::Write>(
//         &self,
//         p: PathBuf,
//         writer: &mut csv::Writer<T>,
//     ) -> Result<(), Box<dyn Error>> {
//         let mut decoder = flate2::read::GzDecoder::new(std::fs::File::open(p)?);
//         let mut rdr = csv::ReaderBuilder::new()
//             .has_headers(false)
//             .from_reader(&mut decoder);
//         for (i, result) in rdr.records().enumerate() {
//             let record = result?;
//             if !(self.filter)(&record) {
//                 continue;
//             }
//             writer.write_record(&record)?;
//             if i % 10_000_000 == 0 && i > 0 {
//                 println!("Reaching record: {}", i);
//             }
//         }
//         writer.flush()?;
//         Ok(())
//     }
// }

pub struct TencentTraceTarGz {
    pub filter: Box<dyn Fn(&csv::StringRecord) -> bool>,
    // pub decoder: tar::Archive<flate2::read::GzDecoder<std::io::BufReader<std::fs::File>>>,
}

impl Default for TencentTraceTarGz {
    fn default() -> Self {
        Self {
            filter: Box::new(|_| true),
        }
    }
}

impl TencentTraceTarGz {
    pub fn new() -> Self {
        Self::default()
    }

    pub fn with_filter(
        &mut self,
        filter: impl Fn(&csv::StringRecord) -> bool + 'static,
    ) -> &mut Self {
        self.filter = Box::new(filter);
        self
    }
}

impl TencentTraceTrait for TencentTraceTarGz {
    fn read(
        &self,
        p: PathBuf,
        mut process: impl FnMut(&csv::StringRecord) -> Result<(), Box<dyn Error>>,
    ) -> Result<(), Box<dyn Error>> {
        let decoder = flate2::read::GzDecoder::new(std::fs::File::open(p)?);
        let mut tar = tar::Archive::new(decoder);
        let files = tar.entries()?;
        for entry in files {
            let entry = entry?;
            let mut rdr = csv::ReaderBuilder::new()
                .has_headers(false)
                .from_reader(entry);
            for (i, result) in rdr.records().enumerate() {
                let record = result?;
                if !(self.filter)(&record) {
                    continue;
                }
                process(&record)?;
                if i % 10_000_000 == 0 && i > 0 {
                    println!("Reaching record: {}", i);
                }
            }
        }
        Ok(())
    }
}

// // pub fn tencent_trace_factory(p: &PathBuf) -> TencentTraceTrait {

// //     if is_csv_from_path(p) {
// //         return Tr
// //     } else if is_gzip_from_path(p) && is_gzip(p).unwrap() {
// //         return Box::new(TencentTraceGz::new());
// //     } else if is_tgz_from_path(p) || is_tar_gz(p).unwrap() {
// //         return Box::new(TencentTraceTarGz::new());
// //     }

// //     Box::new(TencentTraceDefault::new())
// // }
