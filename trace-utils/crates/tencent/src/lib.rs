use std::error::Error;
use std::path::Path;

use clio_utils::path::{is_gzip_from_path, is_tar_gz};

pub trait TencentTraceTrait {
    fn read(
        &self,
        process: impl FnMut(&csv::StringRecord) -> Result<(), Box<dyn Error>>,
    ) -> Result<(), Box<dyn Error>>;
}

pub struct TencentTraceCsv<'a, P: AsRef<Path>> {
    path: P,
    filter: Box<dyn Fn(&csv::StringRecord) -> bool + 'a>,
    csv_builder: Box<dyn Fn(&mut csv::ReaderBuilder) -> &mut csv::ReaderBuilder + 'a>,
}

impl<'a, P: AsRef<Path>> TencentTraceCsv<'a, P> {
    pub fn new(path: P) -> TencentTraceCsv<'a, P> {
        Self {
            path: path,
            filter: Box::new(|_| true),
            csv_builder: Box::new(|builder| builder.has_headers(false)),
        }
    }

    pub fn with_filter(
        &mut self,
        filter: impl Fn(&csv::StringRecord) -> bool + 'a,
    ) -> &TencentTraceCsv<'a, P> {
        self.filter = Box::new(filter);
        self
    }

    pub fn with_csv_builder(
        &mut self,
        csv_builder: impl Fn(&mut csv::ReaderBuilder) -> &mut csv::ReaderBuilder + 'a,
    ) -> &TencentTraceCsv<'a, P> {
        self.csv_builder = Box::new(csv_builder);
        self
    }
}

impl<'a, P: AsRef<Path>> TencentTraceTrait for TencentTraceCsv<'a, P> {
    fn read(
        &self,
        mut process: impl FnMut(&csv::StringRecord) -> Result<(), Box<dyn Error>>,
    ) -> Result<(), Box<dyn Error>> {
        let mut csv_builder = csv::ReaderBuilder::new();
        let csv_builder = (self.csv_builder)(&mut csv_builder);
        let mut rdr = csv_builder.from_path(&self.path)?;
        for result in rdr.records() {
            let record = result?;

            if !(self.filter)(&record) {
                continue;
            }
            process(&record)?;
        }
        Ok(())
    }
}

pub struct TencentTraceGz<'a, P: AsRef<Path>> {
    path: P,
    filter: Box<dyn Fn(&csv::StringRecord) -> bool + 'a>,
    csv_builder: Box<dyn Fn(&mut csv::ReaderBuilder) -> &mut csv::ReaderBuilder + 'a>,
}

impl<'a, P: AsRef<Path>> TencentTraceGz<'a, P> {
    pub fn new(path: P) -> TencentTraceGz<'a, P> {
        Self {
            path: path,
            filter: Box::new(|_| true),
            csv_builder: Box::new(|builder| builder.has_headers(false)),
        }
    }

    pub fn with_filter(
        &mut self,
        filter: impl Fn(&csv::StringRecord) -> bool + 'a,
    ) -> &TencentTraceGz<'a, P> {
        self.filter = Box::new(filter);
        self
    }

    pub fn with_csv_builder(
        &mut self,
        csv_builder: impl Fn(&mut csv::ReaderBuilder) -> &mut csv::ReaderBuilder + 'a,
    ) -> &TencentTraceGz<'a, P> {
        self.csv_builder = Box::new(csv_builder);
        self
    }
}

impl<'a, P: AsRef<Path>> TencentTraceTrait for TencentTraceGz<'a, P> {
    fn read(
        &self,
        mut process: impl FnMut(&csv::StringRecord) -> Result<(), Box<dyn Error>>,
    ) -> Result<(), Box<dyn Error>> {
        let mut decoder = flate2::read::GzDecoder::new(std::fs::File::open(&self.path)?);
        let mut csv_builder = csv::ReaderBuilder::new();
        let csv_builder = (self.csv_builder)(&mut csv_builder);
        let mut rdr = csv_builder.from_reader(&mut decoder);
        for result in rdr.records() {
            let record = result?;
            if !(self.filter)(&record) {
                continue;
            }
            process(&record)?;
        }
        Ok(())
    }
}

pub struct TencentTraceTarGz<'a, P: AsRef<Path>> {
    path: P,
    filter: Box<dyn Fn(&csv::StringRecord) -> bool + 'a>,
    csv_builder: Box<dyn Fn(&mut csv::ReaderBuilder) -> &mut csv::ReaderBuilder + 'a>,
}

impl<'a, P: AsRef<Path>> TencentTraceTarGz<'a, P> {
    pub fn new(path: P) -> Self {
        Self {
            path: path,
            filter: Box::new(|_| true),
            csv_builder: Box::new(|builder| builder.has_headers(false)),
        }
    }

    pub fn with_filter(
        &mut self,
        filter: impl Fn(&csv::StringRecord) -> bool + 'a,
    ) -> &TencentTraceTarGz<'a, P> {
        self.filter = Box::new(filter);
        self
    }

    pub fn with_csv_builder(
        &mut self,
        csv_builder: impl Fn(&mut csv::ReaderBuilder) -> &mut csv::ReaderBuilder + 'a,
    ) -> &TencentTraceTarGz<'a, P> {
        self.csv_builder = Box::new(csv_builder);
        self
    }
}

impl<'a, P: AsRef<Path>> TencentTraceTrait for TencentTraceTarGz<'a, P> {
    fn read(
        &self,
        mut process: impl FnMut(&csv::StringRecord) -> Result<(), Box<dyn Error>>,
    ) -> Result<(), Box<dyn Error>> {
        let decoder = flate2::read::GzDecoder::new(std::fs::File::open(&self.path)?);
        let mut tar = tar::Archive::new(decoder);
        let files = tar.entries()?;
        for entry in files {
            let entry = entry?;
            let mut csv_builder = csv::ReaderBuilder::new();
            let csv_builder = (self.csv_builder)(&mut csv_builder);
            let mut rdr = csv_builder.from_reader(entry);
            for result in rdr.records() {
                let record = result?;
                if !(self.filter)(&record) {
                    continue;
                }
                process(&record)?;
            }
        }
        Ok(())
    }
}

pub enum TencentTraceBuilder<'a, P: AsRef<Path>> {
    Csv(TencentTraceCsv<'a, P>),
    Gz(TencentTraceGz<'a, P>),
    TarGz(TencentTraceTarGz<'a, P>),
}

impl<'a, P: AsRef<Path>> TencentTraceBuilder<'a, P> {
    pub fn new(path: P) -> Result<Self, Box<dyn Error>> {
        let p = path.as_ref();
        if is_gzip_from_path(p) {
            Ok(Self::Gz(TencentTraceGz::new(path)))
        } else if is_tar_gz(p)? {
            Ok(Self::TarGz(TencentTraceTarGz::new(path)))
        } else {
            Ok(Self::Csv(TencentTraceCsv::new(path)))
        }
    }

    pub fn with_filter(
        &mut self,
        filter: impl Fn(&csv::StringRecord) -> bool + 'a,
    ) -> &TencentTraceBuilder<'a, P> {
        if let Self::Csv(trace) = self {
            trace.with_filter(filter);
        } else if let Self::Gz(trace) = self {
            trace.with_filter(filter);
        } else if let Self::TarGz(trace) = self {
            trace.with_filter(filter);
        }

        self
    }

    pub fn with_csv_builder(
        &mut self,
        csv_builder: impl Fn(&mut csv::ReaderBuilder) -> &mut csv::ReaderBuilder + 'a,
    ) -> &TencentTraceBuilder<'a, P> {
        if let Self::Csv(trace) = self {
            trace.with_csv_builder(csv_builder);
        } else if let Self::Gz(trace) = self {
            trace.with_csv_builder(csv_builder);
        } else if let Self::TarGz(trace) = self {
            trace.with_csv_builder(csv_builder);
        }

        self
    }
}

impl<'a, P: AsRef<Path>> TencentTraceTrait for TencentTraceBuilder<'a, P> {
    fn read(
        &self,
        process: impl FnMut(&csv::StringRecord) -> Result<(), Box<dyn Error>>,
    ) -> Result<(), Box<dyn Error>> {
        match self {
            Self::Csv(trace) => trace.read(process),
            Self::Gz(trace) => trace.read(process),
            Self::TarGz(trace) => trace.read(process),
        }
    }
}
