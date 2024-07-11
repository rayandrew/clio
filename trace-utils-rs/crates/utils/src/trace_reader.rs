use std::error::Error;
use std::path::Path;

use crate::path::{is_gzip, is_tar_gz};

pub trait TraceReaderTrait {
    fn read(
        &self,
        process: impl FnMut(&csv::StringRecord) -> Result<(), Box<dyn Error>>,
    ) -> Result<(), Box<dyn Error>>;
}

pub struct TraceReaderCsv<'a, P: AsRef<Path>> {
    path: P,
    filter: Box<dyn Fn(&csv::StringRecord) -> bool + 'a>,
    csv_builder: Box<dyn Fn(&mut csv::ReaderBuilder) -> &mut csv::ReaderBuilder + 'a>,
}

impl<'a, P: AsRef<Path>> TraceReaderCsv<'a, P> {
    pub fn new(path: P) -> TraceReaderCsv<'a, P> {
        Self {
            path: path,
            filter: Box::new(|_| true),
            csv_builder: Box::new(|builder| builder.has_headers(false)),
        }
    }

    pub fn with_filter(
        &mut self,
        filter: impl Fn(&csv::StringRecord) -> bool + 'a,
    ) -> &TraceReaderCsv<'a, P> {
        self.filter = Box::new(filter);
        self
    }

    pub fn with_csv_builder(
        &mut self,
        csv_builder: impl Fn(&mut csv::ReaderBuilder) -> &mut csv::ReaderBuilder + 'a,
    ) -> &TraceReaderCsv<'a, P> {
        self.csv_builder = Box::new(csv_builder);
        self
    }

    pub fn get_reader(&self) -> Result<csv::Reader<std::fs::File>, Box<dyn Error>> {
        let mut csv_builder = csv::ReaderBuilder::new();
        let csv_builder = (self.csv_builder)(&mut csv_builder);
        let reader = csv_builder.from_path(&self.path)?;
        Ok(reader)
    }
}

impl<'a, P: AsRef<Path>> TraceReaderTrait for TraceReaderCsv<'a, P> {
    fn read(
        &self,
        mut process: impl FnMut(&csv::StringRecord) -> Result<(), Box<dyn Error>>,
    ) -> Result<(), Box<dyn Error>> {
        let mut reader = self.get_reader()?;
        for result in reader.records() {
            let record = result?;

            if !(self.filter)(&record) {
                continue;
            }
            process(&record)?;
        }
        Ok(())
    }
}

pub struct TraceReaderGz<'a, P: AsRef<Path>> {
    path: P,
    filter: Box<dyn Fn(&csv::StringRecord) -> bool + 'a>,
    csv_builder: Box<dyn Fn(&mut csv::ReaderBuilder) -> &mut csv::ReaderBuilder + 'a>,
}

impl<'a, P: AsRef<Path>> TraceReaderGz<'a, P> {
    pub fn new(path: P) -> TraceReaderGz<'a, P> {
        Self {
            path: path,
            filter: Box::new(|_| true),
            csv_builder: Box::new(|builder| builder.has_headers(false)),
        }
    }

    pub fn with_filter(
        &mut self,
        filter: impl Fn(&csv::StringRecord) -> bool + 'a,
    ) -> &TraceReaderGz<'a, P> {
        self.filter = Box::new(filter);
        self
    }

    pub fn with_csv_builder(
        &mut self,
        csv_builder: impl Fn(&mut csv::ReaderBuilder) -> &mut csv::ReaderBuilder + 'a,
    ) -> &TraceReaderGz<'a, P> {
        self.csv_builder = Box::new(csv_builder);
        self
    }

    pub fn get_reader(
        &self,
    ) -> Result<csv::Reader<flate2::read::GzDecoder<std::fs::File>>, Box<dyn Error>> {
        let decoder = flate2::read::GzDecoder::new(std::fs::File::open(&self.path)?);
        let mut csv_builder = csv::ReaderBuilder::new();
        let csv_builder = (self.csv_builder)(&mut csv_builder);
        let reader = csv_builder.from_reader(decoder);
        Ok(reader)
    }
}

impl<'a, P: AsRef<Path>> TraceReaderTrait for TraceReaderGz<'a, P> {
    fn read(
        &self,
        mut process: impl FnMut(&csv::StringRecord) -> Result<(), Box<dyn Error>>,
    ) -> Result<(), Box<dyn Error>> {
        let mut reader = self.get_reader()?;
        for result in reader.records() {
            let record = result?;
            if !(self.filter)(&record) {
                continue;
            }
            process(&record)?;
        }
        Ok(())
    }
}

pub struct TraceReaderTarGz<'a, P: AsRef<Path>> {
    path: P,
    filter: Box<dyn Fn(&csv::StringRecord) -> bool + 'a>,
    csv_builder: Box<dyn Fn(&mut csv::ReaderBuilder) -> &mut csv::ReaderBuilder + 'a>,
}

impl<'a, P: AsRef<Path>> TraceReaderTarGz<'a, P> {
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
    ) -> &TraceReaderTarGz<'a, P> {
        self.filter = Box::new(filter);
        self
    }

    pub fn with_csv_builder(
        &mut self,
        csv_builder: impl Fn(&mut csv::ReaderBuilder) -> &mut csv::ReaderBuilder + 'a,
    ) -> &TraceReaderTarGz<'a, P> {
        self.csv_builder = Box::new(csv_builder);
        self
    }

    pub fn get_tar_reader(
        &self,
    ) -> Result<tar::Archive<flate2::read::GzDecoder<std::fs::File>>, Box<dyn Error>> {
        let decoder = flate2::read::GzDecoder::new(std::fs::File::open(&self.path)?);
        let tar = tar::Archive::new(decoder);
        Ok(tar)
    }
}

impl<'a, P: AsRef<Path>> TraceReaderTrait for TraceReaderTarGz<'a, P> {
    fn read(
        &self,
        mut process: impl FnMut(&csv::StringRecord) -> Result<(), Box<dyn Error>>,
    ) -> Result<(), Box<dyn Error>> {
        let mut tar = self.get_tar_reader()?;
        let files = tar.entries()?;
        for entry in files {
            let entry = entry?;
            // let path = entry.path()?;
            // let trace = TraceReaderCsv {
            //     path: path,
            //     filter: self.filter,
            //     csv_builder: self.csv_builder,
            // };
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

pub enum TraceReaderBuilder<'a, P: AsRef<Path>> {
    Csv(TraceReaderCsv<'a, P>),
    Gz(TraceReaderGz<'a, P>),
    TarGz(TraceReaderTarGz<'a, P>),
}

impl<'a, P: AsRef<Path>> TraceReaderBuilder<'a, P> {
    pub fn new(path: P) -> Result<Self, Box<dyn Error>> {
        let p = path.as_ref();
        if is_tar_gz(p)? {
            Ok(Self::TarGz(TraceReaderTarGz::new(path)))
        } else if is_gzip(p)? {
            Ok(Self::Gz(TraceReaderGz::new(path)))
        } else {
            Ok(Self::Csv(TraceReaderCsv::new(path)))
        }
    }

    pub fn with_filter(
        &mut self,
        filter: impl Fn(&csv::StringRecord) -> bool + 'a,
    ) -> &TraceReaderBuilder<'a, P> {
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
    ) -> &TraceReaderBuilder<'a, P> {
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

impl<'a, P: AsRef<Path>> TraceReaderTrait for TraceReaderBuilder<'a, P> {
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
