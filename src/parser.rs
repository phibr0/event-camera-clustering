mod evt21;
mod evt3;

use crate::Event;
use crate::Result;
use crate::evt2::{DecodedEvt2, Evt2Reader, ExtTrigger, RawHeader, read_raw_header};
use evt3::Evt3EventStream;
use evt21::Evt21EventStream;
use std::fmt;
use std::fs::File;
use std::io::{BufRead, BufReader};
use std::path::Path;
use std::str::FromStr;

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum Endian {
    Little,
    Little32,
    Big,
}

impl FromStr for Endian {
    type Err = String;

    fn from_str(value: &str) -> std::result::Result<Self, Self::Err> {
        match value {
            "little" => Ok(Self::Little),
            "little32" => Ok(Self::Little32),
            "big" => Ok(Self::Big),
            _ => Err("endian must be little, little32, or big".to_owned()),
        }
    }
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum EventFormat {
    Auto,
    Evt2,
    Evt21,
    Evt3,
}

impl Default for EventFormat {
    fn default() -> Self {
        Self::Auto
    }
}

impl fmt::Display for EventFormat {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Self::Auto => f.write_str("auto"),
            Self::Evt2 => f.write_str("evt2"),
            Self::Evt21 => f.write_str("evt21"),
            Self::Evt3 => f.write_str("evt3"),
        }
    }
}

impl FromStr for EventFormat {
    type Err = String;

    fn from_str(value: &str) -> std::result::Result<Self, Self::Err> {
        match value {
            "auto" => Ok(Self::Auto),
            "evt2" | "evt2.0" | "2.0" => Ok(Self::Evt2),
            "evt21" | "evt2.1" | "2.1" => Ok(Self::Evt21),
            "evt3" | "evt3.0" | "3.0" => Ok(Self::Evt3),
            _ => Err("format must be auto, evt2, evt21, or evt3".to_owned()),
        }
    }
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum EventRecord {
    Event(Event),
    ExtTrigger(ExtTrigger),
    Other,
}

pub trait EventStream {
    fn next_record(&mut self) -> Result<Option<EventRecord>>;
}

pub struct OpenedEventStream {
    pub header: RawHeader,
    pub format: EventFormat,
    pub stream: Box<dyn EventStream>,
}

pub fn open_event_stream(
    path: impl AsRef<Path>,
    requested_format: EventFormat,
    endian: Endian,
) -> Result<OpenedEventStream> {
    match requested_format {
        EventFormat::Auto => open_auto_stream(path, endian),
        EventFormat::Evt2 => open_evt2_stream(path, endian),
        EventFormat::Evt21 => open_evt21_stream(path, endian),
        EventFormat::Evt3 => open_evt3_stream(path, endian),
    }
}

fn open_auto_stream(path: impl AsRef<Path>, endian: Endian) -> Result<OpenedEventStream> {
    let file = File::open(path)?;
    let mut reader = BufReader::new(file);
    let header = read_raw_header(&mut reader)?;

    match format_from_header(&header).unwrap_or(EventFormat::Evt2) {
        EventFormat::Evt2 => Ok(OpenedEventStream {
            header,
            format: EventFormat::Evt2,
            stream: Box::new(Evt2EventStream {
                reader: Evt2Reader::from_reader(reader, endian),
            }),
        }),
        EventFormat::Evt21 => Ok(OpenedEventStream {
            header,
            format: EventFormat::Evt21,
            stream: Box::new(Evt21EventStream::new(reader, endian)),
        }),
        EventFormat::Evt3 => Ok(OpenedEventStream {
            header,
            format: EventFormat::Evt3,
            stream: Box::new(Evt3EventStream::new(reader, endian)),
        }),
        EventFormat::Auto => unreachable!(),
    }
}

fn open_evt2_stream(path: impl AsRef<Path>, endian: Endian) -> Result<OpenedEventStream> {
    let (header, reader) = Evt2Reader::from_path(path, endian)?;
    Ok(OpenedEventStream {
        header,
        format: EventFormat::Evt2,
        stream: Box::new(Evt2EventStream { reader }),
    })
}

fn open_evt21_stream(path: impl AsRef<Path>, endian: Endian) -> Result<OpenedEventStream> {
    let file = File::open(path)?;
    let mut reader = BufReader::new(file);
    let header = read_raw_header(&mut reader)?;
    Ok(OpenedEventStream {
        header,
        format: EventFormat::Evt21,
        stream: Box::new(Evt21EventStream::new(reader, endian)),
    })
}

fn open_evt3_stream(path: impl AsRef<Path>, endian: Endian) -> Result<OpenedEventStream> {
    let file = File::open(path)?;
    let mut reader = BufReader::new(file);
    let header = read_raw_header(&mut reader)?;
    Ok(OpenedEventStream {
        header,
        format: EventFormat::Evt3,
        stream: Box::new(Evt3EventStream::new(reader, endian)),
    })
}

fn format_from_header(header: &RawHeader) -> Option<EventFormat> {
    match header.evt_version.as_deref() {
        Some("2.0") => Some(EventFormat::Evt2),
        Some("2.1") => Some(EventFormat::Evt21),
        Some("3.0") => Some(EventFormat::Evt3),
        _ => None,
    }
}

struct Evt2EventStream<R> {
    reader: Evt2Reader<R>,
}

impl<R: BufRead> EventStream for Evt2EventStream<R> {
    fn next_record(&mut self) -> Result<Option<EventRecord>> {
        loop {
            let Some(decoded) = self.reader.next_decoded()? else {
                return Ok(None);
            };

            match decoded {
                DecodedEvt2::Event(event) => return Ok(Some(EventRecord::Event(event))),
                DecodedEvt2::ExtTrigger(trigger) => {
                    return Ok(Some(EventRecord::ExtTrigger(trigger)));
                }
                DecodedEvt2::Other { .. } => return Ok(Some(EventRecord::Other)),
            }
        }
    }
}
