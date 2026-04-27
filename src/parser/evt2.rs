use crate::Result;
use crate::event::Event;
use crate::parser::Endian;
use std::fs::File;
use std::io::{self, BufRead, BufReader};
use std::path::Path;

#[derive(Debug, Clone, PartialEq, Eq)]
pub struct RawHeader {
    pub lines: Vec<String>,
    pub evt_version: Option<String>,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct ExtTrigger {
    pub timestamp_us: u64,
    pub id: u8,
    pub value: bool,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum DecodedEvt2 {
    Event(Event),
    ExtTrigger(ExtTrigger),
    Other { event_type: u8, word: u32 },
}

pub struct Evt2Reader<R> {
    reader: R,
    endian: Endian,
    time_high: Option<u32>,
}

impl Evt2Reader<BufReader<File>> {
    pub fn from_path(path: impl AsRef<Path>, endian: Endian) -> Result<(RawHeader, Self)> {
        let file = File::open(path)?;
        Self::new(BufReader::new(file), endian)
    }
}

impl<R: BufRead> Evt2Reader<R> {
    pub fn new(mut reader: R, endian: Endian) -> Result<(RawHeader, Self)> {
        let header = read_raw_header(&mut reader)?;
        Ok((header, Self::from_reader(reader, endian)))
    }

    pub(crate) fn from_reader(reader: R, endian: Endian) -> Self {
        Self {
            reader,
            endian,
            time_high: None,
        }
    }

    pub fn next_decoded(&mut self) -> Result<Option<DecodedEvt2>> {
        loop {
            let mut bytes = [0_u8; 4];
            match self.reader.read_exact(&mut bytes) {
                Ok(()) => {}
                Err(error) if error.kind() == io::ErrorKind::UnexpectedEof => return Ok(None),
                Err(error) => return Err(error.into()),
            }

            let word = match self.endian {
                Endian::Little | Endian::Little32 => u32::from_le_bytes(bytes),
                Endian::Big => u32::from_be_bytes(bytes),
            };

            if let Some(decoded) = decode_word(word, &mut self.time_high) {
                return Ok(Some(decoded));
            }
        }
    }
}

impl<R: BufRead> Iterator for Evt2Reader<R> {
    type Item = Result<DecodedEvt2>;

    fn next(&mut self) -> Option<Self::Item> {
        self.next_decoded().transpose()
    }
}

pub(crate) fn read_raw_header<R: BufRead>(reader: &mut R) -> Result<RawHeader> {
    let mut lines = Vec::new();
    let mut evt_version = None;

    loop {
        let buffer = reader.fill_buf()?;
        if buffer.first() != Some(&b'%') {
            break;
        }

        let mut line = Vec::new();
        reader.read_until(b'\n', &mut line)?;
        let line = String::from_utf8_lossy(&line).trim().to_owned();

        if let Some(value) = line.strip_prefix("% evt ") {
            evt_version = Some(value.to_owned());
        }

        lines.push(line);
    }

    Ok(RawHeader { lines, evt_version })
}

fn decode_word(word: u32, time_high: &mut Option<u32>) -> Option<DecodedEvt2> {
    let event_type = (word >> 28) as u8;

    match event_type {
        0x0 | 0x1 => {
            let high = (*time_high)? as u64;
            let timestamp_low = ((word >> 22) & 0x3f) as u64;
            let x = ((word >> 11) & 0x7ff) as u16;
            let y = (word & 0x7ff) as u16;

            Some(DecodedEvt2::Event(Event {
                timestamp_us: (high << 6) | timestamp_low,
                x,
                y,
                polarity: event_type == 0x1,
            }))
        }
        0x8 => {
            *time_high = Some(word & 0x0fff_ffff);
            None
        }
        0xa => {
            let high = (*time_high)? as u64;
            let timestamp_low = ((word >> 22) & 0x3f) as u64;
            let id = ((word >> 8) & 0x1f) as u8;
            let value = (word & 0x1) != 0;

            Some(DecodedEvt2::ExtTrigger(ExtTrigger {
                timestamp_us: (high << 6) | timestamp_low,
                id,
                value,
            }))
        }
        _ => Some(DecodedEvt2::Other { event_type, word }),
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::io::Cursor;

    #[test]
    fn skips_raw_header_and_reads_evt_version() {
        let bytes = b"% Date today\n% evt 2.0\n\x00\x00\x00\x80";
        let (header, mut reader) = Evt2Reader::new(Cursor::new(bytes), Endian::Little).unwrap();

        assert_eq!(header.evt_version.as_deref(), Some("2.0"));
        assert_eq!(header.lines.len(), 2);
        assert!(reader.next_decoded().unwrap().is_none());
    }

    #[test]
    fn decodes_cd_event_timestamp_coordinates_and_polarity() {
        let time_high = 12_u32;
        let event_word = (0x1_u32 << 28) | (3 << 22) | (45 << 11) | 67;
        let mut bytes = Vec::new();
        bytes.extend_from_slice(&(0x8 << 28 | time_high).to_le_bytes());
        bytes.extend_from_slice(&event_word.to_le_bytes());

        let (_, mut reader) = Evt2Reader::new(Cursor::new(bytes), Endian::Little).unwrap();
        let decoded = reader.next_decoded().unwrap().unwrap();
        assert_eq!(
            decoded,
            DecodedEvt2::Event(Event {
                timestamp_us: (12 << 6) | 3,
                x: 45,
                y: 67,
                polarity: true,
            })
        );
    }
}
