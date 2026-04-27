use super::evt2::ExtTrigger;
use crate::parser::{Endian, EventRecord, EventStream};
use crate::{Event, Result};
use std::collections::VecDeque;
use std::io::{self, BufRead};

pub(crate) struct Evt3EventStream<R> {
    reader: R,
    endian: Endian,
    time_high: Option<u16>,
    time_low: Option<u16>,
    y: Option<u16>,
    vector_x: u16,
    vector_polarity: bool,
    pending: VecDeque<EventRecord>,
}

impl<R: BufRead> Evt3EventStream<R> {
    pub(crate) fn new(reader: R, endian: Endian) -> Self {
        Self {
            reader,
            endian,
            time_high: None,
            time_low: None,
            y: None,
            vector_x: 0,
            vector_polarity: false,
            pending: VecDeque::new(),
        }
    }

    fn read_word(&mut self) -> Result<Option<u16>> {
        let mut bytes = [0_u8; 2];
        match self.reader.read_exact(&mut bytes) {
            Ok(()) => {}
            Err(error) if error.kind() == io::ErrorKind::UnexpectedEof => return Ok(None),
            Err(error) => return Err(error.into()),
        }

        Ok(Some(match self.endian {
            Endian::Little | Endian::Little32 => u16::from_le_bytes(bytes),
            Endian::Big => u16::from_be_bytes(bytes),
        }))
    }

    fn timestamp_us(&self) -> Option<u64> {
        Some(((self.time_high? as u64) << 12) | self.time_low? as u64)
    }

    fn decode_word(&mut self, word: u16) {
        let event_type = (word >> 12) as u8;
        match event_type {
            0x0 => self.y = Some(word & 0x07ff),
            0x2 => {
                if let (Some(timestamp_us), Some(y)) = (self.timestamp_us(), self.y) {
                    self.pending.push_back(EventRecord::Event(Event {
                        timestamp_us,
                        x: word & 0x07ff,
                        y,
                        polarity: (word & 0x0800) != 0,
                    }));
                }
            }
            0x3 => {
                self.vector_x = word & 0x07ff;
                self.vector_polarity = (word & 0x0800) != 0;
            }
            0x4 => self.decode_vector(word & 0x0fff, 12),
            0x5 => self.decode_vector(word & 0x00ff, 8),
            0x6 => self.time_low = Some(word & 0x0fff),
            0x8 => self.time_high = Some(word & 0x0fff),
            0xa => {
                if let Some(timestamp_us) = self.timestamp_us() {
                    self.pending.push_back(EventRecord::ExtTrigger(ExtTrigger {
                        timestamp_us,
                        id: ((word >> 8) & 0x0f) as u8,
                        value: (word & 0x1) != 0,
                    }));
                }
            }
            _ => self.pending.push_back(EventRecord::Other),
        }
    }

    fn decode_vector(&mut self, valid: u16, width: u16) {
        if let (Some(timestamp_us), Some(y)) = (self.timestamp_us(), self.y) {
            for offset in 0..width {
                if (valid & (1 << offset)) != 0 {
                    self.pending.push_back(EventRecord::Event(Event {
                        timestamp_us,
                        x: self.vector_x + offset,
                        y,
                        polarity: self.vector_polarity,
                    }));
                }
            }
        }
        self.vector_x = self.vector_x.saturating_add(width);
    }
}

impl<R: BufRead> EventStream for Evt3EventStream<R> {
    fn next_record(&mut self) -> Result<Option<EventRecord>> {
        loop {
            if let Some(record) = self.pending.pop_front() {
                return Ok(Some(record));
            }

            let Some(word) = self.read_word()? else {
                return Ok(None);
            };
            self.decode_word(word);
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::io::Cursor;

    #[test]
    fn decodes_evt3_single_and_vector_events() {
        let words = [
            (0x8_u16 << 12) | 2,
            (0x6_u16 << 12) | 10,
            4,
            (0x2_u16 << 12) | (1 << 11) | 11,
            (0x3_u16 << 12) | 20,
            (0x4_u16 << 12) | 0b101,
        ];
        let mut bytes = Vec::new();
        for word in words {
            bytes.extend_from_slice(&word.to_le_bytes());
        }
        let mut stream = Evt3EventStream::new(Cursor::new(bytes), Endian::Little);

        assert_eq!(
            stream.next_record().unwrap(),
            Some(EventRecord::Event(Event {
                timestamp_us: (2 << 12) | 10,
                x: 11,
                y: 4,
                polarity: true,
            }))
        );
        assert_eq!(
            stream.next_record().unwrap(),
            Some(EventRecord::Event(Event {
                timestamp_us: (2 << 12) | 10,
                x: 20,
                y: 4,
                polarity: false,
            }))
        );
        assert_eq!(
            stream.next_record().unwrap(),
            Some(EventRecord::Event(Event {
                timestamp_us: (2 << 12) | 10,
                x: 22,
                y: 4,
                polarity: false,
            }))
        );
    }
}
