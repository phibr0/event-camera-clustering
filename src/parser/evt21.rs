use super::evt2::ExtTrigger;
use crate::parser::{Endian, EventRecord, EventStream};
use crate::{Event, Result};
use std::collections::VecDeque;
use std::io::{self, BufRead};

pub(crate) struct Evt21EventStream<R> {
    reader: R,
    endian: Endian,
    time_high: Option<u32>,
    pending: VecDeque<EventRecord>,
}

impl<R: BufRead> Evt21EventStream<R> {
    pub(crate) fn new(reader: R, endian: Endian) -> Self {
        Self {
            reader,
            endian,
            time_high: None,
            pending: VecDeque::new(),
        }
    }

    fn read_word(&mut self) -> Result<Option<u64>> {
        let mut bytes = [0_u8; 8];
        match self.reader.read_exact(&mut bytes) {
            Ok(()) => {}
            Err(error) if error.kind() == io::ErrorKind::UnexpectedEof => return Ok(None),
            Err(error) => return Err(error.into()),
        }

        Ok(Some(match self.endian {
            Endian::Little => u64::from_le_bytes(bytes),
            Endian::Little32 => {
                let high = u32::from_le_bytes(bytes[0..4].try_into().unwrap()) as u64;
                let low = u32::from_le_bytes(bytes[4..8].try_into().unwrap()) as u64;
                (high << 32) | low
            }
            Endian::Big => u64::from_be_bytes(bytes),
        }))
    }

    fn decode_word(&mut self, word: u64) {
        let event_type = (word >> 60) as u8;
        match event_type {
            0x0 | 0x1 => {
                let Some(time_high) = self.time_high else {
                    return;
                };
                let timestamp_low = (word >> 54) & 0x3f;
                let base_x = ((word >> 43) & 0x7ff) as u16;
                let y = ((word >> 32) & 0x7ff) as u16;
                let valid = word as u32;
                let timestamp_us = ((time_high as u64) << 6) | timestamp_low;

                for offset in 0..32 {
                    if (valid & (1 << offset)) != 0 {
                        self.pending.push_back(EventRecord::Event(Event {
                            timestamp_us,
                            x: base_x + offset as u16,
                            y,
                            polarity: event_type == 0x1,
                        }));
                    }
                }
            }
            0x8 => {
                self.time_high = Some(((word >> 32) & 0x0fff_ffff) as u32);
            }
            0xa => {
                let Some(time_high) = self.time_high else {
                    return;
                };
                let timestamp_low = (word >> 54) & 0x3f;
                self.pending.push_back(EventRecord::ExtTrigger(ExtTrigger {
                    timestamp_us: ((time_high as u64) << 6) | timestamp_low,
                    id: ((word >> 40) & 0x1f) as u8,
                    value: ((word >> 32) & 0x1) != 0,
                }));
            }
            _ => self.pending.push_back(EventRecord::Other),
        }
    }
}

impl<R: BufRead> EventStream for Evt21EventStream<R> {
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
    fn decodes_evt21_vector_events() {
        let time_high_word = (0x8_u64 << 60) | (7 << 32);
        let event_word = (0x1_u64 << 60) | (3 << 54) | (32 << 43) | (9 << 32) | 0b101;
        let mut bytes = Vec::new();
        bytes.extend_from_slice(&time_high_word.to_le_bytes());
        bytes.extend_from_slice(&event_word.to_le_bytes());
        let mut stream = Evt21EventStream::new(Cursor::new(bytes), Endian::Little);

        assert_eq!(
            stream.next_record().unwrap(),
            Some(EventRecord::Event(Event {
                timestamp_us: (7 << 6) | 3,
                x: 32,
                y: 9,
                polarity: true,
            }))
        );
        assert_eq!(
            stream.next_record().unwrap(),
            Some(EventRecord::Event(Event {
                timestamp_us: (7 << 6) | 3,
                x: 34,
                y: 9,
                polarity: true,
            }))
        );
    }

    #[test]
    fn decodes_evt21_little32_words() {
        let time_high_word = (0x8_u64 << 60) | (7 << 32);
        let event_word = (0x1_u64 << 60) | (3 << 54) | (32 << 43) | (9 << 32) | 0b101;
        let mut bytes = Vec::new();
        bytes.extend_from_slice(&((time_high_word >> 32) as u32).to_le_bytes());
        bytes.extend_from_slice(&(time_high_word as u32).to_le_bytes());
        bytes.extend_from_slice(&((event_word >> 32) as u32).to_le_bytes());
        bytes.extend_from_slice(&(event_word as u32).to_le_bytes());
        let mut stream = Evt21EventStream::new(Cursor::new(bytes), Endian::Little32);

        assert_eq!(
            stream.next_record().unwrap(),
            Some(EventRecord::Event(Event {
                timestamp_us: (7 << 6) | 3,
                x: 32,
                y: 9,
                polarity: true,
            }))
        );
    }
}
