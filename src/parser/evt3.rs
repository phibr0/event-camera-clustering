use super::evt2::ExtTrigger;
use crate::parser::{Endian, EventRecord, EventStream};
use crate::{Event, Result};
use std::collections::VecDeque;
use std::io::{self, BufRead};

/// Decoder state between EVT3 words.
#[derive(Default)]
struct Evt3State {
    time_high: Option<u16>,
    time_low: Option<u16>,
    y: Option<u16>,
    vector_x: u16,
    vector_polarity: bool,
}

impl Evt3State {
    fn timestamp_us(&self) -> Option<u64> {
        Some(((self.time_high? as u64) << 12) | self.time_low? as u64)
    }

    fn decode_word(&mut self, word: u16, emit: &mut impl FnMut(EventRecord)) {
        let event_type = (word >> 12) as u8;
        match event_type {
            0x0 => self.y = Some(word & 0x07ff),
            0x2 => {
                if let (Some(timestamp_us), Some(y)) = (self.timestamp_us(), self.y) {
                    emit(EventRecord::Event(Event {
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
            0x4 => self.decode_vector(word & 0x0fff, 12, emit),
            0x5 => self.decode_vector(word & 0x00ff, 8, emit),
            0x6 => self.time_low = Some(word & 0x0fff),
            0x8 => self.time_high = Some(word & 0x0fff),
            0xa => {
                if let Some(timestamp_us) = self.timestamp_us() {
                    emit(EventRecord::ExtTrigger(ExtTrigger {
                        timestamp_us,
                        id: ((word >> 8) & 0x0f) as u8,
                        value: (word & 0x1) != 0,
                    }));
                }
            }
            _ => emit(EventRecord::Other),
        }
    }

    fn decode_vector(&mut self, valid: u16, width: u16, emit: &mut impl FnMut(EventRecord)) {
        if let (Some(timestamp_us), Some(y)) = (self.timestamp_us(), self.y) {
            for offset in 0..width {
                if (valid & (1 << offset)) != 0 {
                    emit(EventRecord::Event(Event {
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

pub(crate) struct Evt3EventStream<R> {
    reader: R,
    endian: Endian,
    state: Evt3State,
    pending: VecDeque<EventRecord>,
}

impl<R: BufRead> Evt3EventStream<R> {
    pub(crate) fn new(reader: R, endian: Endian) -> Self {
        Self {
            reader,
            endian,
            state: Evt3State::default(),
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
            let pending = &mut self.pending;
            self.state
                .decode_word(word, &mut |record| pending.push_back(record));
        }
    }
}

/// Decodes little-endian EVT3 bytes as they arrive from a camera.
///
/// Unlike the file reader, timestamps continue across the 24-bit wrap of the
/// format every 16.8 s and never decrease, so they can run for hours.
#[derive(Default)]
pub struct Evt3Decoder {
    state: Evt3State,
    /// First byte of a word that was split between two buffers.
    carry: Option<u8>,
    wraps_us: u64,
    last_us: u64,
}

impl Evt3Decoder {
    pub fn new() -> Self {
        Self::default()
    }

    /// Appends the events in `bytes` to `events`.
    pub fn decode(&mut self, bytes: &[u8], events: &mut Vec<Event>) {
        let mut bytes = bytes;
        if let (Some(low), Some((&high, rest))) = (self.carry.take(), bytes.split_first()) {
            self.word(u16::from_le_bytes([low, high]), events);
            bytes = rest;
        }
        let mut words = bytes.chunks_exact(2);
        for word in &mut words {
            self.word(u16::from_le_bytes([word[0], word[1]]), events);
        }
        if let [low] = *words.remainder() {
            self.carry = Some(low);
        }
    }

    fn word(&mut self, word: u16, events: &mut Vec<Event>) {
        if word >> 12 == 0x8 {
            let high = word & 0x0fff;
            if let Some(previous) = self.state.time_high.filter(|&previous| previous != high) {
                // The high part counts 4096 us steps and wraps after 4096 of them.
                if previous > high && previous - high > 2048 {
                    self.wraps_us += 1 << 24;
                }
                // The low part restarts with every step of the high part.
                self.state.time_low = Some(0);
            }
        }
        let (wraps_us, last_us) = (self.wraps_us, &mut self.last_us);
        self.state.decode_word(word, &mut |record| {
            if let EventRecord::Event(mut event) = record {
                event.timestamp_us = (event.timestamp_us + wraps_us).max(*last_us);
                *last_us = event.timestamp_us;
                events.push(event);
            }
        });
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::io::Cursor;

    #[test]
    fn live_decoder_continues_across_split_words_and_the_timestamp_wrap() {
        let event = |x: u16| (0x2_u16 << 12) | x;
        let words = [
            (0x8_u16 << 12) | 4095,
            (0x6_u16 << 12) | 4000,
            7,
            event(1),
            // The high part wraps to zero; the low part has not been sent yet.
            0x8_u16 << 12,
            event(2),
            (0x6_u16 << 12) | 5,
            event(3),
        ];
        let bytes: Vec<u8> = words.iter().flat_map(|word| word.to_le_bytes()).collect();
        let mut decoder = Evt3Decoder::new();
        let mut events = Vec::new();
        // Two buffers that split a word in the middle.
        decoder.decode(&bytes[..9], &mut events);
        decoder.decode(&bytes[9..], &mut events);
        let times: Vec<_> = events.iter().map(|e| (e.x, e.timestamp_us)).collect();
        let before = (4095 << 12) | 4000;
        assert_eq!(times, [(1, before), (2, 1 << 24), (3, (1 << 24) + 5)]);
        assert!(events.iter().all(|e| e.y == 7 && !e.polarity));
    }

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
