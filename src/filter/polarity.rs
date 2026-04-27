use crate::Event;
use crate::filter::EventFilter;

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum PolarityMode {
    All,
    Positive,
    Negative,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct PolarityFilterConfig {
    pub mode: PolarityMode,
    pub invert_polarity: bool,
}

impl Default for PolarityFilterConfig {
    fn default() -> Self {
        Self {
            mode: PolarityMode::All,
            invert_polarity: true,
        }
    }
}

pub struct PolarityEventFilter {
    config: PolarityFilterConfig,
}

impl PolarityEventFilter {
    pub fn new(config: PolarityFilterConfig) -> Self {
        Self { config }
    }

    pub fn set_config(&mut self, config: PolarityFilterConfig) {
        self.config = config;
    }
}

impl EventFilter for PolarityEventFilter {
    fn accepts(&mut self, event: Event) -> bool {
        let polarity = if self.config.invert_polarity {
            !event.polarity
        } else {
            event.polarity
        };

        match self.config.mode {
            PolarityMode::All => true,
            PolarityMode::Positive => polarity,
            PolarityMode::Negative => !polarity,
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn polarity_filter_can_invert_labels() {
        let mut filter = PolarityEventFilter::new(PolarityFilterConfig {
            mode: PolarityMode::Positive,
            invert_polarity: true,
        });

        assert!(filter.accepts(Event {
            timestamp_us: 0,
            x: 0,
            y: 0,
            polarity: false,
        }));
        assert!(!filter.accepts(Event {
            timestamp_us: 0,
            x: 0,
            y: 0,
            polarity: true,
        }));
    }
}
