use crate::Event;
use crate::{EventError, Result};
use std::collections::HashMap;

pub trait EventFilter {
    fn accepts(&mut self, event: Event) -> bool;
}

pub struct EventFilterChain {
    polarity_filter: PolarityEventFilter,
    static_filter: StaticEventFilter,
}

impl EventFilterChain {
    pub fn new(
        polarity_config: PolarityFilterConfig,
        static_config: StaticEventFilterConfig,
    ) -> Result<Self> {
        Ok(Self {
            polarity_filter: PolarityEventFilter::new(polarity_config),
            static_filter: StaticEventFilter::new(static_config)?,
        })
    }

    pub fn set_config(
        &mut self,
        polarity_config: PolarityFilterConfig,
        static_config: StaticEventFilterConfig,
    ) -> Result<()> {
        self.polarity_filter.set_config(polarity_config);
        self.static_filter.set_config(static_config)?;
        Ok(())
    }
}

impl EventFilter for EventFilterChain {
    fn accepts(&mut self, event: Event) -> bool {
        self.polarity_filter.accepts(event) && self.static_filter.accepts(event)
    }
}

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

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct StaticEventFilterConfig {
    pub enabled: bool,
    pub cell_size: u16,
    pub stable_after_us: u64,
    pub min_events: u32,
    pub inactive_after_us: u64,
}

impl Default for StaticEventFilterConfig {
    fn default() -> Self {
        Self {
            enabled: false,
            cell_size: 4,
            stable_after_us: 250_000,
            min_events: 200,
            inactive_after_us: 1_000_000,
        }
    }
}

impl StaticEventFilterConfig {
    pub fn validate(&self) -> Result<()> {
        if self.cell_size == 0 {
            return Err(EventError::InvalidConfig {
                field: "static_cell_size",
                message: "must be positive",
            });
        }
        Ok(())
    }
}

pub struct StaticEventFilter {
    config: StaticEventFilterConfig,
    cells: HashMap<(u16, u16), CellActivity>,
    last_cleanup_us: u64,
}

#[derive(Debug, Clone, Copy)]
struct CellActivity {
    first_seen_us: u64,
    last_seen_us: u64,
    event_count: u32,
}

impl StaticEventFilter {
    pub fn new(config: StaticEventFilterConfig) -> Result<Self> {
        config.validate()?;
        Ok(Self {
            config,
            cells: HashMap::new(),
            last_cleanup_us: 0,
        })
    }

    pub fn set_config(&mut self, config: StaticEventFilterConfig) -> Result<()> {
        config.validate()?;
        if self.config.cell_size != config.cell_size {
            self.cells.clear();
        }
        self.config = config;
        Ok(())
    }

    pub fn reset(&mut self) {
        self.cells.clear();
        self.last_cleanup_us = 0;
    }
}

impl EventFilter for StaticEventFilter {
    fn accepts(&mut self, event: Event) -> bool {
        if !self.config.enabled {
            return true;
        }

        if event.timestamp_us.saturating_sub(self.last_cleanup_us) > self.config.inactive_after_us {
            cleanup_inactive_cells(
                &mut self.cells,
                event.timestamp_us,
                self.config.inactive_after_us,
            );
            self.last_cleanup_us = event.timestamp_us;
        }

        let key = (
            event.x / self.config.cell_size,
            event.y / self.config.cell_size,
        );
        let activity = self.cells.entry(key).or_insert(CellActivity {
            first_seen_us: event.timestamp_us,
            last_seen_us: event.timestamp_us,
            event_count: 0,
        });

        activity.last_seen_us = event.timestamp_us;
        activity.event_count = activity.event_count.saturating_add(1);

        let active_duration_us = activity.last_seen_us.saturating_sub(activity.first_seen_us);
        !(active_duration_us >= self.config.stable_after_us
            && activity.event_count >= self.config.min_events)
    }
}

fn cleanup_inactive_cells(
    cells: &mut HashMap<(u16, u16), CellActivity>,
    now_us: u64,
    inactive_after_us: u64,
) {
    cells.retain(|_, activity| now_us.saturating_sub(activity.last_seen_us) <= inactive_after_us);
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn suppresses_cells_that_stay_active() {
        let mut filter = StaticEventFilter::new(StaticEventFilterConfig {
            enabled: true,
            cell_size: 4,
            stable_after_us: 1_000,
            min_events: 3,
            inactive_after_us: 10_000,
        })
        .unwrap();

        assert!(filter.accepts(event_at(0, 10, 10)));
        assert!(filter.accepts(event_at(500, 11, 11)));
        assert!(!filter.accepts(event_at(1_000, 10, 10)));
    }

    #[test]
    fn allows_transient_motion_through_cell() {
        let mut filter = StaticEventFilter::new(StaticEventFilterConfig {
            enabled: true,
            cell_size: 4,
            stable_after_us: 1_000,
            min_events: 3,
            inactive_after_us: 10_000,
        })
        .unwrap();

        assert!(filter.accepts(event_at(0, 10, 10)));
        assert!(filter.accepts(event_at(100, 30, 30)));
        assert!(filter.accepts(event_at(200, 50, 50)));
    }

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

    #[test]
    fn rejects_invalid_static_filter_config() {
        let mut config = StaticEventFilterConfig::default();
        config.cell_size = 0;

        assert!(StaticEventFilter::new(config).is_err());
    }

    fn event_at(timestamp_us: u64, x: u16, y: u16) -> Event {
        Event {
            timestamp_us,
            x,
            y,
            polarity: true,
        }
    }
}
