use crate::filter::EventFilter;
use crate::{Event, EventError, Result};
use std::collections::HashMap;

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct BackgroundActivityFilterConfig {
    pub enabled: bool,
    pub radius_px: u16,
    pub time_window_us: u64,
    pub cleanup_after_us: u64,
}

impl Default for BackgroundActivityFilterConfig {
    fn default() -> Self {
        Self {
            enabled: false,
            radius_px: 2,
            time_window_us: 5_000,
            cleanup_after_us: 100_000,
        }
    }
}

impl BackgroundActivityFilterConfig {
    pub fn validate(&self) -> Result<()> {
        if self.time_window_us == 0 {
            return Err(EventError::InvalidConfig {
                field: "background_time_window_us",
                message: "must be positive",
            });
        }
        if self.cleanup_after_us < self.time_window_us {
            return Err(EventError::InvalidConfig {
                field: "background_cleanup_after_us",
                message: "must be at least background_time_window_us",
            });
        }
        Ok(())
    }
}

pub struct BackgroundActivityFilter {
    config: BackgroundActivityFilterConfig,
    last_seen_by_pixel: HashMap<(u16, u16), u64>,
    last_cleanup_us: u64,
}

impl BackgroundActivityFilter {
    pub fn new(config: BackgroundActivityFilterConfig) -> Result<Self> {
        config.validate()?;
        Ok(Self {
            config,
            last_seen_by_pixel: HashMap::new(),
            last_cleanup_us: 0,
        })
    }

    pub fn set_config(&mut self, config: BackgroundActivityFilterConfig) -> Result<()> {
        config.validate()?;
        if self.config.radius_px != config.radius_px
            || self.config.time_window_us != config.time_window_us
        {
            self.last_seen_by_pixel.clear();
        }
        self.config = config;
        Ok(())
    }

    pub fn reset(&mut self) {
        self.last_seen_by_pixel.clear();
        self.last_cleanup_us = 0;
    }
}

impl EventFilter for BackgroundActivityFilter {
    fn accepts(&mut self, event: Event) -> bool {
        if !self.config.enabled {
            return true;
        }

        if event.timestamp_us.saturating_sub(self.last_cleanup_us) > self.config.cleanup_after_us {
            cleanup_old_pixels(
                &mut self.last_seen_by_pixel,
                event.timestamp_us,
                self.config.cleanup_after_us,
            );
            self.last_cleanup_us = event.timestamp_us;
        }

        let has_neighbor = has_recent_neighbor(
            &self.last_seen_by_pixel,
            event,
            self.config.radius_px,
            self.config.time_window_us,
        );
        self.last_seen_by_pixel
            .insert((event.x, event.y), event.timestamp_us);

        has_neighbor
    }
}

fn has_recent_neighbor(
    last_seen_by_pixel: &HashMap<(u16, u16), u64>,
    event: Event,
    radius_px: u16,
    time_window_us: u64,
) -> bool {
    let radius = i32::from(radius_px);
    let radius_sq = radius * radius;
    let x = i32::from(event.x);
    let y = i32::from(event.y);

    for dx in -radius..=radius {
        for dy in -radius..=radius {
            if dx == 0 && dy == 0 {
                continue;
            }
            if dx * dx + dy * dy > radius_sq {
                continue;
            }

            let neighbor_x = x + dx;
            let neighbor_y = y + dy;
            if neighbor_x < 0 || neighbor_y < 0 {
                continue;
            }

            let Some(&last_seen_us) =
                last_seen_by_pixel.get(&(neighbor_x as u16, neighbor_y as u16))
            else {
                continue;
            };
            if event.timestamp_us.saturating_sub(last_seen_us) <= time_window_us {
                return true;
            }
        }
    }

    false
}

fn cleanup_old_pixels(
    last_seen_by_pixel: &mut HashMap<(u16, u16), u64>,
    now_us: u64,
    cleanup_after_us: u64,
) {
    last_seen_by_pixel
        .retain(|_, last_seen_us| now_us.saturating_sub(*last_seen_us) <= cleanup_after_us);
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn rejects_isolated_event_and_accepts_recent_neighbor() {
        let mut filter = BackgroundActivityFilter::new(BackgroundActivityFilterConfig {
            enabled: true,
            radius_px: 2,
            time_window_us: 1_000,
            cleanup_after_us: 10_000,
        })
        .unwrap();

        assert!(!filter.accepts(event_at(0, 10, 10)));
        assert!(filter.accepts(event_at(500, 11, 10)));
    }

    #[test]
    fn rejects_neighbor_outside_time_window() {
        let mut filter = BackgroundActivityFilter::new(BackgroundActivityFilterConfig {
            enabled: true,
            radius_px: 2,
            time_window_us: 1_000,
            cleanup_after_us: 10_000,
        })
        .unwrap();

        assert!(!filter.accepts(event_at(0, 10, 10)));
        assert!(!filter.accepts(event_at(2_000, 11, 10)));
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
