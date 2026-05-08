mod background_activity;
mod chain;
mod configured;
mod polarity;
mod static_event;

pub use background_activity::{BackgroundActivityFilter, BackgroundActivityFilterConfig};
pub use chain::EventFilterChain;
pub use configured::ConfiguredEventFilters;
pub use polarity::{PolarityEventFilter, PolarityFilterConfig, PolarityMode};
pub use static_event::{StaticEventFilter, StaticEventFilterConfig};

use crate::Event;

pub trait EventFilter {
    fn accepts(&mut self, event: Event) -> bool;
}
