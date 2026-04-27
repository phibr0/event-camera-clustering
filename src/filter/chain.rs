use crate::event::Event;
use crate::filter::EventFilter;

pub struct EventFilterChain {
    filters: Vec<Box<dyn EventFilter>>,
}

impl EventFilterChain {
    pub fn new() -> Self {
        Self {
            filters: Vec::new(),
        }
    }

    pub fn from_filters(filters: impl IntoIterator<Item = Box<dyn EventFilter>>) -> Self {
        Self {
            filters: filters.into_iter().collect(),
        }
    }

    pub fn push(&mut self, filter: impl EventFilter + 'static) {
        self.filters.push(Box::new(filter));
    }
}

impl Default for EventFilterChain {
    fn default() -> Self {
        Self::new()
    }
}

impl EventFilter for EventFilterChain {
    fn accepts(&mut self, event: Event) -> bool {
        self.filters.iter_mut().all(|filter| filter.accepts(event))
    }
}
