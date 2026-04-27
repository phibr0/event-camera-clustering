use crate::Result;
use crate::event::Event;
use crate::filter::{
    EventFilter, PolarityEventFilter, PolarityFilterConfig, StaticEventFilter,
    StaticEventFilterConfig,
};

pub struct ConfiguredEventFilters {
    polarity_filter: PolarityEventFilter,
    static_filter: StaticEventFilter,
}

impl ConfiguredEventFilters {
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

impl EventFilter for ConfiguredEventFilters {
    fn accepts(&mut self, event: Event) -> bool {
        self.polarity_filter.accepts(event) && self.static_filter.accepts(event)
    }
}
