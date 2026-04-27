use crate::Event;
use crate::Result;
use crate::algorithms::EventAlgorithm;
use crate::filters::EventFilter;
use crate::parser::{EventRecord, EventStream};

#[derive(Debug)]
pub struct ProcessedEvent<T> {
    pub event: Event,
    pub accepted: bool,
    pub outputs: Vec<T>,
}

pub struct EventPipeline<F, A> {
    stream: Box<dyn EventStream>,
    filters: F,
    algorithm: A,
}

impl<F, A> EventPipeline<F, A>
where
    F: EventFilter,
    A: EventAlgorithm,
{
    pub fn new(stream: Box<dyn EventStream>, filters: F, algorithm: A) -> Self {
        Self {
            stream,
            filters,
            algorithm,
        }
    }

    pub fn next_event(&mut self) -> Result<Option<ProcessedEvent<A::Output>>> {
        loop {
            let Some(record) = self.stream.next_record()? else {
                return Ok(None);
            };

            let EventRecord::Event(event) = record else {
                continue;
            };

            let accepted = self.filters.accepts(event);
            let outputs = if accepted {
                self.algorithm.process_event(event)
            } else {
                Vec::new()
            };

            return Ok(Some(ProcessedEvent {
                event,
                accepted,
                outputs,
            }));
        }
    }

    pub fn finish(&mut self) -> Vec<A::Output> {
        self.algorithm.finish()
    }

    pub fn filters_mut(&mut self) -> &mut F {
        &mut self.filters
    }

    pub fn algorithm_mut(&mut self) -> &mut A {
        &mut self.algorithm
    }
}
