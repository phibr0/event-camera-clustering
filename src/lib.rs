pub mod algorithms;
pub mod error;
pub mod event;
pub mod evt2;
pub mod filters;
pub mod parser;
pub mod pipeline;

pub use error::{EventError, Result};
pub use event::{BoundingBox, Event};
