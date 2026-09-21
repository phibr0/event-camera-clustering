pub mod algorithms;
pub mod ball;
pub mod error;
pub mod event;
pub mod filter;
pub mod motion;
pub mod parabola;
pub mod parser;
pub mod pipeline;

pub use error::{EventError, Result};
pub use event::{BoundingBox, Event};
