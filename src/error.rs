use std::io;

#[derive(Debug, thiserror::Error)]
pub enum EventError {
    #[error("I/O error: {0}")]
    Io(#[from] io::Error),

    #[error("invalid configuration for {field}: {message}")]
    InvalidConfig {
        field: &'static str,
        message: &'static str,
    },
}

pub type Result<T> = std::result::Result<T, EventError>;
