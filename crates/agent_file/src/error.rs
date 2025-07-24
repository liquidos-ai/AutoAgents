//! Defines the custom error types for the .af file parsing library.

use thiserror::Error;

/// Represents the possible errors that can occur while parsing an .af file.
#[derive(Error, Debug)]
pub enum AgentFileError {
    /// An error occurred during file I/O (e.g., reading the file).
    #[error("File I/O error: {0}")]
    Io(#[from] std::io::Error),

    /// An error occurred during JSON deserialization.
    #[error("JSON deserialization error: {0}")]
    Json(#[from] serde_json::Error),
}
