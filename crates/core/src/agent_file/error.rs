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

    /// An error occurred during temporary directory creation.
    #[error("Temporary directory creation error: {0}")]
    TempDir(#[from] tempfile::PersistError),

    /// An error occurred during file writing.
    #[error("File writing error: {0}")]
    Write(String),

    /// Invalid agent file format
    #[error("Invalid agent file format: {0}")]
    InvalidFormat(String),

    /// Missing required field
    #[error("Missing required field: {0}")]
    MissingField(String),

    /// Invalid tool configuration
    #[error("Invalid tool configuration: {0}")]
    InvalidTool(String),

    /// Invalid memory configuration
    #[error("Invalid memory configuration: {0}")]
    InvalidMemory(String),
} 