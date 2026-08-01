use std::path::PathBuf;

#[derive(Debug, thiserror::Error)]
pub enum SkillError {
    #[error("skill document at '{path}' is invalid: {message}")]
    InvalidDocument { path: PathBuf, message: String },

    #[error("skill source '{source_id}' is unavailable: {message}")]
    SourceUnavailable { source_id: String, message: String },

    #[error("skill source worker failed: {0}")]
    SourceWorker(String),

    #[error("watched skill refresh requires the 'skill-watch' Cargo feature")]
    WatchUnavailable,

    #[error("skill '{0}' was not found")]
    UnknownSkill(String),

    #[error("skill '{0}' is not active in this conversation")]
    InactiveSkill(String),

    #[error("skill operation was denied by policy: {0}")]
    PolicyDenied(String),

    #[error("skill tool selector '{0}' is invalid")]
    InvalidToolSelector(String),

    #[error("resource path '{0}' is invalid")]
    InvalidResourcePath(String),

    #[error("resource '{path}' exceeds the {limit} byte limit")]
    ResourceTooLarge { path: PathBuf, limit: usize },

    #[error("resource '{0}' is not valid UTF-8")]
    ResourceNotUtf8(PathBuf),

    #[error("resource '{0}' changed after skill discovery; refresh and reactivate the skill")]
    ResourceChanged(PathBuf),

    #[error("skill tool name '{0}' conflicts with an agent tool")]
    ToolNameConflict(String),
}

impl SkillError {
    pub(crate) fn invalid_document(path: impl Into<PathBuf>, message: impl Into<String>) -> Self {
        Self::InvalidDocument {
            path: path.into(),
            message: message.into(),
        }
    }

    pub(crate) fn source_unavailable(
        source: impl Into<String>,
        message: impl Into<String>,
    ) -> Self {
        Self::SourceUnavailable {
            source_id: source.into(),
            message: message.into(),
        }
    }
}
