use std::path::{Path, PathBuf};

/// Source for loading a model from disk or HuggingFace.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct ModelSource {
    kind: ModelSourceKind,
}

#[derive(Debug, Clone, PartialEq, Eq)]
enum ModelSourceKind {
    File {
        path: PathBuf,
    },
    HuggingFace {
        repo_id: String,
        filename: String,
        revision: Option<String>,
    },
    HuggingFaceDir {
        repo_id: String,
        directory: String,
        revision: Option<String>,
    },
}

impl ModelSource {
    /// Create a source backed by a local model file.
    pub fn from_file(path: impl Into<PathBuf>) -> Self {
        Self {
            kind: ModelSourceKind::File { path: path.into() },
        }
    }

    /// Create a source backed by a HuggingFace repo + filename.
    pub fn from_hf(repo_id: impl Into<String>, filename: impl Into<String>) -> Self {
        Self {
            kind: ModelSourceKind::HuggingFace {
                repo_id: repo_id.into(),
                filename: filename.into(),
                revision: None,
            },
        }
    }

    /// Create a source backed by a HuggingFace repo + directory prefix.
    pub fn from_hf_dir(repo_id: impl Into<String>, directory: impl Into<String>) -> Self {
        Self {
            kind: ModelSourceKind::HuggingFaceDir {
                repo_id: repo_id.into(),
                directory: directory.into(),
                revision: None,
            },
        }
    }

    /// Set the HuggingFace revision (branch, tag, or commit SHA).
    pub fn with_revision(mut self, revision: impl Into<String>) -> Self {
        match &mut self.kind {
            ModelSourceKind::HuggingFace { revision: slot, .. }
            | ModelSourceKind::HuggingFaceDir { revision: slot, .. } => {
                *slot = Some(revision.into());
            }
            _ => {}
        }
        self
    }

    /// Resolve the model path, downloading if necessary.
    pub fn resolve(&self) -> Result<PathBuf, ModelSourceError> {
        match &self.kind {
            ModelSourceKind::File { path } => {
                if path.is_file() {
                    Ok(path.clone())
                } else {
                    Err(ModelSourceError::MissingLocalFile(path.clone()))
                }
            }
            ModelSourceKind::HuggingFace {
                repo_id,
                filename,
                revision,
            } => resolve_hf(repo_id, filename, revision.as_deref()),
            ModelSourceKind::HuggingFaceDir {
                repo_id,
                directory,
                revision,
            } => resolve_hf_dir(repo_id, directory, revision.as_deref()),
        }
    }

    /// Return the local path when the source is a file.
    pub fn local_path(&self) -> Option<&Path> {
        match &self.kind {
            ModelSourceKind::File { path } => Some(path.as_path()),
            _ => None,
        }
    }

    /// Return the HuggingFace repo ID if applicable.
    pub fn repo_id(&self) -> Option<&str> {
        match &self.kind {
            ModelSourceKind::HuggingFace { repo_id, .. }
            | ModelSourceKind::HuggingFaceDir { repo_id, .. } => Some(repo_id.as_str()),
            _ => None,
        }
    }

    /// Return the HuggingFace filename if applicable.
    pub fn filename(&self) -> Option<&str> {
        match &self.kind {
            ModelSourceKind::HuggingFace { filename, .. } => Some(filename.as_str()),
            _ => None,
        }
    }

    /// Return the HuggingFace directory prefix if applicable.
    pub fn directory(&self) -> Option<&str> {
        match &self.kind {
            ModelSourceKind::HuggingFaceDir { directory, .. } => Some(directory.as_str()),
            _ => None,
        }
    }
}

#[derive(Debug, thiserror::Error)]
pub enum ModelSourceError {
    #[error("Model file not found: {0}")]
    MissingLocalFile(PathBuf),
    #[error("HuggingFace support is not enabled; enable the `model-hf` feature")]
    HuggingFaceDisabled,
    #[error("HuggingFace download failed: {0}")]
    HuggingFaceDownload(String),
    #[error("HuggingFace repo id is required")]
    MissingRepoId,
    #[error("HuggingFace filename is required")]
    MissingFilename,
    #[error("HuggingFace directory is required")]
    MissingDirectory,
}

#[cfg(feature = "model-hf")]
fn resolve_hf(
    repo_id: &str,
    filename: &str,
    revision: Option<&str>,
) -> Result<PathBuf, ModelSourceError> {
    use hf_hub::api::sync::ApiBuilder;
    use hf_hub::{Cache, Repo, RepoType};

    if repo_id.is_empty() {
        return Err(ModelSourceError::MissingRepoId);
    }
    if filename.is_empty() {
        return Err(ModelSourceError::MissingFilename);
    }

    let cache = Cache::from_env();
    let mut api_builder = ApiBuilder::from_cache(cache);
    if let Ok(endpoint) = std::env::var("HF_ENDPOINT") {
        api_builder = api_builder.with_endpoint(endpoint);
    }
    if let Some(token) = hf_token() {
        api_builder = api_builder.with_token(Some(token));
    }
    let api = api_builder
        .build()
        .map_err(|err| ModelSourceError::HuggingFaceDownload(err.to_string()))?;
    let revision = revision.unwrap_or("main");
    let repo = Repo::with_revision(repo_id.to_string(), RepoType::Model, revision.to_string());
    let api_repo = api.repo(repo);
    let path = api_repo
        .get(filename)
        .map_err(|err| ModelSourceError::HuggingFaceDownload(err.to_string()))?;
    Ok(path)
}

#[cfg(feature = "model-hf")]
fn resolve_hf_dir(
    repo_id: &str,
    directory: &str,
    revision: Option<&str>,
) -> Result<PathBuf, ModelSourceError> {
    use hf_hub::api::sync::ApiBuilder;
    use hf_hub::{Cache, Repo, RepoType};

    if repo_id.is_empty() {
        return Err(ModelSourceError::MissingRepoId);
    }
    if directory.is_empty() {
        return Err(ModelSourceError::MissingDirectory);
    }

    let cache = Cache::from_env();
    let mut api_builder = ApiBuilder::from_cache(cache);
    if let Ok(endpoint) = std::env::var("HF_ENDPOINT") {
        api_builder = api_builder.with_endpoint(endpoint);
    }
    if let Some(token) = hf_token() {
        api_builder = api_builder.with_token(Some(token));
    }
    let api = api_builder
        .build()
        .map_err(|err| ModelSourceError::HuggingFaceDownload(err.to_string()))?;
    let revision = revision.unwrap_or("main");
    let repo = Repo::with_revision(repo_id.to_string(), RepoType::Model, revision.to_string());
    let api_repo = api.repo(repo);
    let info = api_repo
        .info()
        .map_err(|err| ModelSourceError::HuggingFaceDownload(err.to_string()))?;

    let prefix = if directory.ends_with('/') {
        directory.to_string()
    } else {
        format!("{directory}/")
    };

    let mut local_dir: Option<PathBuf> = None;
    let mut found = false;

    for sibling in info.siblings {
        let filename = sibling.rfilename;
        if !filename.starts_with(&prefix) {
            continue;
        }
        found = true;
        let path = api_repo
            .get(&filename)
            .map_err(|err| ModelSourceError::HuggingFaceDownload(err.to_string()))?;

        if local_dir.is_none() {
            let local = derive_directory(&path, &prefix, &filename);
            local_dir = Some(local);
        }
    }

    if !found {
        return Err(ModelSourceError::MissingDirectory);
    }

    local_dir.ok_or(ModelSourceError::MissingDirectory)
}

#[cfg(not(feature = "model-hf"))]
fn resolve_hf_dir(
    _repo_id: &str,
    _directory: &str,
    _revision: Option<&str>,
) -> Result<PathBuf, ModelSourceError> {
    Err(ModelSourceError::HuggingFaceDisabled)
}

#[cfg(feature = "model-hf")]
fn derive_directory(path: &Path, directory: &str, rfilename: &str) -> PathBuf {
    let prefix_path = Path::new(directory);
    let prefix_count = prefix_path.components().count();
    let file_components = Path::new(rfilename).components().count();
    let pops = file_components.saturating_sub(prefix_count);

    let mut local = path.to_path_buf();
    for _ in 0..pops {
        local.pop();
    }
    local
}

#[cfg(not(feature = "model-hf"))]
fn resolve_hf(
    _repo_id: &str,
    _filename: &str,
    _revision: Option<&str>,
) -> Result<PathBuf, ModelSourceError> {
    Err(ModelSourceError::HuggingFaceDisabled)
}

#[cfg(feature = "model-hf")]
fn hf_token() -> Option<String> {
    std::env::var("HUGGINGFACE_HUB_TOKEN")
        .ok()
        .or_else(|| std::env::var("HF_TOKEN").ok())
        .or_else(|| std::env::var("HUGGINGFACE_TOKEN").ok())
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::io::Write;

    #[test]
    fn from_file_tracks_path() {
        let source = ModelSource::from_file("model.onnx");
        assert_eq!(source.local_path(), Some(Path::new("model.onnx")));
        assert!(source.repo_id().is_none());
    }

    #[test]
    fn resolve_missing_file_returns_error() {
        let source = ModelSource::from_file("missing.onnx");
        let err = source.resolve().unwrap_err();
        match err {
            ModelSourceError::MissingLocalFile(path) => {
                assert_eq!(path, PathBuf::from("missing.onnx"));
            }
            other => panic!("unexpected error: {other:?}"),
        }
    }

    #[test]
    fn resolve_existing_file() {
        let mut file = tempfile::NamedTempFile::new().unwrap();
        writeln!(file, "test").unwrap();
        let path = file.path().to_path_buf();

        let source = ModelSource::from_file(&path);
        let resolved = source.resolve().unwrap();
        assert_eq!(resolved, path);
    }

    #[test]
    fn from_hf_tracks_repo_and_filename() {
        let source = ModelSource::from_hf("org/model", "model.onnx");
        assert_eq!(source.repo_id(), Some("org/model"));
        assert_eq!(source.filename(), Some("model.onnx"));
    }

    #[test]
    fn from_hf_dir_tracks_repo_and_directory() {
        let source = ModelSource::from_hf_dir("org/model", "weights");
        assert_eq!(source.repo_id(), Some("org/model"));
        assert_eq!(source.directory(), Some("weights"));
        assert!(source.filename().is_none());
    }

    #[test]
    #[cfg(not(feature = "model-hf"))]
    fn resolve_hf_requires_feature() {
        let source = ModelSource::from_hf("org/model", "model.onnx");
        let err = source.resolve().unwrap_err();
        match err {
            ModelSourceError::HuggingFaceDisabled => {}
            other => panic!("unexpected error: {other:?}"),
        }
    }

    #[test]
    #[cfg(not(feature = "model-hf"))]
    fn resolve_hf_dir_requires_feature() {
        let source = ModelSource::from_hf_dir("org/model", "weights");
        let err = source.resolve().unwrap_err();
        match err {
            ModelSourceError::HuggingFaceDisabled => {}
            other => panic!("unexpected error: {other:?}"),
        }
    }
}
