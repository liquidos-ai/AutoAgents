//! HuggingFace GGUF resolver using hf-hub cache.

use crate::config::LlamaCppConfig;
use crate::error::LlamaCppProviderError;
use hf_hub::api::Siblings;
use hf_hub::api::sync::{Api, ApiBuilder};
use hf_hub::{Cache, Repo, RepoType};
use std::fs;
use std::io;
use std::path::{Path, PathBuf};

pub(crate) fn resolve_hf_model(
    repo_id: &str,
    filename_override: Option<&str>,
    config: &LlamaCppConfig,
) -> Result<String, LlamaCppProviderError> {
    if repo_id.is_empty() {
        return Err(LlamaCppProviderError::Config(
            "HuggingFace repo_id is required".to_string(),
        ));
    }

    let cache = build_cache(config)?;
    let api = build_api(cache.clone())?;
    let revision = config.hf_revision.as_deref().unwrap_or("main");
    let repo = Repo::with_revision(repo_id.to_string(), RepoType::Model, revision.to_string());
    let api_repo = api.repo(repo.clone());

    let filename = match filename_override.or(config.hf_filename.as_deref()) {
        Some(filename) => filename.to_string(),
        None => {
            if let Some(local) = pick_cached_gguf(&cache, &repo)? {
                return Ok(local.to_string_lossy().to_string());
            }
            let model_info = api_repo.info().map_err(|err| {
                LlamaCppProviderError::Other(format!("HuggingFace API error: {}", err))
            })?;
            select_gguf_filename(&model_info.siblings)?
        }
    };

    let model_path = api_repo.get(&filename).map_err(|err| {
        LlamaCppProviderError::Other(format!("HuggingFace download error: {}", err))
    })?;
    Ok(model_path.to_string_lossy().to_string())
}

pub(crate) fn resolve_hf_file(
    repo_id: &str,
    filename: &str,
    config: &LlamaCppConfig,
) -> Result<String, LlamaCppProviderError> {
    if repo_id.is_empty() {
        return Err(LlamaCppProviderError::Config(
            "HuggingFace repo_id is required".to_string(),
        ));
    }
    if filename.is_empty() {
        return Err(LlamaCppProviderError::Config(
            "HuggingFace filename is required".to_string(),
        ));
    }

    let cache = build_cache(config)?;
    let api = build_api(cache.clone())?;
    let revision = config.hf_revision.as_deref().unwrap_or("main");
    let repo = Repo::with_revision(repo_id.to_string(), RepoType::Model, revision.to_string());
    let api_repo = api.repo(repo);
    let file_path = api_repo.get(filename).map_err(|err| {
        LlamaCppProviderError::Other(format!("HuggingFace download error: {}", err))
    })?;
    Ok(file_path.to_string_lossy().to_string())
}

fn build_cache(config: &LlamaCppConfig) -> Result<Cache, LlamaCppProviderError> {
    let cache = match config.model_dir.as_deref() {
        Some(dir) => Cache::new(resolve_cache_dir(dir)?),
        None => Cache::from_env(),
    };
    Ok(cache)
}

static HF_ENDPOINT_ENV: &str = "HF_ENDPOINT";

fn build_api(cache: Cache) -> Result<Api, LlamaCppProviderError> {
    let mut builder = ApiBuilder::from_cache(cache);
    if let Ok(endpoint) = std::env::var(HF_ENDPOINT_ENV) {
        builder = builder.with_endpoint(endpoint);
    }
    if let Some(token) = hf_token() {
        builder = builder.with_token(Some(token));
    }
    builder
        .build()
        .map_err(|err| LlamaCppProviderError::Other(format!("HuggingFace API error: {}", err)))
}

fn resolve_cache_dir(model_dir: &str) -> Result<PathBuf, LlamaCppProviderError> {
    let path = PathBuf::from(model_dir);
    if path.is_absolute() {
        return Ok(path);
    }
    let cwd = std::env::current_dir().map_err(|err| {
        LlamaCppProviderError::Other(format!("Failed to resolve current dir: {}", err))
    })?;
    Ok(cwd.join(path))
}

fn pick_cached_gguf(cache: &Cache, repo: &Repo) -> Result<Option<PathBuf>, LlamaCppProviderError> {
    let repo_dir = cache.path().join(repo.folder_name());
    if !repo_dir.exists() {
        return Ok(None);
    }
    let ref_path = repo_dir.join("refs").join(repo.revision());
    let commit_hash = match fs::read_to_string(&ref_path) {
        Ok(content) => content,
        Err(err) if err.kind() == io::ErrorKind::NotFound => return Ok(None),
        Err(err) => {
            return Err(LlamaCppProviderError::Other(format!(
                "Failed to read HuggingFace cache ref {}: {}",
                ref_path.display(),
                err
            )));
        }
    };
    let commit_hash = commit_hash.trim();
    if commit_hash.is_empty() {
        return Ok(None);
    }
    let snapshot_dir = repo_dir.join("snapshots").join(commit_hash);
    if !snapshot_dir.exists() {
        return Ok(None);
    }

    let mut candidates = Vec::new();
    collect_gguf_files(&snapshot_dir, &mut candidates)?;
    match candidates.len() {
        0 => Ok(None),
        1 => Ok(Some(candidates.remove(0))),
        _ => Err(LlamaCppProviderError::Config(
            "Multiple GGUF files found in HuggingFace cache; set HuggingFace filename to choose one"
                .to_string(),
        )),
    }
}

fn collect_gguf_files(
    dir: &Path,
    candidates: &mut Vec<PathBuf>,
) -> Result<(), LlamaCppProviderError> {
    for entry in fs::read_dir(dir).map_err(|err| {
        LlamaCppProviderError::Other(format!(
            "Failed to read HuggingFace cache directory {}: {}",
            dir.display(),
            err
        ))
    })? {
        let entry = entry.map_err(|err| {
            LlamaCppProviderError::Other(format!(
                "Failed to read HuggingFace cache directory entry: {}",
                err
            ))
        })?;
        let file_type = entry.file_type().map_err(|err| {
            LlamaCppProviderError::Other(format!(
                "Failed to inspect HuggingFace cache entry {}: {}",
                entry.path().display(),
                err
            ))
        })?;
        let path = entry.path();
        if file_type.is_dir() {
            collect_gguf_files(&path, candidates)?;
            continue;
        }
        if path.extension().and_then(|ext| ext.to_str()) == Some("gguf")
            && (file_type.is_file() || file_type.is_symlink())
        {
            candidates.push(path);
        }
    }
    Ok(())
}

fn select_gguf_filename(siblings: &[Siblings]) -> Result<String, LlamaCppProviderError> {
    let mut gguf_files = siblings
        .iter()
        .filter_map(|sibling| {
            if sibling.rfilename.ends_with(".gguf") {
                Some(sibling.rfilename.clone())
            } else {
                None
            }
        })
        .collect::<Vec<_>>();

    gguf_files.sort();
    match gguf_files.len() {
        0 => Err(LlamaCppProviderError::Config(
            "No GGUF files found in HuggingFace repo".to_string(),
        )),
        1 => Ok(gguf_files.remove(0)),
        _ => Err(LlamaCppProviderError::Config(
            "Multiple GGUF files found in HuggingFace repo; set HuggingFace filename to choose one"
                .to_string(),
        )),
    }
}

static HUGGINGFACE_TOKEN_KEY: &str = "HUGGINGFACE_TOKEN";
static HF_TOKEN_KEY: &str = "HF_TOKEN";
static HUGGINGFACE_HUB_TOKEN_KEY: &str = "HUGGINGFACE_HUB_TOKEN";

fn hf_token_from_env<F>(mut get: F) -> Option<String>
where
    F: FnMut(&str) -> Option<String>,
{
    get(HUGGINGFACE_TOKEN_KEY)
        .or_else(|| get(HF_TOKEN_KEY))
        .or_else(|| get(HUGGINGFACE_HUB_TOKEN_KEY))
}

fn hf_token() -> Option<String> {
    hf_token_from_env(|key| std::env::var(key).ok())
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::collections::HashMap;

    #[test]
    fn test_resolve_cache_dir_relative() {
        let rel = "models/cache";
        let resolved = resolve_cache_dir(rel).unwrap();
        let cwd = std::env::current_dir().unwrap();
        assert_eq!(resolved, cwd.join(rel));
    }

    #[test]
    fn test_resolve_cache_dir_absolute() {
        let abs = if cfg!(windows) {
            "C:\\models"
        } else {
            "/tmp/models"
        };
        let resolved = resolve_cache_dir(abs).unwrap();
        assert_eq!(resolved, PathBuf::from(abs));
    }

    #[test]
    fn test_select_gguf_filename() {
        let siblings = vec![
            Siblings {
                rfilename: "a.txt".to_string(),
            },
            Siblings {
                rfilename: "model.gguf".to_string(),
            },
        ];
        let filename = select_gguf_filename(&siblings).unwrap();
        assert_eq!(filename, "model.gguf");
    }

    #[test]
    fn test_select_gguf_filename_none() {
        let siblings = vec![Siblings {
            rfilename: "a.txt".to_string(),
        }];
        let err = select_gguf_filename(&siblings).unwrap_err();
        assert!(err.to_string().contains("No GGUF files"));
    }

    #[test]
    fn test_select_gguf_filename_multiple() {
        let siblings = vec![
            Siblings {
                rfilename: "a.gguf".to_string(),
            },
            Siblings {
                rfilename: "b.gguf".to_string(),
            },
        ];
        let err = select_gguf_filename(&siblings).unwrap_err();
        assert!(err.to_string().contains("Multiple GGUF files"));
    }

    #[test]
    fn test_pick_cached_gguf_single() {
        let tmp = tempfile::tempdir().unwrap();
        let cache = Cache::new(tmp.path().to_path_buf());
        let repo =
            Repo::with_revision("org/model".to_string(), RepoType::Model, "main".to_string());
        let repo_dir = cache.path().join(repo.folder_name());
        let refs_dir = repo_dir.join("refs");
        let snapshots_dir = repo_dir.join("snapshots");
        std::fs::create_dir_all(&refs_dir).unwrap();
        std::fs::create_dir_all(&snapshots_dir).unwrap();
        std::fs::write(refs_dir.join(repo.revision()), "abc123").unwrap();
        let snapshot = snapshots_dir.join("abc123");
        std::fs::create_dir_all(&snapshot).unwrap();
        let gguf = snapshot.join("model.gguf");
        std::fs::write(&gguf, b"test").unwrap();

        let found = pick_cached_gguf(&cache, &repo).unwrap().unwrap();
        assert_eq!(found, gguf);
    }

    #[test]
    fn test_pick_cached_gguf_multiple() {
        let tmp = tempfile::tempdir().unwrap();
        let cache = Cache::new(tmp.path().to_path_buf());
        let repo =
            Repo::with_revision("org/model".to_string(), RepoType::Model, "main".to_string());
        let repo_dir = cache.path().join(repo.folder_name());
        let refs_dir = repo_dir.join("refs");
        let snapshots_dir = repo_dir.join("snapshots");
        std::fs::create_dir_all(&refs_dir).unwrap();
        std::fs::create_dir_all(&snapshots_dir).unwrap();
        std::fs::write(refs_dir.join(repo.revision()), "abc123").unwrap();
        let snapshot = snapshots_dir.join("abc123");
        std::fs::create_dir_all(&snapshot).unwrap();
        std::fs::write(snapshot.join("a.gguf"), b"a").unwrap();
        std::fs::write(snapshot.join("b.gguf"), b"b").unwrap();

        let err = pick_cached_gguf(&cache, &repo).unwrap_err();
        assert!(err.to_string().contains("Multiple GGUF files"));
    }

    #[test]
    fn test_hf_token_precedence() {
        let mut env = HashMap::<&str, &str>::new();
        let get = |key: &str| env.get(key).map(|value| value.to_string());
        assert!(hf_token_from_env(get).is_none());

        env.insert(HF_TOKEN_KEY, "hf");
        let get = |key: &str| env.get(key).map(|value| value.to_string());
        assert_eq!(hf_token_from_env(get).as_deref(), Some("hf"));

        env.insert(HUGGINGFACE_TOKEN_KEY, "primary");
        let get = |key: &str| env.get(key).map(|value| value.to_string());
        assert_eq!(hf_token_from_env(get).as_deref(), Some("primary"));

        env.remove(HUGGINGFACE_TOKEN_KEY);
        env.insert(HUGGINGFACE_HUB_TOKEN_KEY, "hub");
        let get = |key: &str| env.get(key).map(|value| value.to_string());
        assert_eq!(hf_token_from_env(get).as_deref(), Some("hf"));

        env.remove(HF_TOKEN_KEY);
        let get = |key: &str| env.get(key).map(|value| value.to_string());
        assert_eq!(hf_token_from_env(get).as_deref(), Some("hub"));
    }
}
