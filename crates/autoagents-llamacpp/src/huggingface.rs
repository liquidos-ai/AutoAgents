//! HuggingFace GGUF downloader and cache resolver.

use crate::config::LlamaCppConfig;
use crate::error::LlamaCppProviderError;
use serde::Deserialize;
use std::fs;
use std::io::{self, Read, Write};
use std::path::{Path, PathBuf};

#[derive(Debug, Deserialize)]
struct HfModelInfo {
    siblings: Vec<HfSibling>,
}

#[derive(Debug, Deserialize)]
struct HfSibling {
    rfilename: String,
}

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

    let models_dir = resolve_models_dir(config.model_dir.as_deref())?;
    let repo_dir = models_dir.join(repo_id);
    fs::create_dir_all(&repo_dir).map_err(|err| {
        LlamaCppProviderError::Other(format!(
            "Failed to create model cache directory {}: {}",
            repo_dir.display(),
            err
        ))
    })?;

    let filename = match filename_override.or(config.hf_filename.as_deref()) {
        Some(filename) => filename.to_string(),
        None => {
            if let Some(local) = pick_cached_gguf(&repo_dir)? {
                return Ok(local.to_string_lossy().to_string());
            }
            let model_info = fetch_model_info(repo_id, config.hf_revision.as_deref())?;
            select_gguf_filename(&model_info)?
        }
    };

    let model_path = repo_dir.join(&filename);
    if model_path.exists() {
        return Ok(model_path.to_string_lossy().to_string());
    }

    download_file(repo_id, &filename, config.hf_revision.as_deref(), &model_path)?;
    Ok(model_path.to_string_lossy().to_string())
}

fn resolve_models_dir(model_dir: Option<&str>) -> Result<PathBuf, LlamaCppProviderError> {
    let base = model_dir.unwrap_or("models");
    let path = PathBuf::from(base);
    if path.is_absolute() {
        return Ok(path);
    }
    let cwd = std::env::current_dir().map_err(|err| {
        LlamaCppProviderError::Other(format!("Failed to resolve current dir: {}", err))
    })?;
    Ok(cwd.join(path))
}

fn pick_cached_gguf(repo_dir: &Path) -> Result<Option<PathBuf>, LlamaCppProviderError> {
    if !repo_dir.exists() {
        return Ok(None);
    }
    let mut candidates = Vec::new();
    for entry in fs::read_dir(repo_dir).map_err(|err| {
        LlamaCppProviderError::Other(format!(
            "Failed to read model cache directory {}: {}",
            repo_dir.display(),
            err
        ))
    })? {
        let entry = entry.map_err(|err| {
            LlamaCppProviderError::Other(format!(
                "Failed to read model cache directory entry: {}",
                err
            ))
        })?;
        let path = entry.path();
        if path.extension().and_then(|ext| ext.to_str()) == Some("gguf") {
            candidates.push(path);
        }
    }

    match candidates.len() {
        0 => Ok(None),
        1 => Ok(Some(candidates.remove(0))),
        _ => Err(LlamaCppProviderError::Config(
            "Multiple GGUF files found in cache; set HuggingFace filename to choose one"
                .to_string(),
        )),
    }
}

fn fetch_model_info(
    repo_id: &str,
    revision: Option<&str>,
) -> Result<HfModelInfo, LlamaCppProviderError> {
    let mut url = format!("https://huggingface.co/api/models/{}", repo_id);
    if let Some(revision) = revision {
        url.push_str("?revision=");
        url.push_str(revision);
    }

    let request = ureq::get(&url);
    let request = apply_hf_auth(request);
    let response = request.call().map_err(|err| {
        LlamaCppProviderError::Other(format!("HuggingFace API error: {}", err))
    })?;
    let body = response
        .into_body()
        .read_to_string()
        .map_err(|err| LlamaCppProviderError::Other(format!("Failed to read HF response: {}", err)))?;

    serde_json::from_str(&body)
        .map_err(|err| LlamaCppProviderError::Other(format!("Failed to parse HF response: {}", err)))
}

fn select_gguf_filename(info: &HfModelInfo) -> Result<String, LlamaCppProviderError> {
    let mut gguf_files = info
        .siblings
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

fn download_file(
    repo_id: &str,
    filename: &str,
    revision: Option<&str>,
    dest: &Path,
) -> Result<(), LlamaCppProviderError> {
    let revision = revision.unwrap_or("main");
    let url = format!(
        "https://huggingface.co/{}/resolve/{}/{}",
        repo_id, revision, filename
    );

    let request = ureq::get(&url);
    let request = apply_hf_auth(request);
    let response = request.call().map_err(|err| {
        LlamaCppProviderError::Other(format!("HuggingFace download error: {}", err))
    })?;
    let total_bytes = response
        .headers()
        .get("content-length")
        .and_then(|value| value.to_str().ok())
        .and_then(|value| value.parse::<u64>().ok());

    let tmp_path = dest.with_extension("gguf.part");
    if tmp_path.exists() {
        let _ = fs::remove_file(&tmp_path);
    }

    let mut reader = response.into_body().into_reader();
    let mut file = fs::File::create(&tmp_path).map_err(|err| {
        LlamaCppProviderError::Other(format!("Failed to create file {}: {}", tmp_path.display(), err))
    })?;
    eprintln!("Downloading {}...", filename);
    let mut buffer = [0u8; 1024 * 128];
    let mut downloaded = 0u64;
    let mut last_pct = None;
    loop {
        let read = reader.read(&mut buffer).map_err(|err| {
            LlamaCppProviderError::Other(format!("Failed to download model: {}", err))
        })?;
        if read == 0 {
            break;
        }
        file.write_all(&buffer[..read]).map_err(|err| {
            LlamaCppProviderError::Other(format!("Failed to write model: {}", err))
        })?;
        downloaded += read as u64;
        if let Some(total) = total_bytes {
            let pct = (downloaded.saturating_mul(100) / total).min(100_u64);
            if last_pct != Some(pct) {
                eprint!("\rDownloading {}... {}%", filename, pct);
                let _ = io::stderr().flush();
                last_pct = Some(pct);
            }
        }
    }
    if total_bytes.is_some() {
        eprintln!("\rDownloading {}... 100%", filename);
    }

    fs::rename(&tmp_path, dest).map_err(|err| {
        LlamaCppProviderError::Other(format!(
            "Failed to finalize model download {}: {}",
            dest.display(),
            err
        ))
    })?;
    Ok(())
}

fn apply_hf_auth<B>(request: ureq::RequestBuilder<B>) -> ureq::RequestBuilder<B> {
    if let Some(token) = hf_token() {
        request.header("Authorization", format!("Bearer {}", token))
    } else {
        request
    }
}

fn hf_token() -> Option<String> {
    std::env::var("HUGGINGFACE_TOKEN")
        .ok()
        .or_else(|| std::env::var("HF_TOKEN").ok())
        .or_else(|| std::env::var("HUGGINGFACE_HUB_TOKEN").ok())
}
