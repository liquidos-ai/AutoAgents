use std::path::Path;

use tokio::fs::File;
use tokio::io::AsyncReadExt;

use super::error::DocumentSourceError;
use crate::tools::document_parsing::config::DocumentParserConfig;
use crate::utils::path_sandbox;

pub async fn load_local_file(
    path: &str,
    config: &DocumentParserConfig,
) -> Result<Vec<u8>, DocumentSourceError> {
    let path = Path::new(path);

    let canonical = match config.allowed_roots.as_deref() {
        Some(roots) => path_sandbox::ensure_within_any_root(path, roots).map_err(|error| {
            if error.kind() == std::io::ErrorKind::PermissionDenied {
                DocumentSourceError::PathOutsideRoot {
                    path: path.to_path_buf(),
                }
            } else {
                DocumentSourceError::Io(error)
            }
        })?,
        None => path_sandbox::canonicalize_or_normalize(path),
    };

    read_bounded_file(&canonical, config.max_local_file_bytes).await
}

async fn read_bounded_file(path: &Path, max_bytes: usize) -> Result<Vec<u8>, DocumentSourceError> {
    let metadata = tokio::fs::metadata(path)
        .await
        .map_err(DocumentSourceError::from)?;
    if metadata.len() > max_bytes as u64 {
        return Err(DocumentSourceError::LocalFileTooLarge {
            limit: max_bytes,
            observed: metadata.len() as usize,
        });
    }

    let mut file = File::open(path).await.map_err(DocumentSourceError::from)?;
    let mut collected = Vec::with_capacity(metadata.len().min(max_bytes as u64) as usize);
    let mut chunk = [0u8; 8_192];

    loop {
        let read = file
            .read(&mut chunk)
            .await
            .map_err(DocumentSourceError::from)?;
        if read == 0 {
            break;
        }

        if collected.len() + read > max_bytes {
            return Err(DocumentSourceError::LocalFileTooLarge {
                limit: max_bytes,
                observed: collected.len() + read,
            });
        }

        collected.extend_from_slice(&chunk[..read]);
    }

    Ok(collected)
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::io::Write;
    use std::path::PathBuf;
    use tempfile::tempdir;

    use crate::tools::document_parsing::config::DEFAULT_MAX_LOCAL_FILE_BYTES;

    fn test_config(
        max_local_file_bytes: usize,
        roots: Option<Vec<PathBuf>>,
    ) -> DocumentParserConfig {
        let mut config =
            DocumentParserConfig::default().with_max_local_file_bytes(max_local_file_bytes);
        if let Some(roots) = roots {
            config = config.with_allowed_roots(roots);
        }
        config
    }

    #[tokio::test]
    async fn load_without_roots_allows_readable_file() {
        let dir = tempdir().expect("tempdir");
        let file_path = dir.path().join("doc.txt");
        std::fs::File::create(&file_path)
            .expect("create")
            .write_all(b"hello")
            .expect("write");

        let bytes = load_local_file(
            &file_path.display().to_string(),
            &test_config(DEFAULT_MAX_LOCAL_FILE_BYTES, None),
        )
        .await
        .expect("read file");
        assert_eq!(bytes, b"hello");
    }

    #[tokio::test]
    async fn load_with_roots_blocks_outside_path() {
        let dir = tempdir().expect("tempdir");
        let outside =
            std::env::temp_dir().join(format!("outside-autoagents-{}", std::process::id()));
        std::fs::write(&outside, b"secret").expect("write outside file");

        let error = load_local_file(
            &outside.display().to_string(),
            &test_config(
                DEFAULT_MAX_LOCAL_FILE_BYTES,
                Some(vec![dir.path().to_path_buf()]),
            ),
        )
        .await
        .expect_err("outside path should be blocked");

        assert!(matches!(error, DocumentSourceError::PathOutsideRoot { .. }));

        std::fs::remove_file(outside).ok();
    }

    #[tokio::test]
    async fn load_with_roots_allows_inside_path() {
        let dir = tempdir().expect("tempdir");
        let file_path = dir.path().join("doc.txt");
        std::fs::write(&file_path, b"inside").expect("write");

        let bytes = load_local_file(
            &file_path.display().to_string(),
            &test_config(
                DEFAULT_MAX_LOCAL_FILE_BYTES,
                Some(vec![dir.path().to_path_buf()]),
            ),
        )
        .await
        .expect("inside path allowed");

        assert_eq!(bytes, b"inside");
    }

    #[tokio::test]
    async fn load_rejects_oversized_local_file() {
        let dir = tempdir().expect("tempdir");
        let file_path = dir.path().join("large.txt");
        std::fs::write(&file_path, vec![b'a'; 2048]).expect("write");

        let error = load_local_file(&file_path.display().to_string(), &test_config(1024, None))
            .await
            .expect_err("oversized local file");

        assert!(matches!(
            error,
            DocumentSourceError::LocalFileTooLarge { .. }
        ));
    }
}
