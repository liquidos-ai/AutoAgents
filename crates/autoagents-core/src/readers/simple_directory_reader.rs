use std::collections::HashSet;
use std::ffi::OsStr;
use std::fs;
use std::path::{Path, PathBuf};

use serde_json::json;
use walkdir::WalkDir;

use crate::document::Document;

#[derive(Debug, thiserror::Error)]
pub enum ReaderError {
    #[error("Root path does not exist: {0}")]
    MissingPath(PathBuf),

    #[error("Failed to read file {path:?}: {source}")]
    Io {
        path: PathBuf,
        source: std::io::Error,
    },

    #[error("File {0:?} is not valid UTF-8")]
    Utf8(PathBuf),
}

#[derive(Clone, Debug)]
pub struct SimpleDirectoryReader {
    root: PathBuf,
    recursive: bool,
    extensions: Option<HashSet<String>>,
}

impl SimpleDirectoryReader {
    pub fn new(root: impl Into<PathBuf>) -> Self {
        Self {
            root: root.into(),
            recursive: true,
            extensions: None,
        }
    }

    /// Limit the reader to a specific set of extensions (without dots).
    pub fn with_extensions<I, S>(mut self, extensions: I) -> Self
    where
        I: IntoIterator<Item = S>,
        S: Into<String>,
    {
        self.extensions = Some(extensions.into_iter().map(|ext| ext.into()).collect());
        self
    }

    pub fn recursive(mut self, recursive: bool) -> Self {
        self.recursive = recursive;
        self
    }

    pub fn load_data(&self) -> Result<Vec<Document>, ReaderError> {
        if !self.root.exists() {
            return Err(ReaderError::MissingPath(self.root.clone()));
        }

        let mut docs = Vec::new();
        let walker = if self.recursive {
            WalkDir::new(&self.root)
        } else {
            WalkDir::new(&self.root).max_depth(1)
        };

        for entry in walker {
            let entry = match entry {
                Ok(e) => e,
                Err(err) => {
                    return Err(ReaderError::Io {
                        path: self.root.clone(),
                        source: std::io::Error::other(err),
                    });
                }
            };

            if entry.file_type().is_dir() {
                continue;
            }

            if let Some(exts) = &self.extensions {
                if let Some(ext) = entry.path().extension().and_then(OsStr::to_str) {
                    if !exts.contains(ext) {
                        continue;
                    }
                } else {
                    continue;
                }
            }

            let content = match fs::read_to_string(entry.path()) {
                Ok(content) => content,
                Err(err) if err.kind() == std::io::ErrorKind::InvalidData => {
                    return Err(ReaderError::Utf8(entry.path().to_path_buf()));
                }
                Err(source) => {
                    return Err(ReaderError::Io {
                        path: entry.path().to_path_buf(),
                        source,
                    });
                }
            };

            let relative = path_relative_to(entry.path(), &self.root)
                .unwrap_or_else(|| entry.file_name().to_string_lossy().to_string());

            let metadata = json!({
                "source": relative,
                "absolute_path": entry.path().to_string_lossy(),
                "extension": entry.path().extension().and_then(OsStr::to_str).unwrap_or_default(),
            });

            docs.push(Document::with_metadata(content, metadata));
        }

        Ok(docs)
    }
}

fn path_relative_to(path: &Path, base: &Path) -> Option<String> {
    path.strip_prefix(base)
        .ok()
        .map(|p| p.to_string_lossy().to_string())
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::fs;

    #[test]
    fn test_missing_path_error() {
        let reader = SimpleDirectoryReader::new("/nonexistent/path/xyz123");
        let result = reader.load_data();
        assert!(result.is_err());
        assert!(matches!(result.unwrap_err(), ReaderError::MissingPath(_)));
    }

    #[test]
    fn test_empty_directory() {
        let dir = std::env::temp_dir().join("autoagents_test_empty_dir");
        fs::create_dir_all(&dir).unwrap();
        let reader = SimpleDirectoryReader::new(&dir);
        let docs = reader.load_data().unwrap();
        assert!(docs.is_empty());
        fs::remove_dir_all(&dir).ok();
    }

    #[test]
    fn test_single_file_load() {
        let dir = std::env::temp_dir().join("autoagents_test_single_file");
        fs::create_dir_all(&dir).unwrap();
        fs::write(dir.join("test.txt"), "hello world").unwrap();

        let reader = SimpleDirectoryReader::new(&dir);
        let docs = reader.load_data().unwrap();
        assert_eq!(docs.len(), 1);
        assert_eq!(docs[0].page_content, "hello world");
        assert_eq!(docs[0].metadata["extension"], "txt");

        fs::remove_dir_all(&dir).ok();
    }

    #[test]
    fn test_extension_filter() {
        let dir = std::env::temp_dir().join("autoagents_test_ext_filter");
        fs::create_dir_all(&dir).unwrap();
        fs::write(dir.join("file.txt"), "text").unwrap();
        fs::write(dir.join("file.md"), "markdown").unwrap();

        let reader = SimpleDirectoryReader::new(&dir).with_extensions(["txt"]);
        let docs = reader.load_data().unwrap();
        assert_eq!(docs.len(), 1);
        assert_eq!(docs[0].page_content, "text");

        fs::remove_dir_all(&dir).ok();
    }

    #[test]
    fn test_non_recursive_mode() {
        let dir = std::env::temp_dir().join("autoagents_test_nonrecursive");
        let sub = dir.join("sub");
        fs::create_dir_all(&sub).unwrap();
        fs::write(dir.join("top.txt"), "top").unwrap();
        fs::write(sub.join("nested.txt"), "nested").unwrap();

        let reader = SimpleDirectoryReader::new(&dir).recursive(false);
        let docs = reader.load_data().unwrap();
        assert_eq!(docs.len(), 1);
        assert_eq!(docs[0].page_content, "top");

        fs::remove_dir_all(&dir).ok();
    }

    #[test]
    fn test_relative_path_metadata() {
        let dir = std::env::temp_dir().join("autoagents_test_relpath");
        fs::create_dir_all(&dir).unwrap();
        fs::write(dir.join("file.txt"), "content").unwrap();

        let reader = SimpleDirectoryReader::new(&dir);
        let docs = reader.load_data().unwrap();
        assert_eq!(docs.len(), 1);
        assert_eq!(docs[0].metadata["source"], "file.txt");

        fs::remove_dir_all(&dir).ok();
    }

    #[test]
    fn test_recursive_mode() {
        let dir = std::env::temp_dir().join("autoagents_test_recursive");
        let sub = dir.join("sub");
        fs::create_dir_all(&sub).unwrap();
        fs::write(dir.join("top.txt"), "top").unwrap();
        fs::write(sub.join("nested.txt"), "nested").unwrap();

        let reader = SimpleDirectoryReader::new(&dir);
        let docs = reader.load_data().unwrap();
        assert_eq!(docs.len(), 2);

        fs::remove_dir_all(&dir).ok();
    }

    #[test]
    fn test_path_relative_to_fn() {
        let result = path_relative_to(Path::new("/a/b/c.txt"), Path::new("/a/b"));
        assert_eq!(result, Some("c.txt".to_string()));
    }

    #[test]
    fn test_path_relative_to_fn_no_prefix() {
        let result = path_relative_to(Path::new("/x/y.txt"), Path::new("/a/b"));
        assert_eq!(result, None);
    }
}
