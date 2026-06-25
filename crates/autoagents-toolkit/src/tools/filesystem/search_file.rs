use std::path::Path;

use autoagents::core::{
    ractor::async_trait,
    tool::{ToolCallError, ToolRuntime, ToolT},
};
use autoagents_derive::{ToolInput, tool};
use log::debug;
use serde::{Deserialize, Serialize};
use serde_json::{Value, json};
use tokio::fs;

use super::{BaseFileTool, FilesystemSandbox, sandbox_error};

#[derive(Serialize, Deserialize, ToolInput, Debug)]
pub struct SearchFileArgs {
    #[input(description = "Directory path to search in")]
    directory: String,
    #[input(description = "Pattern to search for (supports wildcards: * and ?)")]
    pattern: String,
}

#[tool(
    name = "search_file",
    description = "Search for files by name pattern or content in a directory",
    input = SearchFileArgs,
)]
pub struct SearchFile {
    sandbox: FilesystemSandbox,
    max_iterations: usize,
}

impl SearchFile {
    pub fn new(root: impl AsRef<Path>, max_iterations: usize) -> std::io::Result<Self> {
        Ok(Self {
            sandbox: FilesystemSandbox::new(root)?,
            max_iterations,
        })
    }

    pub fn with_sandbox(sandbox: FilesystemSandbox, max_iterations: usize) -> Self {
        Self {
            sandbox,
            max_iterations,
        }
    }

    fn matches_pattern(filename: &str, pattern: &str, case_sensitive: bool) -> bool {
        let filename = if case_sensitive {
            filename.to_string()
        } else {
            filename.to_lowercase()
        };

        let pattern = if case_sensitive {
            pattern.to_string()
        } else {
            pattern.to_lowercase()
        };

        let pattern_parts: Vec<&str> = pattern.split('*').collect();

        if pattern_parts.len() == 1 {
            return filename.contains(&pattern);
        }

        let mut pos = 0;
        for (i, part) in pattern_parts.iter().enumerate() {
            if part.is_empty() {
                continue;
            }

            if i == 0 && !pattern.starts_with('*') {
                if !filename.starts_with(part) {
                    return false;
                }
                pos = part.len();
            } else if i == pattern_parts.len() - 1 && !pattern.ends_with('*') {
                return filename.ends_with(part);
            } else if let Some(found_pos) = filename[pos..].find(part) {
                pos += found_pos + part.len();
            } else {
                return false;
            }
        }

        true
    }

    async fn search_content_in_file(
        file_path: &Path,
        pattern: &str,
        case_sensitive: bool,
    ) -> Result<bool, ToolCallError> {
        let content = match fs::read_to_string(file_path).await {
            Ok(content) => content,
            Err(_) => return Ok(false),
        };

        if case_sensitive {
            Ok(content.contains(pattern))
        } else {
            Ok(content.to_lowercase().contains(&pattern.to_lowercase()))
        }
    }
}

impl BaseFileTool for SearchFile {
    fn sandbox(&self) -> &FilesystemSandbox {
        &self.sandbox
    }
}

#[async_trait]
impl ToolRuntime for SearchFile
where
    Self: BaseFileTool,
{
    async fn execute(&self, args: Value) -> Result<Value, ToolCallError> {
        let SearchFileArgs { directory, pattern } = serde_json::from_value(args)?;

        debug!(
            "Search File Executing: Directory: {} - Pattern: {}",
            directory, pattern
        );

        let dir_path = self
            .sandbox()
            .resolve_relative(&directory)
            .map_err(sandbox_error)?;
        let search_content = false;
        let case_sensitive = true;

        if !dir_path.exists() {
            return Err(ToolCallError::RuntimeError(
                format!("Directory does not exist: {}", dir_path.display()).into(),
            ));
        }

        if !dir_path.is_dir() {
            return Err(ToolCallError::RuntimeError(
                format!("Path is not a directory: {}", dir_path.display()).into(),
            ));
        }

        let dir_path = self
            .sandbox()
            .ensure_resolved(&dir_path)
            .map_err(sandbox_error)?;

        let mut results = Vec::new();
        let mut iterations = 0usize;
        let mut iteration_limit_reached = false;

        let walker = self.sandbox().walk_dir(&dir_path);

        for entry in walker.into_iter() {
            if iterations >= self.max_iterations {
                iteration_limit_reached = true;
                break;
            }

            iterations += 1;
            let entry = entry.map_err(|e| ToolCallError::RuntimeError(Box::new(e)))?;

            let path = entry.path();
            if path == dir_path {
                continue;
            }

            if entry.file_type().is_dir() {
                continue;
            }

            let validated_path = self
                .sandbox()
                .validate_walk_entry(path)
                .map_err(sandbox_error)?;

            let file_name = entry.file_name().to_string_lossy().to_string();
            let metadata = entry
                .metadata()
                .map_err(|e| ToolCallError::RuntimeError(Box::new(e)))?;

            if search_content {
                if Self::search_content_in_file(&validated_path, &pattern, case_sensitive).await? {
                    results.push(json!({
                        "path": validated_path.display().to_string(),
                        "name": file_name,
                        "size": metadata.len(),
                        "match_type": "content"
                    }));
                }
            } else if Self::matches_pattern(&file_name, &pattern, case_sensitive) {
                results.push(json!({
                    "path": validated_path.display().to_string(),
                    "name": file_name,
                    "size": metadata.len(),
                    "match_type": "filename"
                }));
            }
        }

        Ok(json!({
            "success": true,
            "directory": dir_path.display().to_string(),
            "pattern": pattern,
            "max_iterations": self.max_iterations,
            "iterations": iterations,
            "iteration_limit_reached": iteration_limit_reached,
            "search_type": if search_content { "content" } else { "filename" },
            "count": results.len(),
            "results": results
        }))
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use tempfile::tempdir;

    #[tokio::test]
    async fn test_search_by_filename() {
        let temp_dir = tempdir().expect("Failed to create temp dir");

        std::fs::write(temp_dir.path().join("test.txt"), "content")
            .expect("Failed to create test.txt");
        std::fs::write(temp_dir.path().join("data.json"), "{}")
            .expect("Failed to create data.json");
        std::fs::write(temp_dir.path().join("test.rs"), "fn main()")
            .expect("Failed to create test.rs");

        let search_file = SearchFile::new(temp_dir.path(), 100).expect("sandbox");
        let args = json!({
            "directory": ".",
            "pattern": "test*"
        });

        let result = search_file
            .execute(args)
            .await
            .expect("Failed to search files");
        let results = result.get("results").and_then(|v| v.as_array()).unwrap();
        assert_eq!(results.len(), 2);
    }

    #[tokio::test]
    async fn test_max_iterations_limit() {
        let temp_dir = tempdir().expect("Failed to create temp dir");

        for idx in 0..5 {
            let path = temp_dir.path().join(format!("test_{idx}.txt"));
            std::fs::write(path, "content").expect("Failed to create test file");
        }

        let search_file = SearchFile::new(temp_dir.path(), 3).expect("sandbox");
        let args = json!({
            "directory": ".",
            "pattern": "test*"
        });

        let result = search_file
            .execute(args)
            .await
            .expect("Failed to search files");

        assert_eq!(
            result.get("max_iterations").and_then(|v| v.as_u64()),
            Some(3)
        );
        assert_eq!(
            result
                .get("iteration_limit_reached")
                .and_then(|v| v.as_bool()),
            Some(true)
        );
    }

    #[tokio::test]
    async fn test_pattern_matching() {
        assert!(SearchFile::matches_pattern("test.txt", "test*", true));
        assert!(SearchFile::matches_pattern("test.txt", "*txt", true));
        assert!(SearchFile::matches_pattern("test.txt", "test.txt", true));
        assert!(SearchFile::matches_pattern("test_file.txt", "*file*", true));
        assert!(!SearchFile::matches_pattern("test.txt", "data*", true));

        assert!(SearchFile::matches_pattern("TEST.txt", "test*", false));
        assert!(!SearchFile::matches_pattern("TEST.txt", "test*", true));
    }
}
