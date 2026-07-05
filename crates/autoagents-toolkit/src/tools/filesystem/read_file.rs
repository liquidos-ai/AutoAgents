use autoagents::core::{
    ractor::async_trait,
    tool::{ToolCallError, ToolRuntime, ToolT},
};
use autoagents_derive::{ToolInput, tool};
use log::debug;
use serde::{Deserialize, Serialize};
use serde_json::{Value, json};
use tokio::fs;

use super::{BaseFileTool, default_root_dir};

#[derive(Serialize, Deserialize, ToolInput, Debug)]
pub struct ReadFileArgs {
    #[input(description = "Path of the file to read")]
    file_path: String,
}

#[tool(
    name = "read_file",
    description = "Read the contents of a file from the filesystem",
    input = ReadFileArgs,
)]
pub struct ReadFile {
    root_dir: Option<String>,
}

impl Default for ReadFile {
    fn default() -> Self {
        Self {
            root_dir: Some(default_root_dir()),
        }
    }
}

impl ReadFile {
    pub fn new() -> Self {
        Self::default()
    }

    pub fn new_unrestricted() -> Self {
        Self { root_dir: None }
    }

    pub fn new_with_root_dir(root_dir: String) -> Self {
        Self {
            root_dir: Some(root_dir),
        }
    }
}

impl BaseFileTool for ReadFile {
    fn root_dir(&self) -> Option<String> {
        self.root_dir.clone()
    }
}

#[async_trait]
impl ToolRuntime for ReadFile
where
    Self: BaseFileTool,
{
    async fn execute(&self, args: Value) -> Result<Value, ToolCallError> {
        let ReadFileArgs { file_path } = serde_json::from_value(args)?;

        debug!("Read File Executing: Source: {}", file_path);

        let path = self
            .resolve_path(&file_path)
            .map_err(|e| ToolCallError::RuntimeError(Box::new(e)))?;

        // Validate file exists
        if !path.exists() {
            return Err(ToolCallError::RuntimeError(
                format!("File does not exist: {}", path.display()).into(),
            ));
        }

        if !path.is_file() {
            return Err(ToolCallError::RuntimeError(
                format!("Path is not a file: {}", path.display()).into(),
            ));
        }

        let encoding = "utf8".to_string();

        match encoding.as_str() {
            "utf8" | "utf-8" => {
                let content = fs::read_to_string(&path)
                    .await
                    .map_err(|e| ToolCallError::RuntimeError(Box::new(e)))?;

                Ok(json!({
                    "success": true,
                    "path": self.output_path(&path),
                    "content": content,
                    "encoding": encoding
                }))
            }
            "base64" => {
                let bytes = fs::read(&path)
                    .await
                    .map_err(|e| ToolCallError::RuntimeError(Box::new(e)))?;

                use base64::{Engine as _, engine::general_purpose::STANDARD};
                let encoded = STANDARD.encode(bytes);

                Ok(json!({
                    "success": true,
                    "path": self.output_path(&path),
                    "content": encoded,
                    "encoding": encoding
                }))
            }
            "bytes" => {
                let bytes = fs::read(&path)
                    .await
                    .map_err(|e| ToolCallError::RuntimeError(Box::new(e)))?;

                Ok(json!({
                    "success": true,
                    "path": self.output_path(&path),
                    "content": bytes,
                    "encoding": encoding
                }))
            }
            _ => Err(ToolCallError::RuntimeError(
                format!("Unsupported encoding: {}", encoding).into(),
            )),
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::io::Write;
    use tempfile::tempdir;

    #[tokio::test]
    async fn test_read_file_utf8() {
        let temp_dir = tempdir().expect("Failed to create temp dir");
        let file_path = temp_dir.path().join("test.txt");

        // Create test file
        let mut file = std::fs::File::create(&file_path).expect("Failed to create test file");
        file.write_all(b"Hello, World!")
            .expect("Failed to write to test file");
        drop(file);

        let read_file = ReadFile::new_unrestricted();
        let args = json!({
            "file_path": file_path.display().to_string()
        });

        let result = read_file.execute(args).await.expect("Failed to read file");
        let content = result.get("content").and_then(|v| v.as_str()).unwrap();
        assert_eq!(content, "Hello, World!");
    }

    // Base64 test removed - feature not implemented in simplified version

    #[tokio::test]
    async fn test_read_file_not_exists() {
        let temp_dir = tempdir().expect("Failed to create temp dir");
        let file_path = temp_dir.path().join("nonexistent.txt");

        let read_file = ReadFile::new_unrestricted();
        let args = json!({
            "file_path": file_path.display().to_string()
        });

        let result = read_file.execute(args).await;
        assert!(result.is_err());
    }

    #[tokio::test]
    async fn test_read_file_with_root_dir() {
        let temp_dir = tempdir().expect("Failed to create temp dir");
        let root_dir = temp_dir.path().to_str().unwrap().to_string();

        let file_path = temp_dir.path().join("test.txt");
        let mut file = std::fs::File::create(&file_path).expect("Failed to create test file");
        file.write_all(b"Test content")
            .expect("Failed to write to test file");
        drop(file);

        let read_file = ReadFile::new_with_root_dir(root_dir);
        let args = json!({
            "file_path": "test.txt"
        });

        let result = read_file.execute(args).await.expect("Failed to read file");
        let content = result.get("content").and_then(|v| v.as_str()).unwrap();
        assert_eq!(content, "Test content");
    }

    #[tokio::test]
    async fn test_read_file_rejects_absolute_path_with_root_dir() {
        let temp_dir = tempdir().expect("Failed to create temp dir");
        let root_dir = temp_dir.path().to_str().unwrap().to_string();
        let outside = temp_dir.path().join("outside.txt");
        std::fs::write(&outside, "outside").expect("Failed to create file");

        let read_file = ReadFile::new_with_root_dir(root_dir);
        let result = read_file
            .execute(json!({
                "file_path": outside.display().to_string()
            }))
            .await;

        assert!(result.is_err());
    }

    #[tokio::test]
    async fn test_read_file_rejects_traversal_with_root_dir() {
        let temp_dir = tempdir().expect("Failed to create temp dir");
        let root_dir = temp_dir.path().join("root");
        std::fs::create_dir_all(&root_dir).expect("Failed to create root");
        std::fs::write(temp_dir.path().join("outside.txt"), "outside")
            .expect("Failed to create outside file");

        let read_file = ReadFile::new_with_root_dir(root_dir.to_string_lossy().to_string());
        let result = read_file
            .execute(json!({
                "file_path": "../outside.txt"
            }))
            .await;

        assert!(result.is_err());
    }

    #[tokio::test]
    #[cfg(unix)]
    async fn test_read_file_rejects_symlink_escape_with_root_dir() {
        let temp_dir = tempdir().expect("Failed to create temp dir");
        let root_dir = temp_dir.path().join("root");
        let outside_dir = temp_dir.path().join("outside");
        std::fs::create_dir_all(&root_dir).expect("Failed to create root");
        std::fs::create_dir_all(&outside_dir).expect("Failed to create outside");
        std::fs::write(outside_dir.join("secret.txt"), "secret").expect("Failed to create secret");
        std::os::unix::fs::symlink(&outside_dir, root_dir.join("outside_link"))
            .expect("Failed to create symlink");

        let read_file = ReadFile::new_with_root_dir(root_dir.to_string_lossy().to_string());
        let result = read_file
            .execute(json!({
                "file_path": "outside_link/secret.txt"
            }))
            .await;

        assert!(result.is_err());
    }
}
