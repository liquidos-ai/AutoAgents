use autoagents::core::{
    ractor::async_trait,
    tool::{ToolCallError, ToolInputT, ToolRuntime, ToolT},
};
use autoagents_derive::{ToolInput, tool};
use log::debug;
use serde::{Deserialize, Serialize};
use serde_json::{Value, json};
use tokio::fs;

use super::BaseFileTool;

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

impl ReadFile {
    pub fn new() -> Self {
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

impl Default for ReadFile {
    fn default() -> Self {
        Self::new()
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

        let path = self.get_relative_path(&file_path);

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

        // Ensure path is within root if root is set
        if self.root_dir().is_some() {
            self.ensure_within_root(&path)
                .map_err(|e| ToolCallError::RuntimeError(Box::new(e)))?;
        }

        let encoding = "utf8".to_string();

        match encoding.as_str() {
            "utf8" | "utf-8" => {
                let content = fs::read_to_string(&path)
                    .await
                    .map_err(|e| ToolCallError::RuntimeError(Box::new(e)))?;

                Ok(json!({
                    "success": true,
                    "path": path.display().to_string(),
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
                    "path": path.display().to_string(),
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
                    "path": path.display().to_string(),
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

        let read_file = ReadFile::new();
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

        let read_file = ReadFile::new();
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
}
