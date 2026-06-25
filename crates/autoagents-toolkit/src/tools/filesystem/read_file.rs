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
    sandbox: FilesystemSandbox,
}

impl ReadFile {
    pub fn new(root: impl AsRef<Path>) -> std::io::Result<Self> {
        Ok(Self {
            sandbox: FilesystemSandbox::new(root)?,
        })
    }
    pub fn with_sandbox(sandbox: FilesystemSandbox) -> Self {
        Self { sandbox }
    }
}

impl BaseFileTool for ReadFile {
    fn sandbox(&self) -> &FilesystemSandbox {
        &self.sandbox
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
            .sandbox()
            .resolve_relative(&file_path)
            .map_err(sandbox_error)?;

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

        let path = self
            .sandbox()
            .ensure_resolved(&path)
            .map_err(sandbox_error)?;

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

        let mut file = std::fs::File::create(&file_path).expect("Failed to create test file");
        file.write_all(b"Hello, World!")
            .expect("Failed to write to test file");
        drop(file);

        let read_file = ReadFile::new(temp_dir.path()).expect("sandbox");
        let args = json!({
            "file_path": "test.txt"
        });

        let result = read_file.execute(args).await.expect("Failed to read file");
        let content = result.get("content").and_then(|v| v.as_str()).unwrap();
        assert_eq!(content, "Hello, World!");
    }

    #[tokio::test]
    async fn test_read_file_not_exists() {
        let temp_dir = tempdir().expect("Failed to create temp dir");

        let read_file = ReadFile::new(temp_dir.path()).expect("sandbox");
        let args = json!({
            "file_path": "nonexistent.txt"
        });

        let result = read_file.execute(args).await;
        assert!(result.is_err());
    }

    #[tokio::test]
    async fn test_read_file_with_root_dir() {
        let temp_dir = tempdir().expect("Failed to create temp dir");

        let file_path = temp_dir.path().join("test.txt");
        let mut file = std::fs::File::create(&file_path).expect("Failed to create test file");
        file.write_all(b"Test content")
            .expect("Failed to write to test file");
        drop(file);

        let read_file = ReadFile::new(temp_dir.path()).expect("sandbox");
        let args = json!({
            "file_path": "test.txt"
        });

        let result = read_file.execute(args).await.expect("Failed to read file");
        let content = result.get("content").and_then(|v| v.as_str()).unwrap();
        assert_eq!(content, "Test content");
    }
}
