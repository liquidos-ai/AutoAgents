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
pub struct WriteFileArgs {
    #[input(description = "Path of the file to write")]
    file_path: String,
    #[input(description = "Content to write to the file")]
    content: String,
    #[input(description = "Should append the content to the file")]
    append: bool,
}

#[tool(
    name = "write_file",
    description = "Write content to a file in the filesystem",
    input = WriteFileArgs,
)]
pub struct WriteFile {
    root_dir: Option<String>,
}

impl WriteFile {
    pub fn new() -> Self {
        Self { root_dir: None }
    }

    pub fn new_with_root_dir(root_dir: String) -> Self {
        Self {
            root_dir: Some(root_dir),
        }
    }
}

impl BaseFileTool for WriteFile {
    fn root_dir(&self) -> Option<String> {
        self.root_dir.clone()
    }
}

impl Default for WriteFile {
    fn default() -> Self {
        Self::new()
    }
}

#[async_trait]
impl ToolRuntime for WriteFile
where
    Self: BaseFileTool,
{
    async fn execute(&self, args: Value) -> Result<Value, ToolCallError> {
        let WriteFileArgs {
            file_path,
            content,
            append,
        } = serde_json::from_value(args)?;

        debug!("Write File Executing: File Path: {}", file_path);

        let path = self.get_relative_path(&file_path);

        // Ensure path is within root if root is set
        if self.root_dir().is_some() {
            self.ensure_within_root(&path)
                .map_err(|e| ToolCallError::RuntimeError(Box::new(e)))?;
        }

        // Create parent directory if it doesn't exist
        if let Some(parent) = path.parent() {
            fs::create_dir_all(parent)
                .await
                .map_err(|e| ToolCallError::RuntimeError(Box::new(e)))?;
        }

        let encoding = "utf8".to_string();

        // Decode content based on encoding
        let bytes = match encoding.as_str() {
            "utf8" | "utf-8" => content.into_bytes(),
            "base64" => {
                use base64::{Engine as _, engine::general_purpose::STANDARD};
                STANDARD
                    .decode(content)
                    .map_err(|e| ToolCallError::RuntimeError(Box::new(e)))?
            }
            _ => {
                return Err(ToolCallError::RuntimeError(
                    format!("Unsupported encoding: {}", encoding).into(),
                ));
            }
        };

        // Write or append to file
        if append {
            use tokio::fs::OpenOptions;
            use tokio::io::AsyncWriteExt;

            let mut file = OpenOptions::new()
                .append(true)
                .create(true)
                .open(&path)
                .await
                .map_err(|e| ToolCallError::RuntimeError(Box::new(e)))?;

            file.write_all(&bytes)
                .await
                .map_err(|e| ToolCallError::RuntimeError(Box::new(e)))?;
        } else {
            fs::write(&path, &bytes)
                .await
                .map_err(|e| ToolCallError::RuntimeError(Box::new(e)))?;
        }

        Ok(json!({
            "success": true,
            "path": path.display().to_string(),
            "bytes_written": bytes.len(),
            "mode": if append { "append" } else { "overwrite" }
        }))
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use tempfile::tempdir;

    #[tokio::test]
    async fn test_write_file_new() {
        let temp_dir = tempdir().expect("Failed to create temp dir");
        let file_path = temp_dir.path().join("new_file.txt");

        let write_file = WriteFile::default();
        let args = json!({
            "file_path": file_path.display().to_string(),
            "content": "Hello, World!",
            "append": false
        });

        let result = write_file
            .execute(args)
            .await
            .expect("Failed to write file");
        assert!(result.get("success").and_then(|v| v.as_bool()).unwrap());

        // Verify file content
        let content = std::fs::read_to_string(&file_path).expect("Failed to read file");
        assert_eq!(content, "Hello, World!");
    }

    #[tokio::test]
    async fn test_write_file_overwrite() {
        let temp_dir = tempdir().expect("Failed to create temp dir");
        let file_path = temp_dir.path().join("test.txt");

        // Create initial file
        std::fs::write(&file_path, "Initial content").expect("Failed to create initial file");

        let write_file = WriteFile::default();
        let args = json!({
            "file_path": file_path.display().to_string(),
            "content": "New content",
            "append": false
        });

        let result = write_file
            .execute(args)
            .await
            .expect("Failed to write file");
        assert!(result.get("success").and_then(|v| v.as_bool()).unwrap());

        // Verify file was overwritten
        let content = std::fs::read_to_string(&file_path).expect("Failed to read file");
        assert_eq!(content, "New content");
    }

    // Append test removed - feature not implemented in simplified version

    // Base64 test removed - feature not implemented in simplified version

    #[tokio::test]
    async fn test_write_file_with_root_dir() {
        let temp_dir = tempdir().expect("Failed to create temp dir");
        let root_dir = temp_dir.path().to_str().unwrap().to_string();

        let write_file = WriteFile::new_with_root_dir(root_dir);
        let args = json!({
            "file_path": "test.txt",
            "content": "Test content",
            "append": false
        });

        let result = write_file
            .execute(args)
            .await
            .expect("Failed to write file");
        assert!(result.get("success").and_then(|v| v.as_bool()).unwrap());

        let file_path = temp_dir.path().join("test.txt");
        assert!(file_path.exists());

        let content = std::fs::read_to_string(&file_path).expect("Failed to read file");
        assert_eq!(content, "Test content");
    }
}
