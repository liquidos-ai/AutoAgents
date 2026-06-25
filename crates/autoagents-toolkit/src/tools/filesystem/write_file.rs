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

use super::{BaseFileTool, FilesystemSandbox, prepare_mutation_path};

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
    sandbox: FilesystemSandbox,
}

impl WriteFile {
    pub fn new(root: impl AsRef<Path>) -> std::io::Result<Self> {
        Ok(Self {
            sandbox: FilesystemSandbox::new(root)?,
        })
    }

    pub fn with_sandbox(sandbox: FilesystemSandbox) -> Self {
        Self { sandbox }
    }
}

impl BaseFileTool for WriteFile {
    fn sandbox(&self) -> &FilesystemSandbox {
        &self.sandbox
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

        let path = prepare_mutation_path(self.sandbox(), &file_path).await?;

        let bytes = content.into_bytes();

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
            "path": self.sandbox().relative_path_display(&path),
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

        let write_file = WriteFile::new(temp_dir.path()).expect("sandbox");
        let args = json!({
            "file_path": "new_file.txt",
            "content": "Hello, World!",
            "append": false
        });

        let result = write_file
            .execute(args)
            .await
            .expect("Failed to write file");
        assert!(result.get("success").and_then(|v| v.as_bool()).unwrap());

        let content = std::fs::read_to_string(&file_path).expect("Failed to read file");
        assert_eq!(content, "Hello, World!");
    }

    #[tokio::test]
    async fn test_write_file_overwrite() {
        let temp_dir = tempdir().expect("Failed to create temp dir");
        let file_path = temp_dir.path().join("test.txt");

        std::fs::write(&file_path, "Initial content").expect("Failed to create initial file");

        let write_file = WriteFile::new(temp_dir.path()).expect("sandbox");
        let args = json!({
            "file_path": "test.txt",
            "content": "New content",
            "append": false
        });

        let result = write_file
            .execute(args)
            .await
            .expect("Failed to write file");
        assert!(result.get("success").and_then(|v| v.as_bool()).unwrap());

        let content = std::fs::read_to_string(&file_path).expect("Failed to read file");
        assert_eq!(content, "New content");
    }

    #[tokio::test]
    async fn test_write_file_with_root_dir() {
        let temp_dir = tempdir().expect("Failed to create temp dir");

        let write_file = WriteFile::new(temp_dir.path()).expect("sandbox");
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
