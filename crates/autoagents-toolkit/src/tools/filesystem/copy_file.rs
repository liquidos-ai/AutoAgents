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

use super::{BaseFileTool, FilesystemSandbox, prepare_mutation_path, sandbox_error};

#[derive(Serialize, Deserialize, ToolInput, Debug)]
pub struct CopyFileArgs {
    #[input(description = "Path of the file to copy")]
    source_path: String,
    #[input(description = "Path to save the copied file")]
    destination_path: String,
}

#[tool(
    name = "copy_file",
    description = "Copy a file from source path to destination path",
    input = CopyFileArgs,
)]
pub struct CopyFile {
    sandbox: FilesystemSandbox,
}

impl CopyFile {
    pub fn new(root: impl AsRef<Path>) -> std::io::Result<Self> {
        Ok(Self {
            sandbox: FilesystemSandbox::new(root)?,
        })
    }

    pub fn with_sandbox(sandbox: FilesystemSandbox) -> Self {
        Self { sandbox }
    }
}

impl BaseFileTool for CopyFile {
    fn sandbox(&self) -> &FilesystemSandbox {
        &self.sandbox
    }
}

#[async_trait]
impl ToolRuntime for CopyFile
where
    Self: BaseFileTool,
{
    async fn execute(&self, args: Value) -> Result<Value, ToolCallError> {
        let CopyFileArgs {
            source_path,
            destination_path,
        } = serde_json::from_value(args)?;

        debug!(
            "Copy File Executing: Source: {} - Destination: {}",
            source_path, destination_path
        );

        let src_path = self
            .sandbox()
            .resolve_relative(&source_path)
            .map_err(sandbox_error)?;

        if !src_path.exists() {
            return Err(ToolCallError::RuntimeError(
                format!("Source file does not exist: {}", src_path.display()).into(),
            ));
        }

        let src_path = self
            .sandbox()
            .ensure_resolved(&src_path)
            .map_err(sandbox_error)?;
        let dest_path = prepare_mutation_path(self.sandbox(), &destination_path).await?;

        let bytes_copied = fs::copy(&src_path, &dest_path)
            .await
            .map_err(|e| ToolCallError::RuntimeError(Box::new(e)))?;

        Ok(json!({
            "success": true,
            "source": self.sandbox().relative_path_display(&src_path),
            "destination": self.sandbox().relative_path_display(&dest_path),
            "bytes_copied": bytes_copied
        }))
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::io::Write;
    use tempfile::tempdir;

    #[tokio::test]
    async fn test_copy_file_success() {
        let temp_dir = tempdir().expect("Failed to create temp dir");
        let src_path = temp_dir.path().join("source.txt");
        let dest_path = temp_dir.path().join("destination.txt");

        let mut src_file = std::fs::File::create(&src_path).expect("Failed to create source file");
        src_file
            .write_all(b"Hello, World!")
            .expect("Failed to write to source file");
        drop(src_file);

        let copy_file = CopyFile::new(temp_dir.path()).expect("sandbox");
        let args = json!({
            "source_path": "source.txt",
            "destination_path": "destination.txt"
        });

        let result = copy_file.execute(args).await;
        assert!(result.is_ok());

        let content = std::fs::read_to_string(&dest_path).expect("Failed to read destination file");
        assert_eq!(content, "Hello, World!");
    }

    #[tokio::test]
    async fn test_copy_file_source_not_exists() {
        let temp_dir = tempdir().expect("Failed to create temp dir");

        let copy_file = CopyFile::new(temp_dir.path()).expect("sandbox");
        let args = json!({
            "source_path": "nonexistent.txt",
            "destination_path": "destination.txt"
        });

        let result = copy_file.execute(args).await;
        assert!(result.is_err());
    }

    #[tokio::test]
    async fn test_copy_file_with_root_dir() {
        let temp_dir = tempdir().expect("Failed to create temp dir");

        let src_file = temp_dir.path().join("source.txt");
        let mut file = std::fs::File::create(&src_file).expect("Failed to create source file");
        file.write_all(b"Test content")
            .expect("Failed to write to source file");
        drop(file);

        let copy_file = CopyFile::new(temp_dir.path()).expect("sandbox");
        let args = json!({
            "source_path": "source.txt",
            "destination_path": "dest.txt"
        });

        let result = copy_file.execute(args).await;
        assert!(result.is_ok());

        let dest_file = temp_dir.path().join("dest.txt");
        assert!(dest_file.exists());
    }
}
