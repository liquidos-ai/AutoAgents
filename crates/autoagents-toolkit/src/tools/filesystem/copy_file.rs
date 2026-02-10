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
#[derive(Default)]
pub struct CopyFile {
    root_dir: Option<String>,
}

impl CopyFile {
    pub fn new() -> Self {
        Self::default()
    }

    pub fn new_with_root_dir(root_dir: String) -> Self {
        Self {
            root_dir: Some(root_dir),
        }
    }
}

impl BaseFileTool for CopyFile {
    fn root_dir(&self) -> Option<String> {
        self.root_dir.clone()
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

        let src_path = self.get_relative_path(&source_path);
        let dest_path = self.get_relative_path(&destination_path);

        // Validate source exists
        if !src_path.exists() {
            return Err(ToolCallError::RuntimeError(
                format!("Source file does not exist: {}", src_path.display()).into(),
            ));
        }

        // Ensure paths are within root if root is set
        if self.root_dir().is_some() {
            self.ensure_within_root(&src_path)
                .map_err(|e| ToolCallError::RuntimeError(Box::new(e)))?;
            self.ensure_within_root(&dest_path)
                .map_err(|e| ToolCallError::RuntimeError(Box::new(e)))?;
        }

        // Create parent directory if it doesn't exist
        if let Some(parent) = dest_path.parent() {
            fs::create_dir_all(parent)
                .await
                .map_err(|e| ToolCallError::RuntimeError(Box::new(e)))?;
        }

        // Perform the copy
        let bytes_copied = fs::copy(&src_path, &dest_path)
            .await
            .map_err(|e| ToolCallError::RuntimeError(Box::new(e)))?;

        Ok(json!({
            "success": true,
            "source": src_path.display().to_string(),
            "destination": dest_path.display().to_string(),
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

        // Create source file
        let mut src_file = std::fs::File::create(&src_path).expect("Failed to create source file");
        src_file
            .write_all(b"Hello, World!")
            .expect("Failed to write to source file");
        drop(src_file);

        let copy_file = CopyFile::new();
        let args = json!({
            "source_path": src_path.display().to_string(),
            "destination_path": dest_path.display().to_string()
        });

        let result = copy_file.execute(args).await;
        assert!(result.is_ok());

        // Verify destination file exists and has correct content
        let content = std::fs::read_to_string(&dest_path).expect("Failed to read destination file");
        assert_eq!(content, "Hello, World!");
    }

    #[tokio::test]
    async fn test_copy_file_source_not_exists() {
        let temp_dir = tempdir().expect("Failed to create temp dir");
        let src_path = temp_dir.path().join("nonexistent.txt");
        let dest_path = temp_dir.path().join("destination.txt");

        let copy_file = CopyFile::new();
        let args = json!({
            "source_path": src_path.display().to_string(),
            "destination_path": dest_path.display().to_string()
        });

        let result = copy_file.execute(args).await;
        assert!(result.is_err());
    }

    #[tokio::test]
    async fn test_copy_file_with_root_dir() {
        let temp_dir = tempdir().expect("Failed to create temp dir");
        let root_dir = temp_dir.path().to_str().unwrap().to_string();

        let src_file = temp_dir.path().join("source.txt");
        let mut file = std::fs::File::create(&src_file).expect("Failed to create source file");
        file.write_all(b"Test content")
            .expect("Failed to write to source file");
        drop(file);

        let copy_file = CopyFile::new_with_root_dir(root_dir);
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
