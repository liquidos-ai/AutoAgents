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
pub struct DeleteFileArgs {
    #[input(description = "Path of the file or directory to delete")]
    path: String,
}

#[tool(
    name = "delete_file",
    description = "Delete a file or directory from the filesystem",
    input = DeleteFileArgs,
)]
#[derive(Default)]
pub struct DeleteFile {
    root_dir: Option<String>,
}

impl DeleteFile {
    pub fn new() -> Self {
        Self::default()
    }

    pub fn new_with_root_dir(root_dir: String) -> Self {
        Self {
            root_dir: Some(root_dir),
        }
    }
}

impl BaseFileTool for DeleteFile {
    fn root_dir(&self) -> Option<String> {
        self.root_dir.clone()
    }
}

#[async_trait]
impl ToolRuntime for DeleteFile
where
    Self: BaseFileTool,
{
    async fn execute(&self, args: Value) -> Result<Value, ToolCallError> {
        let DeleteFileArgs { path } = serde_json::from_value(args)?;

        debug!("Delete File Executing: Source: {}", path);

        let file_path = self.get_relative_path(&path);
        let recursive = true; // Always allow recursive deletion for safety

        // Validate path exists
        if !file_path.exists() {
            return Err(ToolCallError::RuntimeError(
                format!("Path does not exist: {}", file_path.display()).into(),
            ));
        }

        // Ensure path is within root if root is set
        if self.root_dir().is_some() {
            self.ensure_within_root(&file_path)
                .map_err(|e| ToolCallError::RuntimeError(Box::new(e)))?;
        }

        let is_dir = file_path.is_dir();

        // Delete file or directory
        if is_dir {
            if recursive {
                fs::remove_dir_all(&file_path)
                    .await
                    .map_err(|e| ToolCallError::RuntimeError(Box::new(e)))?;
            } else {
                fs::remove_dir(&file_path)
                    .await
                    .map_err(|e| ToolCallError::RuntimeError(Box::new(e)))?;
            }
        } else {
            fs::remove_file(&file_path)
                .await
                .map_err(|e| ToolCallError::RuntimeError(Box::new(e)))?;
        }

        Ok(json!({
            "success": true,
            "path": file_path.display().to_string(),
            "type": if is_dir { "directory" } else { "file" }
        }))
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::io::Write;
    use tempfile::tempdir;

    #[tokio::test]
    async fn test_delete_file() {
        let temp_dir = tempdir().expect("Failed to create temp dir");
        let file_path = temp_dir.path().join("test.txt");

        // Create test file
        let mut file = std::fs::File::create(&file_path).expect("Failed to create test file");
        file.write_all(b"Test content")
            .expect("Failed to write to test file");
        drop(file);

        assert!(file_path.exists());

        let delete_file = DeleteFile::default();
        let args = json!({
            "path": file_path.display().to_string()
        });

        let result = delete_file
            .execute(args)
            .await
            .expect("Failed to delete file");
        assert!(result.get("success").and_then(|v| v.as_bool()).unwrap());
        assert!(!file_path.exists());
    }

    #[tokio::test]
    async fn test_delete_empty_directory() {
        let temp_dir = tempdir().expect("Failed to create temp dir");
        let dir_path = temp_dir.path().join("test_dir");

        std::fs::create_dir_all(&dir_path).expect("Failed to create test directory");
        assert!(dir_path.exists());

        let delete_file = DeleteFile::default();
        let args = json!({
            "path": dir_path.display().to_string()
        });

        let result = delete_file
            .execute(args)
            .await
            .expect("Failed to delete directory");
        assert!(result.get("success").and_then(|v| v.as_bool()).unwrap());
        assert!(!dir_path.exists());
    }

    #[tokio::test]
    async fn test_delete_directory_recursive() {
        let temp_dir = tempdir().expect("Failed to create temp dir");
        let dir_path = temp_dir.path().join("test_dir");
        let nested_dir = dir_path.join("nested");
        let file_in_nested = nested_dir.join("file.txt");

        std::fs::create_dir_all(&nested_dir).expect("Failed to create nested directories");
        std::fs::write(&file_in_nested, "content").expect("Failed to create file in nested dir");

        assert!(dir_path.exists());
        assert!(file_in_nested.exists());

        let delete_file = DeleteFile::default();
        let args = json!({
            "path": dir_path.display().to_string(),
            "recursive": true
        });

        let result = delete_file
            .execute(args)
            .await
            .expect("Failed to delete directory recursively");
        assert!(result.get("success").and_then(|v| v.as_bool()).unwrap());
        assert!(!dir_path.exists());
    }

    #[tokio::test]
    async fn test_delete_nonexistent_file() {
        let temp_dir = tempdir().expect("Failed to create temp dir");
        let file_path = temp_dir.path().join("nonexistent.txt");

        let delete_file = DeleteFile::default();
        let args = json!({
            "path": file_path.display().to_string()
        });

        let result = delete_file.execute(args).await;
        assert!(result.is_err());
    }

    #[tokio::test]
    async fn test_delete_with_root_dir() {
        let temp_dir = tempdir().expect("Failed to create temp dir");
        let root_dir = temp_dir.path().to_str().unwrap().to_string();

        let file_path = temp_dir.path().join("test.txt");
        std::fs::write(&file_path, "content").expect("Failed to create test file");

        let delete_file = DeleteFile::new_with_root_dir(root_dir);
        let args = json!({
            "path": "test.txt"
        });

        let result = delete_file
            .execute(args)
            .await
            .expect("Failed to delete file");
        assert!(result.get("success").and_then(|v| v.as_bool()).unwrap());
        assert!(!file_path.exists());
    }
}
