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
pub struct MoveFileArgs {
    #[input(description = "Path of the file or directory to move")]
    source_path: String,
    #[input(description = "Destination path for the file or directory")]
    destination_path: String,
}

#[tool(
    name = "move_file",
    description = "Move or rename a file or directory",
    input = MoveFileArgs,
)]
pub struct MoveFile {
    sandbox: FilesystemSandbox,
}

impl MoveFile {
    pub fn new(root: impl AsRef<Path>) -> std::io::Result<Self> {
        Ok(Self {
            sandbox: FilesystemSandbox::new(root)?,
        })
    }

    pub fn with_sandbox(sandbox: FilesystemSandbox) -> Self {
        Self { sandbox }
    }
}

impl BaseFileTool for MoveFile {
    fn sandbox(&self) -> &FilesystemSandbox {
        &self.sandbox
    }
}

#[async_trait]
impl ToolRuntime for MoveFile
where
    Self: BaseFileTool,
{
    async fn execute(&self, args: Value) -> Result<Value, ToolCallError> {
        let MoveFileArgs {
            source_path,
            destination_path,
        } = serde_json::from_value(args)?;

        debug!(
            "Move File Executing: Source: {}, Destination: {}",
            source_path, destination_path
        );

        let src_path = self
            .sandbox()
            .resolve_relative(&source_path)
            .map_err(sandbox_error)?;
        let dest_path = self
            .sandbox()
            .resolve_relative(&destination_path)
            .map_err(sandbox_error)?;

        if !src_path.exists() {
            return Err(ToolCallError::RuntimeError(
                format!("Source path does not exist: {}", src_path.display()).into(),
            ));
        }

        let src_path = self
            .sandbox()
            .ensure_resolved(&src_path)
            .map_err(sandbox_error)?;

        if dest_path.exists() {
            return Err(ToolCallError::RuntimeError(
                format!("Destination already exists: {}", dest_path.display()).into(),
            ));
        }

        let dest_path = prepare_mutation_path(self.sandbox(), &destination_path).await?;

        let is_directory = src_path.is_dir();

        fs::rename(&src_path, &dest_path)
            .await
            .map_err(|e| ToolCallError::RuntimeError(Box::new(e)))?;

        Ok(json!({
            "success": true,
            "source": self.sandbox().relative_path_display(&src_path),
            "destination": self.sandbox().relative_path_display(&dest_path),
            "type": if is_directory { "directory" } else { "file" }
        }))
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::io::Write;
    use tempfile::tempdir;

    #[tokio::test]
    async fn test_move_file() {
        let temp_dir = tempdir().expect("Failed to create temp dir");
        let src_path = temp_dir.path().join("source.txt");
        let dest_path = temp_dir.path().join("destination.txt");

        let mut file = std::fs::File::create(&src_path).expect("Failed to create source file");
        file.write_all(b"Test content")
            .expect("Failed to write to source file");
        drop(file);

        assert!(src_path.exists());

        let move_file = MoveFile::new(temp_dir.path()).expect("sandbox");
        let args = json!({
            "source_path": "source.txt",
            "destination_path": "destination.txt"
        });

        let result = move_file.execute(args).await.expect("Failed to move file");
        assert!(result.get("success").and_then(|v| v.as_bool()).unwrap());

        assert!(!src_path.exists());
        assert!(dest_path.exists());

        let content = std::fs::read_to_string(&dest_path).expect("Failed to read destination file");
        assert_eq!(content, "Test content");
    }

    #[tokio::test]
    async fn test_rename_file() {
        let temp_dir = tempdir().expect("Failed to create temp dir");
        let old_name = temp_dir.path().join("old_name.txt");
        let new_name = temp_dir.path().join("new_name.txt");

        std::fs::write(&old_name, "Content").expect("Failed to create file");

        let move_file = MoveFile::new(temp_dir.path()).expect("sandbox");
        let args = json!({
            "source_path": "old_name.txt",
            "destination_path": "new_name.txt"
        });

        let result = move_file
            .execute(args)
            .await
            .expect("Failed to rename file");
        assert!(result.get("success").and_then(|v| v.as_bool()).unwrap());

        assert!(!old_name.exists());
        assert!(new_name.exists());
    }

    #[tokio::test]
    async fn test_move_directory() {
        let temp_dir = tempdir().expect("Failed to create temp dir");
        let src_dir = temp_dir.path().join("source_dir");
        let file_in_dir = src_dir.join("file.txt");
        let dest_dir = temp_dir.path().join("dest_dir");

        std::fs::create_dir_all(&src_dir).expect("Failed to create source directory");
        std::fs::write(&file_in_dir, "content").expect("Failed to create file in directory");

        let move_file = MoveFile::new(temp_dir.path()).expect("sandbox");
        let args = json!({
            "source_path": "source_dir",
            "destination_path": "dest_dir"
        });

        let result = move_file
            .execute(args)
            .await
            .expect("Failed to move directory");
        assert!(result.get("success").and_then(|v| v.as_bool()).unwrap());
        assert_eq!(
            result.get("type").and_then(|v| v.as_str()),
            Some("directory")
        );

        assert!(!src_dir.exists());
        assert!(dest_dir.exists());
        assert!(dest_dir.join("file.txt").exists());
    }

    #[tokio::test]
    async fn test_move_nonexistent_file() {
        let temp_dir = tempdir().expect("Failed to create temp dir");

        let move_file = MoveFile::new(temp_dir.path()).expect("sandbox");
        let args = json!({
            "source_path": "nonexistent.txt",
            "destination_path": "destination.txt"
        });

        let result = move_file.execute(args).await;
        assert!(result.is_err());
    }

    #[tokio::test]
    async fn test_move_to_existing_destination() {
        let temp_dir = tempdir().expect("Failed to create temp dir");
        let src_path = temp_dir.path().join("source.txt");
        let dest_path = temp_dir.path().join("destination.txt");

        std::fs::write(&src_path, "source").expect("Failed to create source file");
        std::fs::write(&dest_path, "destination").expect("Failed to create destination file");

        let move_file = MoveFile::new(temp_dir.path()).expect("sandbox");
        let args = json!({
            "source_path": "source.txt",
            "destination_path": "destination.txt"
        });

        let result = move_file.execute(args).await;
        assert!(result.is_err());
    }

    #[tokio::test]
    async fn test_move_with_root_dir() {
        let temp_dir = tempdir().expect("Failed to create temp dir");

        let src_file = temp_dir.path().join("source.txt");
        std::fs::write(&src_file, "content").expect("Failed to create source file");

        let move_file = MoveFile::new(temp_dir.path()).expect("sandbox");
        let args = json!({
            "source_path": "source.txt",
            "destination_path": "moved.txt"
        });

        let result = move_file.execute(args).await.expect("Failed to move file");
        assert!(result.get("success").and_then(|v| v.as_bool()).unwrap());

        assert!(!src_file.exists());
        assert!(temp_dir.path().join("moved.txt").exists());
    }
}
