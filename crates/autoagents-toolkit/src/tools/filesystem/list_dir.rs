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
pub struct ListDirArgs {
    #[input(description = "Path of the directory to list")]
    directory_path: String,
}

#[tool(
    name = "list_dir",
    description = "List contents of a directory",
    input = ListDirArgs,
)]
#[derive(Default)]
pub struct ListDir {
    root_dir: Option<String>,
}

impl ListDir {
    pub fn new() -> Self {
        Self::default()
    }

    pub fn new_with_root_dir(root_dir: String) -> Self {
        Self {
            root_dir: Some(root_dir),
        }
    }
}

impl BaseFileTool for ListDir {
    fn root_dir(&self) -> Option<String> {
        self.root_dir.clone()
    }
}

#[async_trait]
impl ToolRuntime for ListDir
where
    Self: BaseFileTool,
{
    async fn execute(&self, args: Value) -> Result<Value, ToolCallError> {
        let ListDirArgs { directory_path } = serde_json::from_value(args)?;

        debug!("List Directory Executing: Directory: {}", directory_path);

        let dir_path = self.get_relative_path(&directory_path);
        let _recursive = false; // Simplified to non-recursive
        let include_hidden = false;
        let filter_extension: Option<String> = None;

        // Validate directory exists
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

        // Ensure path is within root if root is set
        if self.root_dir().is_some() {
            self.ensure_within_root(&dir_path)
                .map_err(|e| ToolCallError::RuntimeError(Box::new(e)))?;
        }

        let mut entries = Vec::new();

        // Simplified to non-recursive listing only
        {
            let mut read_dir = fs::read_dir(&dir_path)
                .await
                .map_err(|e| ToolCallError::RuntimeError(Box::new(e)))?;

            while let Some(entry) = read_dir
                .next_entry()
                .await
                .map_err(|e| ToolCallError::RuntimeError(Box::new(e)))?
            {
                let path = entry.path();
                let file_name = entry.file_name().to_string_lossy().to_string();

                // Skip hidden files if requested
                if !include_hidden && file_name.starts_with('.') {
                    continue;
                }

                let metadata = entry
                    .metadata()
                    .await
                    .map_err(|e| ToolCallError::RuntimeError(Box::new(e)))?;

                let is_dir = metadata.is_dir();

                // Apply extension filter if provided
                if let Some(ref ext_filter) = filter_extension
                    && !is_dir
                {
                    if let Some(ext) = path.extension() {
                        if ext.to_string_lossy() != *ext_filter {
                            continue;
                        }
                    } else {
                        continue;
                    }
                }

                entries.push(json!({
                    "name": file_name,
                    "path": path.display().to_string(),
                    "is_dir": is_dir,
                    "size": if !is_dir { metadata.len() } else { 0 },
                }));
            }
        }

        Ok(json!({
            "success": true,
            "directory": dir_path.display().to_string(),
            "count": entries.len(),
            "entries": entries
        }))
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    // std::io::Write import removed - not used in simplified tests
    use tempfile::tempdir;

    #[tokio::test]
    async fn test_list_dir_simple() {
        let temp_dir = tempdir().expect("Failed to create temp dir");

        // Create some test files and directories
        std::fs::write(temp_dir.path().join("file1.txt"), "content1")
            .expect("Failed to create file1");
        std::fs::write(temp_dir.path().join("file2.rs"), "content2")
            .expect("Failed to create file2");
        std::fs::create_dir_all(temp_dir.path().join("subdir")).expect("Failed to create subdir");

        let list_dir = ListDir::default();
        let args = json!({
            "directory_path": temp_dir.path().display().to_string()
        });

        let result = list_dir
            .execute(args)
            .await
            .expect("Failed to list directory");
        assert!(result.get("success").and_then(|v| v.as_bool()).unwrap());

        let entries = result.get("entries").and_then(|v| v.as_array()).unwrap();
        assert_eq!(entries.len(), 3);
    }

    // Filter test removed - feature not implemented in simplified version

    // Recursive test removed - feature not implemented in simplified version

    // Hidden files test removed - feature not implemented in simplified version

    #[tokio::test]
    async fn test_list_nonexistent_dir() {
        let temp_dir = tempdir().expect("Failed to create temp dir");
        let nonexistent = temp_dir.path().join("nonexistent");

        let list_dir = ListDir::default();
        let args = json!({
            "directory_path": nonexistent.display().to_string()
        });

        let result = list_dir.execute(args).await;
        assert!(result.is_err());
    }

    #[tokio::test]
    async fn test_list_dir_with_root_dir() {
        let temp_dir = tempdir().expect("Failed to create temp dir");
        let root_dir = temp_dir.path().to_str().unwrap().to_string();

        std::fs::write(temp_dir.path().join("test.txt"), "content")
            .expect("Failed to create test file");

        let list_dir = ListDir::new_with_root_dir(root_dir);
        let args = json!({
            "directory_path": "."
        });

        let result = list_dir
            .execute(args)
            .await
            .expect("Failed to list directory");
        assert!(result.get("success").and_then(|v| v.as_bool()).unwrap());

        let entries = result.get("entries").and_then(|v| v.as_array()).unwrap();
        assert_eq!(entries.len(), 1);
    }
}
