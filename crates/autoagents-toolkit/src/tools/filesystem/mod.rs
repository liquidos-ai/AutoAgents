mod copy_file;
mod create_dir;
mod delete_file;
mod list_dir;
mod move_file;
mod read_file;
mod search_file;
mod write_file;

pub use copy_file::CopyFile;
pub use create_dir::CreateDir;
pub use delete_file::DeleteFile;
pub use list_dir::ListDir;
pub use move_file::MoveFile;
pub use read_file::ReadFile;
pub use search_file::SearchFile;
pub use write_file::WriteFile;

use std::path::{Path, PathBuf};

pub trait BaseFileTool {
    fn root_dir(&self) -> Option<String>;

    fn get_relative_path(&self, file_path: &str) -> PathBuf {
        match self.root_dir() {
            Some(root) => {
                let root_path = Path::new(&root);
                let file_path = Path::new(file_path);

                if file_path.is_absolute() {
                    file_path.to_path_buf()
                } else {
                    root_path.join(file_path)
                }
            }
            None => Path::new(file_path).to_path_buf(),
        }
    }

    fn ensure_within_root(&self, path: &Path) -> Result<PathBuf, std::io::Error> {
        fn normalize_path(path: &Path) -> PathBuf {
            use std::path::Component;

            let mut normalized = PathBuf::default();

            for component in path.components() {
                match component {
                    Component::CurDir => {}
                    Component::ParentDir => {
                        normalized.pop();
                    }
                    _ => normalized.push(component.as_os_str()),
                }
            }

            normalized
        }

        let canonical = path.canonicalize().unwrap_or_else(|_| normalize_path(path));

        if let Some(root) = self.root_dir() {
            let root_canonical = Path::new(&root).canonicalize()?;
            if !canonical.starts_with(&root_canonical) {
                return Err(std::io::Error::new(
                    std::io::ErrorKind::PermissionDenied,
                    format!("Path {} is outside of root directory", path.display()),
                ));
            }
        }

        Ok(canonical)
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    struct TestFileTool {
        root: Option<String>,
    }

    impl BaseFileTool for TestFileTool {
        fn root_dir(&self) -> Option<String> {
            self.root.clone()
        }
    }

    #[test]
    fn test_get_relative_path_no_root() {
        let tool = TestFileTool { root: None };
        let path = tool.get_relative_path("some/file.txt");
        assert_eq!(path, PathBuf::from("some/file.txt"));
    }

    #[test]
    fn test_get_relative_path_with_root_relative() {
        let tool = TestFileTool {
            root: Some("/home/user".to_string()),
        };
        let path = tool.get_relative_path("docs/file.txt");
        assert_eq!(path, PathBuf::from("/home/user/docs/file.txt"));
    }

    #[test]
    fn test_get_relative_path_with_root_absolute() {
        let tool = TestFileTool {
            root: Some("/home/user".to_string()),
        };
        let path = tool.get_relative_path("/absolute/path.txt");
        assert_eq!(path, PathBuf::from("/absolute/path.txt"));
    }

    #[test]
    fn test_ensure_within_root_safe_path() {
        let dir = std::env::temp_dir();
        let tool = TestFileTool {
            root: Some(dir.to_string_lossy().to_string()),
        };
        let safe = dir.join("test.txt");
        // Create file so canonicalize works
        std::fs::write(&safe, "").ok();
        let result = tool.ensure_within_root(&safe);
        assert!(result.is_ok());
        std::fs::remove_file(&safe).ok();
    }

    #[test]
    fn test_ensure_within_root_traversal_blocked() {
        let dir = std::env::temp_dir().join("test_root_autoagents");
        std::fs::create_dir_all(&dir).ok();
        let tool = TestFileTool {
            root: Some(dir.to_string_lossy().to_string()),
        };
        let traversal = dir.join("../../etc/passwd");
        let result = tool.ensure_within_root(&traversal);
        assert!(result.is_err());
        std::fs::remove_dir_all(&dir).ok();
    }

    #[test]
    fn test_ensure_within_root_no_root() {
        let tool = TestFileTool { root: None };
        let path = std::env::temp_dir().join("any_file.txt");
        std::fs::write(&path, "").ok();
        let result = tool.ensure_within_root(&path);
        assert!(result.is_ok());
        std::fs::remove_file(&path).ok();
    }
}
