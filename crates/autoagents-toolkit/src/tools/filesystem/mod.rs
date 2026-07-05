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

use std::path::{Component, Path, PathBuf};

use crate::utils::path_sandbox;

pub(crate) fn default_root_dir() -> String {
    std::env::current_dir()
        .unwrap_or_else(|_| PathBuf::from("."))
        .to_string_lossy()
        .into_owned()
}

pub trait BaseFileTool {
    fn root_dir(&self) -> Option<String>;

    fn resolve_path(&self, file_path: &str) -> Result<PathBuf, std::io::Error> {
        match self.root_dir() {
            Some(root) => {
                let root_path = Path::new(&root);
                let file_path = Path::new(file_path);
                path_sandbox::resolve_within_root(file_path, root_path)
            }
            None => Ok(Path::new(file_path).to_path_buf()),
        }
    }

    fn output_path(&self, path: &Path) -> String {
        let Some(root) = self.root_dir() else {
            return path.display().to_string();
        };

        let Ok(root_canonical) = Path::new(&root).canonicalize() else {
            return path.display().to_string();
        };

        match path.strip_prefix(&root_canonical) {
            Ok(relative) if relative.as_os_str().is_empty() => ".".to_string(),
            Ok(relative) => relative.display().to_string(),
            Err(_) => path.display().to_string(),
        }
    }

    fn get_relative_path(&self, file_path: &str) -> PathBuf {
        match self.root_dir() {
            Some(root) => {
                let root_path = Path::new(&root);
                let file_path = Path::new(file_path);
                let relative_path = file_path
                    .components()
                    .filter(|component| {
                        !matches!(component, Component::Prefix(_) | Component::RootDir)
                    })
                    .collect::<PathBuf>();
                root_path.join(relative_path)
            }
            None => Path::new(file_path).to_path_buf(),
        }
    }

    fn ensure_within_root(&self, path: &Path) -> Result<PathBuf, std::io::Error> {
        if let Some(root) = self.root_dir() {
            path_sandbox::ensure_within_root(path, Path::new(&root))
        } else {
            Ok(path_sandbox::canonicalize_or_normalize(path))
        }
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
        assert_eq!(path, PathBuf::from("/home/user/absolute/path.txt"));
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

    #[test]
    fn test_resolve_path_with_root_rejects_absolute_path() {
        let dir = std::env::temp_dir();
        let tool = TestFileTool {
            root: Some(dir.to_string_lossy().to_string()),
        };
        let result = tool.resolve_path("/etc/passwd");
        assert!(result.is_err());
    }

    #[test]
    fn test_resolve_path_no_root_preserves_parent_traversal_for_missing_target() {
        let tool = TestFileTool { root: None };
        let path = tool
            .resolve_path("../missing-file-autoagents.txt")
            .expect("unrestricted path should resolve");
        assert_eq!(path, PathBuf::from("../missing-file-autoagents.txt"));
    }

    #[test]
    fn test_output_path_with_root_returns_relative_path() {
        let dir = std::env::temp_dir().join("output_path_root");
        std::fs::create_dir_all(dir.join("nested")).expect("create temp dir");
        let tool = TestFileTool {
            root: Some(dir.to_string_lossy().to_string()),
        };

        let output = tool.output_path(&dir.canonicalize().expect("canonical root").join("nested"));
        assert_eq!(output, "nested");

        std::fs::remove_dir_all(dir).ok();
    }
}
