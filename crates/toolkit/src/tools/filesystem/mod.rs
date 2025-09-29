mod copy_file;
mod delete_file;
mod list_dir;
mod move_file;
mod read_file;
mod search_file;
mod write_file;

pub use copy_file::CopyFile;
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
        let canonical = path.canonicalize().unwrap_or_else(|_| path.to_path_buf());

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
