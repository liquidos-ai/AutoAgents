//! Canonical filesystem sandbox for toolkit file tools.
//!
//! Symlink-escape regression tests run on Unix platforms. On Windows, junctions and
//! directory symlinks are validated when present, but CI does not yet include
//! Windows-specific escape fixtures.

use std::io;
use std::path::{Component, Path, PathBuf};

use walkdir::WalkDir;

/// Canonical filesystem root for sandboxed file tool operations.
#[derive(Debug, Clone)]
pub struct FilesystemSandbox {
    root: PathBuf,
}

impl FilesystemSandbox {
    /// Create a sandbox rooted at `root`. The path must exist and be a directory.
    pub fn new(root: impl AsRef<Path>) -> io::Result<Self> {
        let root_ref = root.as_ref();
        if !root_ref.exists() {
            return Err(io::Error::new(
                io::ErrorKind::NotFound,
                format!("Sandbox root does not exist: {}", root_ref.display()),
            ));
        }
        if !root_ref.is_dir() {
            return Err(io::Error::new(
                io::ErrorKind::InvalidInput,
                format!("Sandbox root is not a directory: {}", root_ref.display()),
            ));
        }

        let canonical = root_ref.canonicalize()?;
        Ok(Self { root: canonical })
    }

    /// Canonical sandbox root path.
    pub fn root(&self) -> &Path {
        &self.root
    }

    /// Format a resolved path as a sandbox-relative path for tool output.
    pub fn relative_path_display(&self, path: &Path) -> String {
        path.strip_prefix(&self.root)
            .map(|relative| {
                if relative.as_os_str().is_empty() {
                    ".".to_string()
                } else {
                    relative.display().to_string()
                }
            })
            .unwrap_or_else(|_| path.display().to_string())
    }

    /// Resolve a user-supplied relative path under the sandbox root.
    pub fn resolve_relative(&self, user_path: &str) -> io::Result<PathBuf> {
        let user_path = user_path.trim();
        if user_path.is_empty() {
            return Err(io::Error::new(
                io::ErrorKind::InvalidInput,
                "Path must not be empty",
            ));
        }

        let user = Path::new(user_path);
        if user.is_absolute() {
            return Err(io::Error::new(
                io::ErrorKind::PermissionDenied,
                format!("Absolute paths are not allowed: {user_path}"),
            ));
        }

        if user
            .components()
            .any(|component| matches!(component, Component::ParentDir))
        {
            return Err(io::Error::new(
                io::ErrorKind::PermissionDenied,
                format!("Path traversal is not allowed: {user_path}"),
            ));
        }

        let joined = normalize_joined_path(&self.root.join(user))?;
        self.verify_within_root(&joined)?;
        Ok(joined)
    }

    /// Verify that `path` resolves within the sandbox, returning the canonical or logical path.
    pub fn ensure_resolved(&self, path: &Path) -> io::Result<PathBuf> {
        if path.exists() {
            let canonical = path.canonicalize()?;
            self.verify_within_root(&canonical)?;
            return Ok(canonical);
        }

        let mut suffix = PathBuf::default();
        let mut ancestor = path.to_path_buf();

        while !ancestor.exists() {
            match ancestor.file_name() {
                Some(name) => {
                    suffix = PathBuf::from(name).join(suffix);
                }
                None => {
                    return Err(io::Error::new(
                        io::ErrorKind::PermissionDenied,
                        format!("Unable to resolve path within sandbox: {}", path.display()),
                    ));
                }
            }
            if !ancestor.pop() {
                return Err(io::Error::new(
                    io::ErrorKind::PermissionDenied,
                    format!("Unable to resolve path within sandbox: {}", path.display()),
                ));
            }
        }

        let canonical_ancestor = ancestor.canonicalize()?;
        self.verify_within_root(&canonical_ancestor)?;

        let mut resolved = canonical_ancestor;
        for component in suffix.components() {
            match component {
                Component::Normal(name) => resolved.push(name),
                Component::CurDir => {}
                Component::ParentDir => {
                    return Err(io::Error::new(
                        io::ErrorKind::PermissionDenied,
                        format!("Path traversal is not allowed: {}", path.display()),
                    ));
                }
                _ => {
                    return Err(io::Error::new(
                        io::ErrorKind::PermissionDenied,
                        format!("Invalid path component in {}", path.display()),
                    ));
                }
            }
        }

        self.verify_within_root(&resolved)?;
        Ok(resolved)
    }

    /// Validate a directory-walk entry stays within the sandbox.
    pub fn validate_walk_entry(&self, entry_path: &Path) -> io::Result<PathBuf> {
        self.ensure_resolved(entry_path)
    }

    /// Create a `WalkDir` iterator that does not follow symlinks.
    pub fn walk_dir(&self, dir: &Path) -> WalkDir {
        WalkDir::new(dir).follow_links(false)
    }

    fn verify_within_root(&self, path: &Path) -> io::Result<()> {
        if !path.starts_with(&self.root) {
            return Err(io::Error::new(
                io::ErrorKind::PermissionDenied,
                format!(
                    "Path {} is outside of sandbox root {}",
                    path.display(),
                    self.root.display()
                ),
            ));
        }
        Ok(())
    }
}

fn normalize_joined_path(path: &Path) -> io::Result<PathBuf> {
    let mut normalized = PathBuf::default();

    for component in path.components() {
        match component {
            Component::CurDir => {}
            Component::ParentDir => {
                return Err(io::Error::new(
                    io::ErrorKind::PermissionDenied,
                    "Path traversal is not allowed",
                ));
            }
            Component::RootDir | Component::Prefix(_) => normalized.push(component.as_os_str()),
            Component::Normal(name) => normalized.push(name),
        }
    }

    Ok(normalized)
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::fs;
    use tempfile::tempdir;

    #[test]
    fn rejects_absolute_paths() {
        let tmp = tempdir().expect("tempdir");
        let sandbox = FilesystemSandbox::new(tmp.path()).expect("sandbox");

        let err = sandbox
            .resolve_relative("/etc/passwd")
            .expect_err("absolute path");
        assert_eq!(err.kind(), io::ErrorKind::PermissionDenied);
    }

    #[test]
    fn rejects_whitespace_only_paths() {
        let tmp = tempdir().expect("tempdir");
        let sandbox = FilesystemSandbox::new(tmp.path()).expect("sandbox");

        let err = sandbox.resolve_relative("   ").expect_err("whitespace");
        assert_eq!(err.kind(), io::ErrorKind::InvalidInput);
    }

    #[test]
    fn rejects_parent_dir_traversal() {
        let tmp = tempdir().expect("tempdir");
        let sandbox = FilesystemSandbox::new(tmp.path()).expect("sandbox");

        let err = sandbox
            .resolve_relative("../outside")
            .expect_err("traversal");
        assert_eq!(err.kind(), io::ErrorKind::PermissionDenied);
    }

    #[test]
    fn resolves_relative_path_under_root() {
        let tmp = tempdir().expect("tempdir");
        let sandbox = FilesystemSandbox::new(tmp.path()).expect("sandbox");

        let resolved = sandbox.resolve_relative("src/main.rs").expect("resolve");
        assert_eq!(resolved, tmp.path().join("src/main.rs"));
    }

    #[test]
    fn ensure_resolved_existing_file() {
        let tmp = tempdir().expect("tempdir");
        let file = tmp.path().join("test.txt");
        fs::write(&file, "data").expect("write");

        let sandbox = FilesystemSandbox::new(tmp.path()).expect("sandbox");
        let resolved = sandbox.ensure_resolved(&file).expect("ensure");
        assert!(resolved.starts_with(sandbox.root()));
    }

    #[test]
    fn ensure_resolved_nonexistent_path_within_root() {
        let tmp = tempdir().expect("tempdir");
        fs::create_dir_all(tmp.path().join("nested")).expect("mkdir");

        let sandbox = FilesystemSandbox::new(tmp.path()).expect("sandbox");
        let target = tmp.path().join("nested/new.txt");
        let resolved = sandbox.ensure_resolved(&target).expect("ensure");
        assert!(resolved.starts_with(sandbox.root()));
    }

    #[test]
    fn ensure_resolved_rejects_outside_root() {
        let tmp = tempdir().expect("tempdir");
        let outside =
            std::env::temp_dir().join(format!("autoagents-outside-{}", std::process::id()));
        fs::write(&outside, "secret").expect("write");

        let sandbox = FilesystemSandbox::new(tmp.path()).expect("sandbox");
        let err = sandbox.ensure_resolved(&outside).expect_err("outside root");
        assert_eq!(err.kind(), io::ErrorKind::PermissionDenied);

        let _ = fs::remove_file(outside);
    }

    #[cfg(unix)]
    #[test]
    fn ensure_resolved_rejects_symlink_escape() {
        use std::os::unix::fs::symlink;

        let tmp = tempdir().expect("tempdir");
        let outside =
            std::env::temp_dir().join(format!("autoagents-outside-symlink-{}", std::process::id()));
        fs::write(&outside, "secret").expect("write");

        let link = tmp.path().join("escape.txt");
        symlink(&outside, &link).expect("symlink");

        let sandbox = FilesystemSandbox::new(tmp.path()).expect("sandbox");
        let err = sandbox.ensure_resolved(&link).expect_err("symlink escape");
        assert_eq!(err.kind(), io::ErrorKind::PermissionDenied);

        let _ = fs::remove_file(outside);
    }

    #[test]
    fn relative_path_display_strips_sandbox_root() {
        let tmp = tempdir().expect("tempdir");
        let nested = tmp.path().join("nested/file.txt");
        fs::create_dir_all(nested.parent().unwrap()).expect("mkdir");
        fs::write(&nested, "data").expect("write");

        let sandbox = FilesystemSandbox::new(tmp.path()).expect("sandbox");
        let resolved = sandbox.ensure_resolved(&nested).expect("resolve");
        assert_eq!(sandbox.relative_path_display(&resolved), "nested/file.txt");
        assert_eq!(sandbox.relative_path_display(sandbox.root()), ".");
    }

    #[test]
    fn prefix_boundary_sibling_root_rejected() {
        let parent = tempdir().expect("tempdir");
        let root = parent.path().join("workspace");
        let sibling = parent.path().join("workspace_extra");
        fs::create_dir_all(&root).expect("mkdir root");
        fs::create_dir_all(&sibling).expect("mkdir sibling");
        let outside_file = sibling.join("secret.txt");
        fs::write(&outside_file, "secret").expect("write");

        let sandbox = FilesystemSandbox::new(&root).expect("sandbox");
        let err = sandbox
            .ensure_resolved(&outside_file)
            .expect_err("sibling root");
        assert_eq!(err.kind(), io::ErrorKind::PermissionDenied);
    }
}
