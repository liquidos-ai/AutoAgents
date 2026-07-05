use std::path::{Component, Path, PathBuf};

/// Normalize a path by resolving `.` and `..` components without touching the filesystem.
pub fn normalize_path(path: &Path) -> PathBuf {
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

/// Canonicalize `path`, falling back to logical normalization when the target does not exist yet.
pub fn canonicalize_or_normalize(path: &Path) -> PathBuf {
    path.canonicalize().unwrap_or_else(|_| normalize_path(path))
}

fn permission_denied(path: &Path, root: &Path) -> std::io::Error {
    std::io::Error::new(
        std::io::ErrorKind::PermissionDenied,
        format!(
            "Path {} is outside of root directory {}",
            path.display(),
            root.display()
        ),
    )
}

fn deepest_existing_ancestor(path: &Path) -> Option<PathBuf> {
    let mut current = path;

    loop {
        if current.exists() {
            return Some(current.to_path_buf());
        }

        current = current.parent()?;
    }
}

/// Ensure `path` resolves inside `root`.
pub fn ensure_within_root(path: &Path, root: &Path) -> Result<PathBuf, std::io::Error> {
    let canonical = canonicalize_or_normalize(path);
    let root_canonical = root.canonicalize()?;

    if !canonical.starts_with(&root_canonical) {
        return Err(permission_denied(path, root));
    }

    Ok(canonical)
}

/// Resolve a user-provided relative path inside `root`.
///
/// Existing targets are canonicalized directly. For paths that do not exist yet,
/// the deepest existing ancestor is canonicalized so symlinked parent escapes are
/// rejected before write/create operations run.
pub fn resolve_within_root(path: &Path, root: &Path) -> Result<PathBuf, std::io::Error> {
    if path.is_absolute() {
        return Err(std::io::Error::new(
            std::io::ErrorKind::PermissionDenied,
            format!(
                "Absolute paths are not allowed inside root-scoped tools: {}",
                path.display()
            ),
        ));
    }

    let root_canonical = root.canonicalize()?;
    let joined = root_canonical.join(path);
    let normalized = normalize_path(&joined);

    if !normalized.starts_with(&root_canonical) {
        return Err(permission_denied(&joined, root));
    }

    let existing_ancestor = deepest_existing_ancestor(&joined).ok_or_else(|| {
        std::io::Error::new(
            std::io::ErrorKind::NotFound,
            format!("No existing ancestor found for {}", joined.display()),
        )
    })?;
    let ancestor_canonical = existing_ancestor.canonicalize()?;

    if !ancestor_canonical.starts_with(&root_canonical) {
        return Err(permission_denied(&joined, root));
    }

    Ok(normalized)
}

/// Ensure `path` resolves inside at least one of the provided roots.
#[cfg(feature = "document-parsing")]
pub fn ensure_within_any_root(path: &Path, roots: &[PathBuf]) -> Result<PathBuf, std::io::Error> {
    if roots.is_empty() {
        return Err(std::io::Error::new(
            std::io::ErrorKind::InvalidInput,
            "no allowed roots configured",
        ));
    }

    let mut last_error = None;

    for root in roots {
        match ensure_within_root(path, root) {
            Ok(canonical) => return Ok(canonical),
            Err(error) => last_error = Some(error),
        }
    }

    Err(last_error.unwrap_or_else(|| {
        std::io::Error::new(
            std::io::ErrorKind::PermissionDenied,
            format!("Path {} is outside of allowed roots", path.display()),
        )
    }))
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn normalize_path_resolves_parent_dirs() {
        let path = Path::new("workspace/a/../b/./file.txt");
        assert_eq!(normalize_path(path), PathBuf::from("workspace/b/file.txt"));
    }

    #[test]
    fn ensure_within_root_safe_path() {
        let dir = std::env::temp_dir();
        let safe = dir.join("path_sandbox_safe.txt");
        std::fs::write(&safe, "").expect("write temp file");

        let result = ensure_within_root(&safe, &dir);
        assert!(result.is_ok());

        std::fs::remove_file(safe).ok();
    }

    #[test]
    fn ensure_within_root_traversal_blocked() {
        let dir = std::env::temp_dir().join("path_sandbox_root");
        std::fs::create_dir_all(&dir).expect("create temp dir");

        let traversal = dir.join("../../etc/passwd");
        let result = ensure_within_root(&traversal, &dir);
        assert!(result.is_err());

        std::fs::remove_dir_all(dir).ok();
    }

    #[test]
    fn resolve_within_root_rejects_absolute_path() {
        let dir = std::env::temp_dir();
        let result = resolve_within_root(Path::new("/etc/passwd"), &dir);
        assert!(result.is_err());
    }

    #[test]
    fn resolve_within_root_rejects_traversal_for_missing_target() {
        let dir = std::env::temp_dir().join("path_sandbox_missing_target");
        std::fs::create_dir_all(&dir).expect("create temp dir");

        let result = resolve_within_root(Path::new("../outside.txt"), &dir);
        assert!(result.is_err());

        std::fs::remove_dir_all(dir).ok();
    }

    #[test]
    fn resolve_within_root_allows_missing_child_path() {
        let dir = std::env::temp_dir().join("path_sandbox_missing_child");
        std::fs::create_dir_all(&dir).expect("create temp dir");

        let result = resolve_within_root(Path::new("nested/file.txt"), &dir)
            .expect("missing child should resolve under root");
        assert!(result.starts_with(dir.canonicalize().expect("canonical root")));

        std::fs::remove_dir_all(dir).ok();
    }

    #[test]
    #[cfg(unix)]
    fn resolve_within_root_rejects_symlink_parent_escape() {
        let root = std::env::temp_dir().join("path_sandbox_symlink_root");
        let outside = std::env::temp_dir().join("path_sandbox_symlink_outside");
        std::fs::create_dir_all(&root).expect("create root");
        std::fs::create_dir_all(&outside).expect("create outside");
        std::os::unix::fs::symlink(&outside, root.join("outside_link")).expect("create symlink");

        let result = resolve_within_root(Path::new("outside_link/new.txt"), &root);
        assert!(result.is_err());

        std::fs::remove_dir_all(root).ok();
        std::fs::remove_dir_all(outside).ok();
    }

    #[test]
    #[cfg(feature = "document-parsing")]
    fn ensure_within_any_root_accepts_matching_root() {
        let dir = std::env::temp_dir().join("path_sandbox_any_root");
        std::fs::create_dir_all(&dir).expect("create temp dir");

        let file = dir.join("doc.txt");
        std::fs::write(&file, "ok").expect("write temp file");

        let roots = vec![std::env::temp_dir(), dir.clone()];
        let result = ensure_within_any_root(&file, &roots);
        assert!(result.is_ok());

        std::fs::remove_file(file).ok();
        std::fs::remove_dir_all(dir).ok();
    }
}
