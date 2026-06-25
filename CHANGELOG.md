# Changelog

All notable changes to this project will be documented in this file.

## [0.4.0] - Unreleased

### Security

- Filesystem toolkit tools now require an explicit workspace root via `Tool::new(&root)?`.
- Removed unscoped filesystem constructors (`ReadFile::new()`, `Default`, `new_with_root_dir`).
- Reject absolute paths, `..` traversal, and symlink escapes within the sandbox.
- `delete_file` deletes directories non-recursively by default; pass `"recursive": true` for trees.
- Mutating tools re-validate paths after parent directory creation to close TOCTOU gaps.

### Migration

| Before (pre-0.4.0) | After (0.4.0) |
| ------------------ | ------------- |
| `ReadFile::new()` | `ReadFile::new(&workspace_root)?` |
| `ReadFile::new_with_root_dir(path)` | `ReadFile::new(&path)?` |
| `SearchFile::new(100)` | `SearchFile::new(&workspace_root, 100)?` |
| `DeleteFile` on non-empty directory | Add `"recursive": true` to args |

Share one `FilesystemSandbox` across tools when wiring agents dynamically:

```rust
let sandbox = FilesystemSandbox::new(&workspace)?;
let read = ReadFile::with_sandbox(sandbox.clone());
let write = WriteFile::with_sandbox(sandbox);
```

See `examples/coding_agent` for the reference integration.
