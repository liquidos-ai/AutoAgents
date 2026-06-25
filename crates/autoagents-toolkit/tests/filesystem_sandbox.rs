use autoagents::core::tool::{ToolCallError, ToolRuntime};
use autoagents_toolkit::tools::filesystem::{
    CopyFile, CreateDir, DeleteFile, FilesystemSandbox, ListDir, MoveFile, ReadFile, SearchFile,
    WriteFile,
};
use serde_json::json;
use std::fs;
use std::io::ErrorKind;
use std::path::PathBuf;
use tempfile::tempdir;

fn assert_tool_runtime_error(result: Result<serde_json::Value, ToolCallError>) {
    match result {
        Err(ToolCallError::RuntimeError(_)) => {}
        other => panic!("expected runtime error, got: {other:?}"),
    }
}

fn outside_temp_file(label: &str) -> PathBuf {
    let path = std::env::temp_dir().join(format!(
        "autoagents-fs-sandbox-{label}-{}-{}",
        std::process::id(),
        std::time::SystemTime::now()
            .duration_since(std::time::UNIX_EPOCH)
            .expect("clock")
            .as_nanos()
    ));
    fs::write(&path, "outside-secret").expect("write outside file");
    path
}

#[tokio::test]
async fn all_tools_reject_parent_dir_traversal() {
    let root = tempdir().expect("tempdir");
    let tools: Vec<(&str, serde_json::Value)> = vec![
        ("read", json!({ "file_path": "../outside.txt" })),
        (
            "write",
            json!({ "file_path": "../outside.txt", "content": "x", "append": false }),
        ),
        ("delete", json!({ "path": "../outside.txt" })),
        ("list", json!({ "directory_path": "../outside" })),
        (
            "search",
            json!({ "directory": "../outside", "pattern": "*" }),
        ),
        (
            "copy",
            json!({ "source_path": "../outside.txt", "destination_path": "dest.txt" }),
        ),
        (
            "move",
            json!({ "source_path": "../outside.txt", "destination_path": "dest.txt" }),
        ),
        (
            "create_dir",
            json!({ "directory_path": "../outside", "recursive": true }),
        ),
    ];

    let read = ReadFile::new(root.path()).expect("read");
    let write = WriteFile::new(root.path()).expect("write");
    let delete = DeleteFile::new(root.path()).expect("delete");
    let list = ListDir::new(root.path()).expect("list");
    let search = SearchFile::new(root.path(), 10).expect("search");
    let copy = CopyFile::new(root.path()).expect("copy");
    let mov = MoveFile::new(root.path()).expect("move");
    let create = CreateDir::new(root.path()).expect("create");

    for (name, args) in tools {
        let err = match name {
            "read" => read.execute(args).await,
            "write" => write.execute(args).await,
            "delete" => delete.execute(args).await,
            "list" => list.execute(args).await,
            "search" => search.execute(args).await,
            "copy" => copy.execute(args).await,
            "move" => mov.execute(args).await,
            "create_dir" => create.execute(args).await,
            _ => unreachable!(),
        };
        assert!(err.is_err(), "{name} should reject traversal");
    }
}

#[tokio::test]
async fn all_tools_reject_absolute_paths() {
    let root = tempdir().expect("tempdir");
    let outside = outside_temp_file("absolute");

    let read = ReadFile::new(root.path()).expect("read");
    let write = WriteFile::new(root.path()).expect("write");
    let delete = DeleteFile::new(root.path()).expect("delete");
    let list = ListDir::new(root.path()).expect("list");
    let search = SearchFile::new(root.path(), 10).expect("search");
    let copy = CopyFile::new(root.path()).expect("copy");
    let mov = MoveFile::new(root.path()).expect("move");
    let create = CreateDir::new(root.path()).expect("create");

    let abs = outside.to_string_lossy().to_string();

    assert!(read.execute(json!({ "file_path": abs })).await.is_err());
    assert!(
        write
            .execute(json!({ "file_path": abs, "content": "x", "append": false }))
            .await
            .is_err()
    );
    assert!(delete.execute(json!({ "path": abs })).await.is_err());
    assert!(
        list.execute(json!({ "directory_path": abs }))
            .await
            .is_err()
    );
    assert!(
        search
            .execute(json!({ "directory": abs, "pattern": "*" }))
            .await
            .is_err()
    );
    assert!(
        copy.execute(json!({
            "source_path": abs,
            "destination_path": "dest.txt"
        }))
        .await
        .is_err()
    );
    assert!(
        mov.execute(json!({
            "source_path": abs,
            "destination_path": "dest.txt"
        }))
        .await
        .is_err()
    );
    assert!(
        create
            .execute(json!({ "directory_path": abs, "recursive": true }))
            .await
            .is_err()
    );

    let _ = fs::remove_file(outside);
}

#[tokio::test]
async fn write_to_new_path_inside_root_succeeds() {
    let root = tempdir().expect("tempdir");
    let write = WriteFile::new(root.path()).expect("write");

    let result = write
        .execute(json!({
            "file_path": "nested/new.txt",
            "content": "hello",
            "append": false
        }))
        .await
        .expect("write should succeed");

    assert_eq!(result.get("success").and_then(|v| v.as_bool()), Some(true));
    let written = root.path().join("nested/new.txt");
    assert_eq!(fs::read_to_string(written).expect("read"), "hello");
}

#[tokio::test]
async fn delete_non_recursive_non_empty_directory_fails() {
    let root = tempdir().expect("tempdir");
    fs::create_dir_all(root.path().join("dir/nested")).expect("mkdir");

    let delete = DeleteFile::new(root.path()).expect("delete");
    let err = delete
        .execute(json!({ "path": "dir" }))
        .await
        .expect_err("non-recursive delete should fail");
    assert_tool_runtime_error(Err(err));
    assert!(root.path().join("dir").exists());
}

#[tokio::test]
async fn delete_recursive_opt_in_removes_nested_directory() {
    let root = tempdir().expect("tempdir");
    fs::create_dir_all(root.path().join("dir/nested")).expect("mkdir");
    fs::write(root.path().join("dir/nested/file.txt"), "data").expect("write");

    let delete = DeleteFile::new(root.path()).expect("delete");
    delete
        .execute(json!({ "path": "dir", "recursive": true }))
        .await
        .expect("recursive delete");
    assert!(!root.path().join("dir").exists());
}

#[cfg(unix)]
#[tokio::test]
async fn read_rejects_symlink_escape() {
    use std::os::unix::fs::symlink;

    let root = tempdir().expect("tempdir");
    let outside = outside_temp_file("symlink-read");
    let link = root.path().join("escape.txt");
    symlink(&outside, &link).expect("symlink");

    let read = ReadFile::new(root.path()).expect("read");
    let err = read
        .execute(json!({ "file_path": "escape.txt" }))
        .await
        .expect_err("symlink escape");
    assert_tool_runtime_error(Err(err));

    let _ = fs::remove_file(outside);
}

#[cfg(unix)]
#[tokio::test]
async fn write_rejects_symlink_escape_via_parent() {
    use std::os::unix::fs::symlink;

    let root = tempdir().expect("tempdir");
    let outside_dir = std::env::temp_dir().join(format!(
        "autoagents-outside-dir-{}-{}",
        std::process::id(),
        std::time::SystemTime::now()
            .duration_since(std::time::UNIX_EPOCH)
            .expect("clock")
            .as_nanos()
    ));
    fs::create_dir_all(&outside_dir).expect("outside dir");
    let link_parent = root.path().join("linked");
    symlink(&outside_dir, &link_parent).expect("symlink dir");

    let write = WriteFile::new(root.path()).expect("write");
    let err = write
        .execute(json!({
            "file_path": "linked/secret.txt",
            "content": "escaped",
            "append": false
        }))
        .await
        .expect_err("write through symlink parent");

    assert_tool_runtime_error(Err(err));
    assert!(!outside_dir.join("secret.txt").exists());

    let _ = fs::remove_dir_all(outside_dir);
}

#[test]
fn sandbox_prefix_boundary_rejects_sibling_root() {
    let parent = tempdir().expect("tempdir");
    let root = parent.path().join("workspace");
    let sibling = parent.path().join("workspace_extra");
    fs::create_dir_all(&root).expect("root");
    fs::create_dir_all(&sibling).expect("sibling");
    let secret = sibling.join("secret.txt");
    fs::write(&secret, "secret").expect("write");

    let sandbox = FilesystemSandbox::new(&root).expect("sandbox");
    let err = sandbox
        .ensure_resolved(&secret)
        .expect_err("sibling escape");
    assert_eq!(err.kind(), ErrorKind::PermissionDenied);
}

#[cfg(unix)]
#[tokio::test]
async fn search_fails_fast_on_symlink_escape_entry() {
    use std::os::unix::fs::symlink;

    let root = tempdir().expect("tempdir");
    fs::write(root.path().join("allowed.txt"), "ok").expect("write");
    let outside = outside_temp_file("search-symlink");
    let link = root.path().join("escape_link.txt");
    symlink(&outside, &link).expect("symlink");

    let search = SearchFile::new(root.path(), 100).expect("search");
    let err = search
        .execute(json!({ "directory": ".", "pattern": "*" }))
        .await
        .expect_err("search should fail on symlink escape");

    assert_tool_runtime_error(Err(err));
    let _ = fs::remove_file(outside);
}

#[cfg(unix)]
#[tokio::test]
async fn delete_rejects_symlink_escape() {
    use std::os::unix::fs::symlink;

    let root = tempdir().expect("tempdir");
    let outside = outside_temp_file("delete-symlink");
    let link = root.path().join("escape.txt");
    symlink(&outside, &link).expect("symlink");

    let delete = DeleteFile::new(root.path()).expect("delete");
    let err = delete
        .execute(json!({ "path": "escape.txt" }))
        .await
        .expect_err("delete should reject symlink escape");

    assert_tool_runtime_error(Err(err));
    assert!(outside.exists());

    let _ = fs::remove_file(outside);
}

#[test]
fn resolve_relative_rejects_whitespace_only_paths() {
    let root = tempdir().expect("tempdir");
    let sandbox = FilesystemSandbox::new(root.path()).expect("sandbox");
    let err = sandbox
        .resolve_relative("   ")
        .expect_err("whitespace path");
    assert_eq!(err.kind(), ErrorKind::InvalidInput);
}
