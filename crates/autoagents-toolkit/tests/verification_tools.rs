#![cfg(all(not(target_arch = "wasm32"), feature = "verification"))]

use std::path::Path;
use std::process::Command;

use autoagents::core::tool::ToolRuntime;
use autoagents_toolkit::tools::verification::{CheckVerified, RunTests};
use serde_json::json;

fn git(dir: &Path, args: &[&str]) {
    let status = Command::new("git")
        .arg("-C")
        .arg(dir)
        .args(args)
        .status()
        .expect("git runs");
    assert!(status.success(), "git {args:?} failed");
}

fn write(dir: &Path, rel: &str, body: &str) {
    let path = dir.join(rel);
    std::fs::create_dir_all(path.parent().expect("parent")).expect("mkdir");
    std::fs::write(path, body).expect("write");
}

#[tokio::test]
async fn run_tests_and_check_verified_follow_the_tree() {
    let home = tempfile::tempdir().expect("tempdir");
    let repo = tempfile::tempdir().expect("tempdir");
    // Receipts and the signing key stay in the temp dir; nothing process-wide is changed.
    let context = treeproof::Context::new()
        .with_store_dir(home.path().join("cache"))
        .with_key_file(home.path().join("key"));
    let target_dir = home.path().join("target").display().to_string();
    let root = repo.path();
    git(root, &["init", "-q"]);
    write(
        root,
        "Cargo.toml",
        "[package]\nname = \"demo\"\nversion = \"0.1.0\"\nedition = \"2021\"\n\n[workspace]\n",
    );
    write(
        root,
        "src/lib.rs",
        "pub fn add(a: i32, b: i32) -> i32 { a + b }\n\n#[cfg(test)]\nmod tests {\n    #[test]\n    fn adds() { assert_eq!(super::add(1, 2), 3); }\n}\n",
    );
    write(root, ".gitignore", "target/\n");
    Command::new("cargo")
        .arg("generate-lockfile")
        .current_dir(root)
        .status()
        .expect("cargo runs");

    let check = CheckVerified::new(root).with_context(context.clone());
    let before = check.execute(json!({})).await.expect("check runs");
    assert_eq!(before["verified"].as_bool(), Some(false));

    let detected = RunTests::detect(root).expect("cargo project detected");
    assert_eq!(detected.command(), ["cargo", "test"]);
    // The same command with a private target directory, so the nested build does not wait
    // on the lock of the target directory this test binary was built in.
    let run = RunTests::new(
        root,
        vec![
            "cargo".into(),
            "test".into(),
            "--target-dir".into(),
            target_dir.clone(),
        ],
    )
    .with_context(context.clone());
    let first = run.execute(json!({})).await.expect("tests run");
    assert_eq!(first["result"], "PASSED", "{first}");
    assert_eq!(first["status"], "fresh");
    assert_eq!(first["passed"], 1);

    let after = check.execute(json!({})).await.expect("check runs");
    assert_eq!(after["verified"].as_bool(), Some(true), "{after}");

    // Same tree again: served from the cache.
    let second = run.execute(json!({})).await.expect("tests run");
    assert_eq!(second["status"], "cache_hit");

    // An edit makes the tree unverified until the tests run again, and a failure is reported.
    write(
        root,
        "src/lib.rs",
        "pub fn add(a: i32, b: i32) -> i32 { a - b }\n\n#[cfg(test)]\nmod tests {\n    #[test]\n    fn adds() { assert_eq!(super::add(1, 2), 3); }\n}\n",
    );
    let stale = check.execute(json!({})).await.expect("check runs");
    assert_eq!(stale["verified"].as_bool(), Some(false));
    let failed = run.execute(json!({})).await.expect("tests run");
    assert_eq!(failed["result"], "FAILED", "{failed}");
    assert_eq!(failed["failed"], 1);
    assert_eq!(failed["success"].as_bool(), Some(false));
    let still = check.execute(json!({})).await.expect("check runs");
    assert_eq!(still["verified"].as_bool(), Some(false));

    // A filter that matches nothing runs 0 tests and is FAILED, not a pass.
    let nothing = RunTests::new(
        root,
        vec![
            "cargo".into(),
            "test".into(),
            "--target-dir".into(),
            target_dir,
            "no_such_test".into(),
        ],
    )
    .with_context(context);
    let empty = nothing.execute(json!({})).await.expect("tests run");
    assert_eq!(empty["result"], "FAILED", "{empty}");
}
