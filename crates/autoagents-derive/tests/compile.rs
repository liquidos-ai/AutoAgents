//! Compile-time integration tests for proc-macro output across dependency layouts.

use std::path::{Path, PathBuf};
use std::process::Command;
use std::sync::Mutex;

static CARGO_CHECK_LOCK: Mutex<()> = Mutex::new(());

fn lock_cargo_check() -> std::sync::MutexGuard<'static, ()> {
    CARGO_CHECK_LOCK
        .lock()
        .unwrap_or_else(|poisoned| poisoned.into_inner())
}

fn manifest_path(name: &str) -> PathBuf {
    Path::new(env!("CARGO_MANIFEST_DIR"))
        .join("tests")
        .join(name)
        .join("Cargo.toml")
}

fn cargo_test_manifest(manifest: &Path) -> (bool, String) {
    let _guard = lock_cargo_check();
    let output = Command::new(env!("CARGO"))
        .current_dir(manifest.parent().expect("manifest parent"))
        .arg("test")
        .arg("--manifest-path")
        .arg(manifest)
        .output()
        .expect("failed to run cargo test");
    let stderr = String::from_utf8_lossy(&output.stderr);
    let stdout = String::from_utf8_lossy(&output.stdout);
    let combined = format!("{stdout}{stderr}");
    (output.status.success(), combined)
}

fn cargo_check_manifest(manifest: &Path) -> (bool, String) {
    let _guard = lock_cargo_check();
    let output = Command::new(env!("CARGO"))
        .current_dir(manifest.parent().expect("manifest parent"))
        .arg("check")
        .arg("--manifest-path")
        .arg(manifest)
        .output()
        .expect("failed to run cargo check");
    let stderr = String::from_utf8_lossy(&output.stderr);
    let stdout = String::from_utf8_lossy(&output.stdout);
    let combined = format!("{stdout}{stderr}");
    (output.status.success(), combined)
}

#[test]
fn compile_pass_facade() {
    let manifest = manifest_path("compile-pass-facade");
    let (success, output) = cargo_test_manifest(&manifest);
    assert!(
        success,
        "facade derive macro fixture should compile and pass runtime checks; output: {output}"
    );
}

#[test]
fn compile_pass_direct_core() {
    let manifest = manifest_path("compile-pass-direct-core");
    let (success, output) = cargo_test_manifest(&manifest);
    assert!(
        success,
        "direct autoagents-core derive macro fixture should compile and pass runtime checks; output: {output}"
    );
}

#[test]
fn compile_fail_missing_core_dependency() {
    let manifest = manifest_path("compile-fail-missing-core");
    let (success, output) = cargo_check_manifest(&manifest);
    assert!(
        !success,
        "fixture without autoagents/autoagents-core should fail to compile"
    );
    assert!(
        output.contains("autoagents-derive requires either `autoagents` or `autoagents-core`"),
        "unexpected compiler output: {output}"
    );
}

#[test]
fn compile_fail_manual_tool_input_without_derive() {
    let manifest = manifest_path("compile-fail-missing-tool-input-derive");
    let (success, output) = cargo_check_manifest(&manifest);
    assert!(
        !success,
        "manual ToolInputT without #[derive(ToolInput)] should fail to compile"
    );
    assert!(
        output.contains("ToolInputSchema"),
        "expected ToolInputSchema trait bound error, got: {output}"
    );
}

#[test]
fn compile_fail_missing_serde_json_dependency() {
    let manifest = manifest_path("compile-fail-missing-serde-json");
    let (success, output) = cargo_check_manifest(&manifest);
    assert!(
        !success,
        "fixture without direct serde_json should fail to compile; output: {output}"
    );
    assert!(
        output.contains("serde_json"),
        "expected serde_json resolution error, got: {output}"
    );
}
