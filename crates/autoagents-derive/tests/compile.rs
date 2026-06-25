//! Compile-time integration tests for proc-macro output across dependency layouts.

use std::path::{Path, PathBuf};
use std::process::Command;
use std::sync::Mutex;

static CARGO_CHECK_LOCK: Mutex<()> = Mutex::new(());

fn workspace_root() -> &'static Path {
    Path::new(env!("CARGO_MANIFEST_DIR"))
        .parent()
        .and_then(|p| p.parent())
        .expect("workspace root")
}

fn manifest_path(name: &str) -> PathBuf {
    Path::new(env!("CARGO_MANIFEST_DIR"))
        .join("tests")
        .join(name)
        .join("Cargo.toml")
}

fn cargo_check_package(package: &str) -> bool {
    let _guard = CARGO_CHECK_LOCK.lock().expect("cargo check lock poisoned");
    Command::new(env!("CARGO"))
        .current_dir(workspace_root())
        .args(["check", "-p", package])
        .status()
        .expect("failed to run cargo check")
        .success()
}

fn cargo_check_manifest(manifest: &Path) -> (bool, String) {
    let _guard = CARGO_CHECK_LOCK.lock().expect("cargo check lock poisoned");
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
    assert!(
        cargo_check_package("derive-compile-pass-facade"),
        "facade derive macro fixture should compile"
    );
}

#[test]
fn compile_pass_direct_core() {
    assert!(
        cargo_check_package("derive-compile-pass-direct-core"),
        "direct autoagents-core derive macro fixture should compile"
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
        "fixture without direct serde_json should fail to compile"
    );
    assert!(
        output.contains("serde_json"),
        "expected serde_json resolution error, got: {output}"
    );
}
