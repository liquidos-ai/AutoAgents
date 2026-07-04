//! Compile-time integration tests for proc-macro output across dependency layouts.
//!
//! Keep these as isolated crates because `autoagents-derive` uses
//! `proc_macro_crate` to inspect the consuming crate's direct dependencies.

use std::path::{Path, PathBuf};
use std::process::Command;

#[derive(Clone, Copy)]
enum CargoCommand {
    Check,
    Test,
}

impl CargoCommand {
    fn as_str(self) -> &'static str {
        match self {
            Self::Check => "check",
            Self::Test => "test",
        }
    }
}

struct CompileFixture {
    name: &'static str,
    command: CargoCommand,
    expected_success: bool,
    expected_output: Option<&'static str>,
}

fn fixture_dir(name: &str) -> PathBuf {
    Path::new(env!("CARGO_MANIFEST_DIR"))
        .join("tests")
        .join(name)
}

fn target_dir(name: &str) -> PathBuf {
    std::env::var_os("CARGO_TARGET_TMPDIR")
        .map_or_else(
            || Path::new(env!("CARGO_MANIFEST_DIR")).join("target"),
            PathBuf::from,
        )
        .join("compile-fixtures")
        .join(name)
}

fn cargo_fixture(fixture: &CompileFixture) -> (bool, String) {
    let fixture_dir = fixture_dir(fixture.name);
    let manifest = fixture_dir.join("Cargo.toml");
    let output = Command::new(env!("CARGO"))
        .current_dir(&fixture_dir)
        .arg(fixture.command.as_str())
        .arg("--manifest-path")
        .arg(&manifest)
        .arg("--target-dir")
        .arg(target_dir(fixture.name))
        .output()
        .unwrap_or_else(|err| {
            panic!(
                "failed to run cargo {} for fixture `{}`: {err}",
                fixture.command.as_str(),
                fixture.name
            )
        });
    let stderr = String::from_utf8_lossy(&output.stderr);
    let stdout = String::from_utf8_lossy(&output.stdout);
    let combined = format!("{stdout}{stderr}");
    (output.status.success(), combined)
}

#[test]
fn dependency_layout_fixtures() {
    let fixtures = [
        CompileFixture {
            name: "compile-pass-facade",
            command: CargoCommand::Test,
            expected_success: true,
            expected_output: None,
        },
        CompileFixture {
            name: "compile-pass-direct-core",
            command: CargoCommand::Test,
            expected_success: true,
            expected_output: None,
        },
        CompileFixture {
            name: "compile-fail-missing-core",
            command: CargoCommand::Check,
            expected_success: false,
            expected_output: Some(
                "autoagents-derive requires either `autoagents` or `autoagents-core`",
            ),
        },
        CompileFixture {
            name: "compile-fail-missing-serde-json",
            command: CargoCommand::Check,
            expected_success: false,
            expected_output: Some("serde_json"),
        },
        CompileFixture {
            name: "compile-fail-missing-async-trait",
            command: CargoCommand::Check,
            expected_success: false,
            expected_output: Some("async-trait"),
        },
    ];

    for fixture in fixtures {
        let (success, output) = cargo_fixture(&fixture);
        assert_eq!(
            success,
            fixture.expected_success,
            "unexpected cargo {} status for fixture `{}`; output: {output}",
            fixture.command.as_str(),
            fixture.name
        );

        if let Some(expected_output) = fixture.expected_output {
            assert!(
                output.contains(expected_output),
                "fixture `{}` did not emit expected output `{expected_output}`; output: {output}",
                fixture.name
            );
        }
    }
}
