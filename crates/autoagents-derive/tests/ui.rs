use std::collections::BTreeSet;
use std::path::{Component, Path};

const COMPILE_FAIL_FIXTURES: &[&str] = &[
    "tests/ui/compile_fail/agent_missing_name.rs",
    "tests/ui/compile_fail/invalid_choice.rs",
    "tests/ui/compile_fail/invalid_output_field.rs",
    "tests/ui/compile_fail/invalid_strict.rs",
    "tests/ui/compile_fail/missing_tool_input_derive.rs",
    "tests/ui/compile_fail/tool_missing_input.rs",
    "tests/ui/compile_fail/unsupported_agent_output_type.rs",
];

#[test]
fn compile_fail_diagnostics() {
    let t = trybuild::TestCases::new();

    for fixture in COMPILE_FAIL_FIXTURES {
        t.compile_fail(fixture);
    }
}

#[test]
fn compile_fail_fixtures_are_registered() {
    let compile_fail_dir = Path::new(env!("CARGO_MANIFEST_DIR")).join("tests/ui/compile_fail");
    let registered = COMPILE_FAIL_FIXTURES
        .iter()
        .map(|fixture| fixture.to_string())
        .collect::<BTreeSet<_>>();
    let discovered = std::fs::read_dir(&compile_fail_dir)
        .expect("compile_fail fixture directory should exist")
        .map(|entry| {
            let entry = entry.expect("compile_fail fixture entry should be readable");
            entry.path()
        })
        .filter(|path| path.extension().is_some_and(|extension| extension == "rs"))
        .map(|path| {
            let relative_path = path
                .strip_prefix(env!("CARGO_MANIFEST_DIR"))
                .expect("fixture path should be inside crate")
                .to_path_buf();
            slash_separated_path(&relative_path)
        })
        .collect::<BTreeSet<_>>();

    assert_eq!(registered, discovered);
}

fn slash_separated_path(path: &Path) -> String {
    path.components()
        .filter_map(|component| match component {
            Component::Normal(segment) => Some(segment.to_string_lossy()),
            _ => None,
        })
        .collect::<Vec<_>>()
        .join("/")
}
