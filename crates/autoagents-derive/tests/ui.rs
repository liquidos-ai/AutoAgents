#[test]
fn compile_fail_ui() {
    let t = trybuild::TestCases::new();
    t.compile_fail("tests/ui/compile_fail/*.rs");
}
