# Testing

Use Rust’s test harness for unit and integration tests. Async tests use `#[tokio::test]` where needed.

## Fast Test Cycle

```bash
cargo test --workspace --features default \
  --exclude autoagents-burn \
  --exclude autoagents-mistral-rs \
  --exclude wasm_agent
```

## Lint and Format

```bash
cargo fmt --all
cargo clippy --all-features --all-targets -- -D warnings
```

## Coverage

```bash
cargo install cargo-tarpaulin
cargo tarpaulin --all-features --out html
```

Keep tests focused and avoid unrelated changes. Match the style and structure used in existing tests across crates.
