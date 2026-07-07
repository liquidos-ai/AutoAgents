# Testing

Use Rust’s test harness for unit and integration tests. Async tests use `#[tokio::test]` where needed.

## Fast Test Cycle

```bash
cargo test --workspace --features default \
  --exclude autoagents-mistral-rs \
  --exclude wasm_agent
```

## Lint and Format

```bash
cargo fmt --all -- --check
cargo clippy --workspace --all-targets --features full -- -D warnings
```

## Release Readiness

```bash
make release-check
cargo test --workspace --features full --exclude autoagents-mistral-rs
cargo test --workspace --no-default-features --exclude autoagents-mistral-rs
cargo test -p autoagents-mistral-rs
cargo doc --no-deps --features full \
  -p autoagents \
  -p autoagents-core \
  -p autoagents-llm \
  -p autoagents-derive \
  -p autoagents-protocol \
  -p autoagents-toolkit \
  -p autoagents-guardrails \
  -p autoagents-qdrant \
  -p autoagents-speech \
  -p autoagents-telemetry
```

## Coverage

```bash
cargo install cargo-tarpaulin
rustup component add llvm-tools-preview
make coverage-rust
```

Keep tests focused and avoid unrelated changes. Match the style and structure used in existing tests across crates.
