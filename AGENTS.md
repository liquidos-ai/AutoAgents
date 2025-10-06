# Repository Guidelines

## Project Structure & Module Organization
- Workspace layout: `crates/*` (core libraries), `examples/*` (runnable crates), `docs/` (documentation), `assets/` (images), `demo_models/` (sample models).
- Key crates: `autoagents` (entry), `autoagents-core`, `autoagents-llm`, `autoagents-derive`, `autoagents-toolkit`, optional: `autoagents-burn`, `autoagents-onnx`.
- Tests live alongside code (`mod tests`) and as integration tests under each crate’s `tests/`.

## Build, Test, and Development Commands
- Build workspace: `cargo build --workspace --all-features`
- Lint: `cargo clippy --all-features --all-targets -- -D warnings`
- Format: `cargo fmt --all`
- Test fast (CI-like): `cargo test --workspace --features default --exclude autoagents-burn --exclude autoagents-mistral-rs --exclude wasm_agent`
- Docs: `cargo doc --all-features --no-deps`
- Run examples: `cargo run -p basic-example` or `cargo run -p coding_agent`
- Git hooks: install once with `lefthook install`; run locally via `lefthook run pre-commit`

## Coding Style & Naming Conventions
- Rust 2021 edition; 4-space indent; `rustfmt` required.
- Names: crates `kebab-case`, modules/functions `snake_case`, types/traits `UpperCamelCase`, constants `SCREAMING_SNAKE_CASE`.
- Keep warnings at zero (`-D warnings`); prefer `Result` over panics; avoid `unwrap()` in library code.
- Feature flags mirror providers (e.g., `openai`, `anthropic`) and bundles (e.g., `full`, `logging`).

## Testing Guidelines
- Use Rust’s built-in test harness; async tests with `#[tokio::test]` where needed.
- Place integration tests in `crates/<name>/tests/` with descriptive `snake_case` filenames.
- Coverage (optional): `cargo install cargo-tarpaulin && cargo tarpaulin --all-features --out html`.

## Commit & Pull Request Guidelines
- Commit style follows existing history: `[FEATURE]`, `[BUG]`, `[MAINT]`, `[TASK]` + concise subject. Example: `[FEATURE]: Add MCP Support (#79)`.
- PRs must include: clear description, linked issues, tests for new behavior, updated docs/examples where relevant, and passing hooks (fmt, clippy, tests).

## Security & Configuration Tips
- Never commit secrets. Configure providers via env vars (examples): `OPENAI_API_KEY`, `ANTHROPIC_API_KEY`, `GROQ_API_KEY`, `AZURE_OPENAI_API_KEY`.
- Example run with key: `OPENAI_API_KEY=... cargo run -p basic-example`.
- Use provider-specific feature flags only where required; keep defaults minimal.

## Agent-Specific Notes
- Reusable tools belong in `autoagents-toolkit`; project-specific tools can live within examples or crate-local modules.
- Derive macros: `#[agent]`, `#[tool]`, `#[derive(AgentOutput, ToolInput)]` are preferred for consistency and type safety.
