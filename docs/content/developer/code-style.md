# Code Style

Follow idiomatic Rust style and keep warnings at zero.

- Rust 2021 edition; 4‑space indent; format with `rustfmt`
- Names: crates `kebab-case`; modules/functions `snake_case`; types/traits `UpperCamelCase`; constants `SCREAMING_SNAKE_CASE`
- Prefer `Result` over panics; avoid `unwrap()` in library code
- Keep `clippy` warnings at zero (`cargo clippy --all-features --all-targets -- -D warnings`)
- Small, focused modules; clear error types; explicit conversions via `From`/`Into`
- Tests live alongside code (`mod tests`) and as integration tests under each crate’s `tests/`
