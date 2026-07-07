# Release Checklist

This checklist is the source of truth for cutting AutoAgents releases. Complete
each section before publishing a `vX.Y.Z` tag.

## Before the Version Bump

- Confirm all release-target issues are merged or explicitly deferred.
- Run `make release-check` on the release branch.
- Run `cargo fmt --all -- --check`.
- Run `cargo clippy --all-features --all-targets -- -D warnings`.
- Run `cargo test --workspace --features default --exclude autoagents-mistral-rs`.
- Run `cargo test --workspace --features full --exclude autoagents-mistral-rs`.
- Run `cargo test --workspace --no-default-features --exclude autoagents-mistral-rs`.
- Run `cargo test -p autoagents-mistral-rs`.
- Run the CI rustdoc package set:
  `cargo doc --no-deps --features full -p autoagents -p autoagents-core -p autoagents-llm -p autoagents-derive -p autoagents-protocol -p autoagents-toolkit -p autoagents-guardrails -p autoagents-qdrant -p autoagents-speech -p autoagents-telemetry`.
- Run `make python-bindings-test-clean` in a Python 3.12 virtualenv.
- Confirm documented examples compile and CLI examples print help successfully.

## Version Bump

- Update `[workspace.package].version` in `Cargo.toml`.
- Update all AutoAgents workspace dependency pins in `Cargo.toml`.
- Update sibling Python dependency pins in `bindings/python/*/pyproject.toml`.
- Update Rust dependency snippets in `README.md` and `docs/content/**`.
- Run `make release-check` again.

## Package Dry Runs

- Run `cargo package --allow-dirty --no-verify` for publishable Rust crates that
  are part of the release.
- Build Python sdists and wheels through `.github/workflows/python-bindings-ci.yml`.
- Inspect wheel metadata for project URLs, classifiers, extras, and dependency
  pins.
- Confirm local backend wheels are built only for supported platform/accelerator
  combinations.

## Publish

- Publish Rust crates in dependency order.
- Push the signed release tag.
- Let the Python publishing workflow build and upload release artifacts.
- Generate GitHub Release notes from the release diff, then add migration notes
  and known limitations when needed.

## Post-Publish Verification

- Install `autoagents` from crates.io in a new Rust project and run the quick
  start compile path.
- Install `autoagents-py` from PyPI in a clean virtualenv and import it.
- Install at least one CPU local backend extra from PyPI and verify dependency
  resolution.
- Confirm docs, README badges, and release links point to the new version.
