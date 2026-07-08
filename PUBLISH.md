# Publish Guide for AutoAgents

**Follow the below instructions in sequence**

1. Create a release branch from `main`:

```shell
git checkout main
git pull origin main
git checkout -b feature/vx.x.x
```

2. Update the Cargo.toml `[workspace.package]` version and `[workspace.dependencies]` versions. We use SemVer versions.
   Update the pyproject.toml versions, dependency versions, and variants.
   Run `cargo update` and `make python-bindings-build` or `make python-bindings-build-cuda`.

3. Commit the release changes on the release branch:

```shell
git add .
git commit -m "[MAINT]: bump version to x.x.x"
git push origin feature/vx.x.x
```

4. Open a PR from `feature/vx.x.x` to `main`, wait for CI to pass, and get it reviewed.

5. Merge the PR to `main`.

6. After the PR is merged, switch back to `main` and pull the merged commit that will be released:

```shell
git checkout main
git pull origin main
```

7. Validate the Rust crates from `main`:

```shell
cargo publish --dry-run \
  -p autoagents-derive \
  -p autoagents-protocol \
  -p autoagents-llm \
  -p autoagents-llamacpp \
  -p autoagents-mistral-rs \
  -p autoagents-core \
  -p autoagents-guardrails \
  -p autoagents-qdrant \
  -p autoagents-speech \
  -p autoagents-telemetry \
  -p autoagents \
  -p autoagents-toolkit
```

This dry run covers the full Rust crates.io release set:

- `autoagents-derive`
- `autoagents-protocol`
- `autoagents-llm`
- `autoagents-llamacpp`
- `autoagents-mistral-rs`
- `autoagents-core`
- `autoagents-guardrails`
- `autoagents-qdrant`
- `autoagents-speech`
- `autoagents-telemetry`
- `autoagents`
- `autoagents-toolkit`

The workspace `default-members` list intentionally excludes `autoagents-llamacpp`, `autoagents-mistral-rs`, and `autoagents-speech` for normal root-level development commands. The Rust release workflow still publishes those crates.

8. Publish Rust crates from GitHub Actions.

Run the `Release Rust Crates` workflow manually with `dry_run=true` before tagging if you want a CI dry run. Pushing the `vx.x.x` tag publishes the full Rust crates.io release set in dependency order. The workflow requires the `CARGO_REGISTRY_TOKEN` repository secret for real publishing.

9. Build Python packages locally before tagging.

```shell
# Build manylinux wheels (CPU variants)
make python-bindings-build
```

This builds:

- `autoagents` base wheel
- `autoagents-guardrails` wheel
- `autoagents-llamacpp` CPU wheel
- `autoagents-mistral-rs` CPU wheel

To test the CUDA version, run:

```shell
make python-bindings-build-cuda
```

#### Note

- The `Python Bindings CI` GitHub Actions workflow runs on PRs to validate the Python packaging matrix before merge.
- The `Release Python Bindings` GitHub Actions workflow publishes Python bindings when a `v*` tag is pushed.
- Use the local build commands above to validate the Python bindings before creating the release tag.

10. Create the release tag on the merged `main` commit:

```shell
git tag -a vx.x.x -m "Release vx.x.x

Features:
-

Improvements:
-
"
```

```shell
# Push the release tag
git push origin vx.x.x
```

11. Pushing the `vx.x.x` tag triggers both release workflows:

- `Release Rust Crates` publishes the full Rust crates.io release set.
- `Release Python Bindings` publishes the Python packages from GitHub Actions.
