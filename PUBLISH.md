# Cargo Publish Guide for AutoAgents

**Follow the below instructions in sequence**

1. Create a release branch from `main`:

```shell
git checkout main
git pull origin main
git checkout -b feature/vx.x.x
```

2. Update the Cargo.toml `[workspace.package]` version and `[workspace.dependencies]` version. We use SemVer versions.
   Update the pyproject.toml version and dependencies version and variants
   Run `Cargo update` and `make python-bindings-build` or `make python-bindings-build-cuda`

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

7. Publish to Crates.io from `main` (MAINTAIN the order)

```shell
cd crates/autoagents-derive
cargo publish --dry-run # Test first
cargo publish
```

```shell
cd crates/autoagents-protocol
cargo publish --dry-run # Test first
cargo publish
```

```shell
cd ../autoagents-llm
cargo publish --dry-run
cargo publish
```

```shell
cd ../autoagents-llamacpp
cargo publish --dry-run
cargo publish
```

---
```shell
cd ../autoagents-mistral-rs
cargo publish --dry-run
cargo publish
```
---

```shell
cd ../autoagents-core
cargo publish --dry-run
cargo publish
```

```shell
cd ../autoagents
cargo publish --dry-run
cargo publish
```

```shell
cd ../autoagents-toolkit
cargo publish --dry-run
cargo publish
```

```shell
cd ../autoagents-qdrant
cargo publish --dry-run
cargo publish
```

```shell
cd ../autoagents-speech
cargo publish --dry-run
cargo publish
```

```shell
cd ../autoagents-telemetry
cargo publish --dry-run
cargo publish
```

```shell
cd ../autoagents-guardrails
cargo publish --dry-run
cargo publish
```

---

8. Build Python packages locally before tagging

```shell
# Build manylinux wheels (CPU variants)
make python-bindings-build
```
This builds:
- `autoagents` base wheel
- `autoagents-guardrails` wheel
- `autoagents-llamacpp` CPU wheel
- `autoagents-mistral-rs` CPU wheel

To Test the cuda version run
```shell
make python-bindings-build-cuda
```

#### Note

- The `Python Bindings CI` GitHub Actions workflow runs on PRs to validate the Python packaging matrix before merge.
- The Python bindings are published via GitHub Actions only when a `v*` tag is pushed.
- Use the local build commands above to validate the Python bindings before creating the release tag.

9. Create the release tag on the merged `main` commit:

```shell
cd ../.. # Back to project root
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

10. Pushing the `vx.x.x` tag triggers the Python bindings workflow and publishes the Python packages from GitHub Actions.
