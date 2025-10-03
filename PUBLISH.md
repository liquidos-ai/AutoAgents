# Cargo Publish Guide for AutoAgents

**Follow the below instructions in sequence**

1. Update the Cargo.toml `[workspace.package]` version and `[workspace.dependencies]` version. We use SemVer versions.
2. `git add . && git commit -m "[MAINT]: bump version to x.x.x"`

3. Publish to Crates.io (MAINTAIN the order)

```shell
cd crates/autoagents-derive
cargo publish --dry-run # Test first
cargo publish
```

```shell
cd ../autoagents-llm
cargo publish --dry-run
cargo publish
```

```shell
cd ../autoagents-onnx
cargo publish --dry-run
cargo publish
```

```shell
cd ../autoagents-burn
cargo publish --dry-run
cargo publish
```

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

4. Create release tag:

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
git push origin main --tags
```
