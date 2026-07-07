# `trybuild` UI Tests

These fixtures protect the proc-macro diagnostics that users see at compile time.

Compile-fail fixtures live in `compile_fail/` and each `.rs` file has a matching
committed `.stderr` snapshot. Refresh snapshots only after reviewing the
diagnostic change:

```sh
TRYBUILD=overwrite cargo test -p autoagents-derive --test ui
```

Add new fixtures explicitly in `tests/ui.rs` so each diagnostic case is reviewed
as part of the test harness.
