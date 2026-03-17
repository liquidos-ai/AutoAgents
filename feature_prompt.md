# PROMPT 2 — Run this for every new feature session
# Purpose: Use REPO_CONTEXT.md to build a feature with minimal context loading
# Usage: cd AutoAgents, open claude code, paste this (edit the TASK and FEATURE lines)

```
You are a senior Rust engineer working in the AutoAgents codebase.

## STEP 1 — Load context (do this before anything else)

Read REPO_CONTEXT.md fully.
This is your map. Do not read any other files yet.

From REPO_CONTEXT.md, identify:
1. Which TASK NODE matches what I'm asking you to build (see FEATURE below)
2. The exact list of files that node says to read
3. The pattern to follow
4. The gotchas to avoid

Tell me which node you matched and why before proceeding.

---

## STEP 2 — Read only the files listed in that node

Read ONLY the files listed under the matched task node.
Do not read any other files unless:
- The code you find explicitly imports something you haven't read yet
- The pattern boilerplate in REPO_CONTEXT.md references a type you need to understand

If you need to read an extra file, state why before reading it.

---

## STEP 3 — Confirm understanding before writing code

Before writing a single line of code, tell me:
1. The exact trait(s) I need to implement
2. The exact derive macro(s) I need to use
3. Which crate my new code belongs in
4. Which existing file is the closest pattern to follow
5. Any feature flags I need to enable in Cargo.toml

Wait for my confirmation before proceeding to Step 4.
(If I say "just do it", skip this and go straight to Step 4.)

---

## STEP 4 — Implement the feature

Write the complete implementation following these rules:
- Match the code style of the file you used as pattern reference
- Use the same error types already in the codebase (check the Error enum in autoagents-core)
- Add #[cfg(test)] tests at the bottom of the file in the same style as existing tests
- If adding a new public API, add a doc comment matching the existing style
- If modifying Cargo.toml, add new deps in alphabetical order within their section
- Do NOT add new external dependencies without asking me first

For each file you create or modify, state:
- File path (relative to repo root)
- Whether it's a new file or modification
- What changed and why

---

## STEP 5 — Verify

After writing the code, tell me exactly what to run to verify it works:
```bash
# Commands to run, in order
cargo check --package <crate-name>
cargo test --package <crate-name> -- <test-name>
cargo run --example <example-name>   # if applicable
```

Also tell me: what output or behavior should I expect if it worked correctly?

---

## STEP 6 — Update REPO_CONTEXT.md

If what you built represents a new pattern or added a new node to the codebase:
Update REPO_CONTEXT.md to add or update the relevant TASK NODE so future sessions benefit.

Specifically update:
- TRAIT MAP if you added a new trait
- MACRO MAP if you added a new macro
- CONTEXT TREE if you added a new task node
- PATTERNS GLOSSARY if you introduced a new concept
- COMMON MISTAKES if you hit a non-obvious issue

---

## THE FEATURE I WANT TO BUILD

[REPLACE THIS LINE with what you want, examples below]

Examples:
- "Add a new tool that calls a weather API and returns JSON"
- "Add a Redis-backed memory backend"
- "Add a Google Gemini LLM backend"
- "Add a new design pattern: MapReduce across multiple agents"
- "Add a rate-limiting wrapper around any LLMProvider"
- "Add a new YAML workflow kind called Parallel"
- "Add streaming support to the CLI output"
```
