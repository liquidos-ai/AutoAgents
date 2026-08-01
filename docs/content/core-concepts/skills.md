---
title: Agent Skills
description: Add progressively disclosed SKILL.md instructions and resources to ReAct and CodeAct agents.
---

# Agent Skills

Agent Skills let an AutoAgents agent discover specialized instructions without placing every instruction in every prompt. The system prompt initially contains only each skill's name and description. When a task matches, the model calls `activate_skill`; the full instructions are injected on the following turn and remain active for the caller-owned conversation session.

## Skill layout

Each direct child of the configured directory is one skill. Its directory name must match the frontmatter `name`.

```text
skills/
└── release-notes/
    ├── SKILL.md
    └── references/
        └── style.md
```

```markdown
---
name: release-notes
description: Write concise product release notes.
license: Apache-2.0
compatibility: Requires access to the supplied style reference.
metadata:
  owner: product
allowed-tools: read_skill_resource
---

Read `references/style.md`, then lead with user impact and use active voice.
```

`name` and `description` are required. `license`, `compatibility`, string-valued `metadata`, and `allowed-tools` are optional. `allowed-tools` is parsed into typed selectors and supplied to the configured `SkillPolicy`. A skill file never grants itself authority: policy approval and the normal permission check for each eventual tool call still apply.

## Attach a registry

Agent Skills require a multi-turn executor. ReAct and CodeAct support them; configuring skills on an unsupported executor fails while building the agent.

```rust
use autoagents::prelude::*;
use std::sync::Arc;

let registry = Arc::new(SkillRegistry::from_directory(
    "./skills",
    SkillRefreshStrategy::Manual,
));
let session = Arc::new(SkillSession::new());
let skills = SkillConfiguration::new(
    registry,
    session,
    Arc::new(TrustedSkillPolicy),
);

let handle = AgentBuilder::<_, DirectAgent>::new(ReActAgent::new(MyAgent))
    .llm(llm)
    .skills(skills)
    .build()
    .await?;
```

The registry scans at build time. Reload behavior is explicit; constructing or importing AutoAgents does not start a watcher. Changes are applied atomically before a model turn or Agent Skills tool call. Routine polling scans compare the `SKILL.md` and resource manifests with cached file metadata first, so unchanged skill files are not read or parsed again. Explicit refreshes and watcher-reported changes verify file contents, including same-size edits on filesystems with coarse timestamps. Invalid in-progress edits retain the last valid skill until the package becomes valid. Updating `SKILL.md` or any resource changes the skill revision and deactivates an active skill at the next boundary, so changed content must pass policy again; deletion deactivates it as well.

## Reload strategies

`DirectorySkillSource::new` and `SkillRegistry::from_directory` require a `SkillRefreshStrategy`:

- `Manual` performs initial discovery and then reuses its cache until `SkillRegistry::refresh_now` or `BaseAgent::refresh_skills` explicitly invalidates it.
- `Poll { interval }` performs another scan at a safe registry boundary after the interval has elapsed. It does not create a filesystem watcher.
- `Watch { debounce, fallback_interval }` uses recursive filesystem notifications, applies settled changes at safe boundaries, and retains polling as a recovery path.

Watcher support is optional. Enable the `skill-watch` feature when declaring AutoAgents:

```toml
autoagents = { version = "0.4", features = ["skill-watch"] }
```

```rust
use std::time::Duration;

let registry = Arc::new(SkillRegistry::from_directory(
    "./skills",
    SkillRefreshStrategy::Watch {
        debounce: Duration::from_millis(150),
        fallback_interval: Duration::from_secs(30),
    },
));
```

Selecting `Watch` without compiling `skill-watch` returns `SkillError::WatchUnavailable`. Without that feature, manual and polling modes do not pull in `notify`; neither mode initializes a watcher even when watcher support is compiled.

`SkillSession` is caller-owned. Reusing it across `agent.run()` calls keeps activated instructions in a conversation; call `BaseAgent::reset_skill_session` when starting or restoring another conversation. `BaseAgent::refresh_skills` forces an immediate source refresh, and `BaseAgent::skill_snapshot` exposes the current immutable catalog. The model can also call `deactivate_skill` when instructions no longer apply.

An active skill is not loaded from disk on every model turn. Its parsed instructions are retained in the immutable registry snapshot and selected by the session's name/revision entry. AutoAgents still includes those instructions in the system message sent on each chat-completions request: OpenAI-compatible chat APIs are request-stateless, so omitting them after the activation turn would make later turns lose the skill. Providers may reuse that stable prefix through their own prompt/KV cache.

`SkillPolicy` gates activation and resource access. `TrustedSkillPolicy` explicitly approves both operations and is intended only for sources the host already trusts; implement `SkillPolicy` when activation or resource reads need an approval boundary. `AgentHooks` exposes skill catalog, activation, deactivation, resource, and error callbacks, while protocol `Event::Skill` provides the corresponding observable lifecycle stream.

Use `DirectorySkillSource::with_precedence` and `SkillRegistry::new` to combine directories. A higher precedence source wins when names collide. Implement `SkillSource` for another source type and construct validated entries with `Skill::from_document` and `SkillSourceSnapshot::new`.

## WASM compatibility

The skill control plane is portable, but the built-in filesystem backend is currently native-only.

| Capability | Native targets | `wasm32` targets |
| --- | --- | --- |
| Skill metadata, parsing, and revisions | Supported | Supported |
| Custom or in-memory `SkillSource` implementations | Supported | Supported |
| Registry refresh, precedence, and immutable snapshots | Supported | Supported |
| Activation policy and conversation-scoped `SkillSession` | Supported | Supported |
| Prompt composition, lifecycle hooks, and protocol events | Supported | Supported |
| `DirectorySkillSource` discovery | Supported | Unavailable; discovery returns `SkillError::SourceUnavailable` |
| Recursive filesystem watching | Supported | Unavailable |
| Filesystem-backed `read_skill_resource` | Supported | Unavailable; the tool returns `SkillError::SourceUnavailable` |
| Capability-scoped and no-follow filesystem reads | Supported | Not compiled |

On `wasm32`, provide a custom `SkillSource` to populate the catalog from application-owned or remote data. Catalog discovery and instruction activation will work, but the current `read_skill_resource` implementation cannot read those resources because resource loading does not yet have a non-filesystem backend.

The current target gate applies to every `wasm32` target, including WASI. The `skill-watch` feature does not change that boundary. Supporting WASI filesystem skills requires a separate WASI resource and watcher implementation rather than enabling the native ambient-filesystem code unchanged.

## Resources and trust

The model can read only a resource that was discovered under an activated skill, using `read_skill_resource`. Paths must be normalized and relative. Directory discovery rejects linked `SKILL.md` files, skips linked resources, bounds document and resource counts and bytes, and captures a capability-scoped directory handle that cannot be redirected by later path replacement. Reads are no-follow, bounded, and checked against the resource digest recorded in the approved skill revision before bytes reach the model.

Default discovery limits are 256 skills, 256 KiB per `SKILL.md` with a 4 MiB aggregate document budget, 512 resources per skill, 1 MiB per resource with a 16 MiB aggregate resource budget, and six resource-directory levels. Override them with `SkillDiscoveryLimits` when the host has a different trusted workload.

Skill instructions are executable prompt content. Configure only trusted skill sources; AutoAgents does not implicitly scan user or system directories. Resource access does not grant arbitrary filesystem access, and Agent Skills reserve the tool names `activate_skill`, `deactivate_skill`, and `read_skill_resource`.

See the runnable [`skills-agent` example](https://github.com/liquidos-ai/AutoAgents/tree/main/examples/skills_agent) for a complete setup.
