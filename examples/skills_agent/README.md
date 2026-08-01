# Agent Skills example

This example explicitly enables AutoAgents' optional `skill-watch` feature and selects `SkillRefreshStrategy::Watch`. It discovers `SKILL.md` directories when the agent is built and keeps a watched catalog for the lifetime of an interactive conversation. Add, edit, or remove a skill below the configured directory while the process is running; the next prompt or `/skills` command uses the new registry snapshot without restarting.

The example currently defaults to a local OpenAI-compatible llama server at
`http://127.0.0.1:8880/v1` using the model loaded on that server:

```bash
cargo run -p skills-agent
```

Enter prompts at `skills>`. Use `/skills` to inspect the live catalog,
`/reset-skills` to clear conversation activations, and `/exit` to stop. Pass a
prompt after `--` for one-shot operation.

Override `LLAMA_SERVER_URL`, `LLAMA_SERVER_MODEL`, or `LLAMA_SERVER_API_KEY`
when the endpoint, served model name, or authentication differs. The API key
falls back to `OPENAI_API_KEY` and then to the harmless local value `local`.

The console prints boot discovery, each agent turn, `activate_skill` and
`read_skill_resource` arguments/results, and the final model response.

Set `AUTOAGENTS_SKILLS_DIR` to use a different directory. The caller-owned skill session persists across direct `run` calls and is reset explicitly at the conversation boundary.

Only point the registry at trusted content: activated skill Markdown becomes part of the agent's system prompt. Skill resource reads remain scoped to discovered relative files under the activated skill.
