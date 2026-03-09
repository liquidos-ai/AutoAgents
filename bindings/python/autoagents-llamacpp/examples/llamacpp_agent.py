"""Example: run a local GGUF model with AutoAgents."""

import asyncio
import sys
from pathlib import Path

CORE_PY_BINDINGS = Path(__file__).resolve().parents[2] / "autoagents"
if str(CORE_PY_BINDINGS) not in sys.path:
    sys.path.insert(0, str(CORE_PY_BINDINGS))

from autoagents import AgentBuilder, Task
from autoagents.prebuilt import ReActAgent, SlidingWindowMemory

from autoagents_llamacpp import LlamaCppBuilder


async def main() -> None:
    llm = await (
        LlamaCppBuilder()
        .repo_id("unsloth/Qwen3.5-9B-GGUF")
        .hf_filename("Qwen3.5-9B-Q4_0.gguf")
        .max_tokens(256)
        .temperature(0.7)
        .build()
    )

    agent_def = ReActAgent("local_llama", "Local llama.cpp assistant").max_turns(10)

    handle = await (
        AgentBuilder(agent_def)
        .llm(llm)
        .memory(SlidingWindowMemory(window_size=20))
        .build()
    )

    result = await handle.run(Task(prompt="Write one short sentence about Rust."))
    print(result["response"])


if __name__ == "__main__":
    asyncio.run(main())
