"""Example: run a local model with AutoAgents via mistral.rs."""

import asyncio
import sys
from pathlib import Path

CORE_PY_BINDINGS = Path(__file__).resolve().parents[2] / "autoagents"
if str(CORE_PY_BINDINGS) not in sys.path:
    sys.path.insert(0, str(CORE_PY_BINDINGS))

from autoagents import AgentBuilder, Task
from autoagents.prebuilt import ReActAgent, SlidingWindowMemory

from autoagents_mistral_rs import MistralRsBuilder


async def main() -> None:
    llm = await (
        MistralRsBuilder()
        .repo_id("Qwen/Qwen3-8B")
        .max_tokens(256)
        .temperature(0.7)
        .logging(True)
        .isq_type("Q8_0")
        .build()
    )

    agent_def = ReActAgent("local_mistral", "Local mistral-rs assistant").max_turns(10)

    handle = await (
        AgentBuilder(agent_def)
        .llm(llm)
        .memory(SlidingWindowMemory(window_size=20))
        .build()
    )

    result = await handle.run(Task(prompt="Give me a two-line summary of Rust."))
    print(result["response"])


if __name__ == "__main__":
    asyncio.run(main())
