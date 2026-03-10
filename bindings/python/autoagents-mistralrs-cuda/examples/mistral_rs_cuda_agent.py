"""Example: run mistral.rs with CUDA variant bindings."""

import asyncio
import os
import sys
from pathlib import Path

CORE_PY_BINDINGS = Path(__file__).resolve().parents[2] / "autoagents"
if str(CORE_PY_BINDINGS) not in sys.path:
    sys.path.insert(0, str(CORE_PY_BINDINGS))

from autoagents import AgentBuilder, Task
from autoagents.prebuilt import ReActAgent, SlidingWindowMemory

from autoagents_mistral_rs_cuda import MistralRsBuilder, backend_build_info


async def main() -> None:
    print("Build info:", backend_build_info())

    llm = await (
        MistralRsBuilder()
        .repo_id("Qwen/Qwen3-8B")
        .max_tokens(256)
        .temperature(0.7)
        .logging(True)
        .isq_type("Q8_0")
        .build()
    )

    agent_def = ReActAgent(
        "local_mistral_cuda", "Local mistral-rs assistant (CUDA)"
    ).max_turns(10)

    handle = await (
        AgentBuilder(agent_def)
        .llm(llm)
        .memory(SlidingWindowMemory(window_size=20))
        .build()
    )

    result = await handle.run(
        Task(prompt="Give me a two-line summary of Rust programming langauge.")
    )
    print(result["response"])


if __name__ == "__main__":
    asyncio.run(main())
    # Skip Python/Rust teardown: a C++ exception escapes the CUDA driver during
    # mistral-rs model cleanup and aborts the process. The CUDA driver reclaims
    # GPU memory at the OS level when the process exits.
    os._exit(0)
