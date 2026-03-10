"""Example: run llama.cpp with CUDA variant bindings."""

import asyncio
import os
import sys
from pathlib import Path

CORE_PY_BINDINGS = Path(__file__).resolve().parents[2] / "autoagents"
if str(CORE_PY_BINDINGS) not in sys.path:
    sys.path.insert(0, str(CORE_PY_BINDINGS))

from autoagents import AgentBuilder, Task
from autoagents.prebuilt import ReActAgent, SlidingWindowMemory

from autoagents_llamacpp_cuda import LlamaCppBuilder, backend_build_info


async def main() -> None:
    print("Build info:", backend_build_info())

    llm = await (
        LlamaCppBuilder()
        .repo_id("unsloth/Qwen3.5-9B-GGUF")
        .hf_filename("Qwen3.5-9B-Q4_0.gguf")
        .max_tokens(256)
        .temperature(0.7)
        .build()
    )

    agent_def = ReActAgent("local_llama_cuda", "Local llama.cpp assistant (CUDA)").max_turns(10)

    handle = await (
        AgentBuilder(agent_def)
        .llm(llm)
        .memory(SlidingWindowMemory(window_size=20))
        .build()
    )

    result = await handle.run(Task(prompt="Write one short sentence about Rust."))
    print(result["response"])

    print("\n=== Streaming ===")
    async for chunk in handle.run_stream(Task(prompt="What is 10 + 32?")):
        print(chunk)


if __name__ == "__main__":
    asyncio.run(main())
    # Skip Python/Rust teardown: CUDA driver cleanup can throw C++ exceptions
    # that abort the process. The CUDA driver reclaims GPU memory at the OS
    # level when the process exits.
    os._exit(0)
