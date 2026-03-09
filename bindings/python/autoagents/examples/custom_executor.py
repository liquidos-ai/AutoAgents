"""Example: custom executor callbacks injected into the Rust runtime.

Python provides a plain callback object.
Rust still owns scheduling, streaming, and execution orchestration.
"""

import asyncio
import os
import sys
from pathlib import Path
from typing import AsyncIterator

PACKAGE_ROOT = Path(__file__).resolve().parents[1]
if str(PACKAGE_ROOT) not in sys.path:
    sys.path.insert(0, str(PACKAGE_ROOT))

from autoagents import (
    HookOutcome,
    LLMBuilder,
)
from autoagents.experimental import CustomExecutor, ExperimentalAgentBuilder
from autoagents.prebuilt import SlidingWindowMemory
from autoagents.types import ExecutorConfig, ExecutorContext, ExecutorOutput, ExecutorTask


class CustomHooks:
    async def on_agent_create(self) -> None:
        print("[hook] on_agent_create")

    async def on_run_start(
        self, task: ExecutorTask, ctx: ExecutorContext
    ) -> HookOutcome:
        print(f"[hook] on_run_start prompt={task.get('prompt')} ctx={ctx.get('name')}")
        return HookOutcome.CONTINUE


class CustomMathExecutor:
    def config(self) -> ExecutorConfig:
        return {"max_turns": 1}

    async def execute(
        self, task: ExecutorTask, ctx: ExecutorContext
    ) -> ExecutorOutput:
        prompt = str(task.get("prompt", ""))
        llm = ctx["llm"]

        # Owns execution loop and can call Rust context resources directly.
        llm_result = await llm.chat([{"role": "user", "content": prompt}])
        text = llm_result.get("text") or ""
        if not text:
            text = f"Custom executor processed: {prompt}"

        return {"response": text, "done": True, "tool_calls": []}

    async def execute_stream(
        self,
        task: ExecutorTask,
        ctx: ExecutorContext,
    ) -> AsyncIterator[ExecutorOutput]:
        result = await self.execute(task, ctx)
        yield {"response": result["response"], "done": True, "tool_calls": []}


async def main() -> None:
    api_key = os.environ.get("OPENAI_API_KEY")
    if not api_key:
        raise RuntimeError("OPENAI_API_KEY is required")

    llm = LLMBuilder("openai").api_key(api_key).model("gpt-4o-mini").build()

    executor = (
        CustomExecutor(
            "custom_executor_agent",
            "Agent using injected custom Python executor",
            CustomMathExecutor(),
        )
        .hooks(CustomHooks())
        .max_turns(1)
    )

    handle = await (
        ExperimentalAgentBuilder(executor)
        .llm(llm)
        .memory(SlidingWindowMemory(window_size=20))
        .build()
    )

    result = await handle.run("What is 20 + 10?")
    print(result["response"])


if __name__ == "__main__":
    asyncio.run(main())
