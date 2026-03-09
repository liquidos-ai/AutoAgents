"""Example: declarative agent definition + hook callbacks with Python bindings."""

import asyncio
import os
import sys
from dataclasses import dataclass
from pathlib import Path

PACKAGE_ROOT = Path(__file__).resolve().parents[1]
if str(PACKAGE_ROOT) not in sys.path:
    sys.path.insert(0, str(PACKAGE_ROOT))


from autoagents import (
    AgentBuilder,
    HookOutcome,
    LLMBuilder,
    Task,
    tool,
)
from autoagents.prebuilt import BasicAgent, SlidingWindowMemory
from autoagents.types import ExecutorContext, ExecutorTask


@tool(description="Add two numbers")
def add(a: float, b: float) -> float:
    return a + b


@dataclass
class MathOutput:
    value: int
    explanation: str


class CustomHooks:
    async def on_agent_create(self) -> None:
        print("Agent Create Hook")

    async def on_run_start(self, task: ExecutorTask, ctx: ExecutorContext) -> HookOutcome:
        print(f"Agent Start Hook: prompt={task.get('prompt')}")
        print(f"Context: id={ctx.get('id')} name={ctx.get('name')}")
        return HookOutcome.CONTINUE


async def main() -> None:
    api_key = os.environ.get("OPENAI_API_KEY")
    if not api_key:
        raise RuntimeError("OPENAI_API_KEY is required")

    llm = LLMBuilder("openai").api_key(api_key).model("gpt-4o-mini").build()

    executor = (
        BasicAgent("hooks_agent", "You are a helpful assistant")
        .tools([add])
        .hooks(CustomHooks())
    )

    handle = await (
        AgentBuilder(executor)
        .llm(llm)
        .memory(SlidingWindowMemory(window_size=20))
        .output(MathOutput)
        .build()
    )

    print("=== Running hooks example ===")
    result = await handle.run(Task(prompt="What is 20 + 10?"))
    print("Result:", result["response"])


if __name__ == "__main__":
    asyncio.run(main())
