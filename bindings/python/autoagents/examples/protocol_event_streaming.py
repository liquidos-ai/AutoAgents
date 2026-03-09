"""Example: consume typed protocol events while a run is in progress."""

import asyncio
import os
import sys
from pathlib import Path

PACKAGE_ROOT = Path(__file__).resolve().parents[1]
if str(PACKAGE_ROOT) not in sys.path:
    sys.path.insert(0, str(PACKAGE_ROOT))

from autoagents import AgentBuilder, LLMBuilder, ToolCallRequested, tool
from autoagents.prebuilt import BasicAgent, SlidingWindowMemory


@tool(description="Add two integers")
def add(a: int, b: int) -> int:
    return a + b


async def main() -> None:
    api_key = os.environ.get("OPENAI_API_KEY")
    if not api_key:
        raise RuntimeError("OPENAI_API_KEY is required")

    llm = LLMBuilder("openai").api_key(api_key).model("gpt-4o-mini").build()
    handle = await (
        AgentBuilder(BasicAgent("events_demo", "Use tools when useful.").tools([add]))
        .llm(llm)
        .memory(SlidingWindowMemory(window_size=20))
        .build()
    )

    stream = handle.event_stream()
    run_task = asyncio.create_task(handle.run("What is 20 + 22?"))

    while True:
        if run_task.done():
            break
        try:
            event = await asyncio.wait_for(anext(stream), timeout=0.25)
        except TimeoutError:
            continue
        print(event)
        if isinstance(event, ToolCallRequested):
            print(f"tool={event.tool_name}")

    result = await run_task
    print(result["response"])


if __name__ == "__main__":
    asyncio.run(main())
