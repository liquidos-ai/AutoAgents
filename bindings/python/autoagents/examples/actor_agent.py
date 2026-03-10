"""Example: actor-based agent running inside an Environment."""

import asyncio
import os
import sys
from pathlib import Path

PACKAGE_ROOT = Path(__file__).resolve().parents[1]
if str(PACKAGE_ROOT) not in sys.path:
    sys.path.insert(0, str(PACKAGE_ROOT))

from autoagents_py import AgentBuilder, Environment, LLMBuilder, TaskComplete, Topic, Runtime
from autoagents_py.prebuilt import BasicAgent, SlidingWindowMemory


async def main() -> None:
    api_key = os.environ.get("OPENAI_API_KEY")
    if not api_key:
        raise RuntimeError("OPENAI_API_KEY is required")

    llm = LLMBuilder("openai").api_key(api_key).model("gpt-4o-mini").build()
    runtime = Runtime()
    environment = Environment()
    environment.register_runtime(runtime)
    topic = Topic("demo.actor")

    _handle = await (
        AgentBuilder(BasicAgent("actor_demo", "Answer briefly."))
        .llm(llm)
        .memory(SlidingWindowMemory(window_size=20))
        .build_actor(runtime, topics=[topic])
    )

    events = environment.event_stream()
    environment.run()
    await runtime.publish(topic, "Say hello from the actor.")

    async for event in events:
        print(event)
        if isinstance(event, TaskComplete):
            break


if __name__ == "__main__":
    asyncio.run(main())
