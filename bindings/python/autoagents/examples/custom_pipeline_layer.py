"""Example: add a small custom Python layer to an LLM pipeline."""

import asyncio
import os
import sys
from pathlib import Path

PACKAGE_ROOT = Path(__file__).resolve().parents[1]
if str(PACKAGE_ROOT) not in sys.path:
    sys.path.insert(0, str(PACKAGE_ROOT))

from autoagents import Agent, LLMBuilder, PipelineBuilder, PipelineLayer, RetryLayer
from autoagents.prebuilt import BasicAgent, SlidingWindowMemory


class PrefixLayer(PipelineLayer):
    async def chat(self, next_provider, messages, schema=None):
        response = await next_provider.chat(messages, schema)
        text = response.get("text")
        if text:
            response["text"] = f"[layered] {text}"
        return response


async def main() -> None:
    api_key = os.environ.get("OPENAI_API_KEY")
    if not api_key:
        raise RuntimeError("OPENAI_API_KEY is required")

    base = LLMBuilder("openai").api_key(api_key).model("gpt-4o-mini").build()
    llm = (
        PipelineBuilder(base)
        .add_layer(RetryLayer(max_attempts=2))
        .add_layer(PrefixLayer())
        .build()
    )

    agent = Agent(
        BasicAgent("pipeline_demo", "Reply in one sentence."),
        llm=llm,
        memory=SlidingWindowMemory(window_size=20),
    )
    result = await agent.run("Say hello.")
    print(result["response"])


if __name__ == "__main__":
    asyncio.run(main())
