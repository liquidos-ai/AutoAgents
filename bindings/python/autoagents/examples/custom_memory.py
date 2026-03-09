"""Example: custom Python memory callbacks backed by a simple list."""

import asyncio
import os
import sys
from pathlib import Path
from typing import List, Optional

PACKAGE_ROOT = Path(__file__).resolve().parents[1]
if str(PACKAGE_ROOT) not in sys.path:
    sys.path.insert(0, str(PACKAGE_ROOT))

from autoagents import LLMBuilder
from autoagents.experimental import Agent
from autoagents.prebuilt import BasicAgent
from autoagents.types import ChatMessagePayload


class ListMemory:
    def __init__(self) -> None:
        self.messages: List[ChatMessagePayload] = []

    async def remember(self, message: ChatMessagePayload) -> None:
        self.messages.append(message)

    async def recall(
        self,
        query: str,
        limit: Optional[int] = None,
    ) -> List[ChatMessagePayload]:
        return self.messages

    async def clear(self) -> None:
        self.messages.clear()

    def size(self) -> int:
        return len(self.messages)


async def main() -> None:
    api_key = os.environ.get("OPENAI_API_KEY")
    if not api_key:
        raise RuntimeError("OPENAI_API_KEY is required")

    llm = LLMBuilder("openai").api_key(api_key).model("gpt-4o-mini").build()
    memory = ListMemory()
    agent = Agent(
        BasicAgent("memory_demo", "You are a simple agent."),
        llm=llm,
        memory=memory,
    )

    result = await agent.run("My name is Tess")
    print(result["response"])
    print(f"stored_messages={memory.size()}")

    result = await agent.run("Whats my name?")
    print(result["response"])


if __name__ == "__main__":
    asyncio.run(main())
