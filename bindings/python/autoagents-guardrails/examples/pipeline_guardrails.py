"""Example: combine an LLM pipeline with the separate guardrails package."""

import asyncio
import os
import sys
from pathlib import Path
from time import perf_counter

GUARDRAILS_ROOT = Path(__file__).resolve().parents[1]
AUTOAGENTS_ROOT = GUARDRAILS_ROOT.parent / "autoagents"

for package_root in (GUARDRAILS_ROOT, AUTOAGENTS_ROOT):
    if str(package_root) not in sys.path:
        sys.path.insert(0, str(package_root))

from autoagents_py import Agent, CacheLayer, LLMBuilder, PipelineBuilder
from autoagents_py.prebuilt import BasicAgent, SlidingWindowMemory

from autoagents_guardrails import (
    EnforcementPolicy,
    GuardrailsBuilder,
    RegexPiiRedactionGuard,
)


async def main() -> None:
    api_key = os.environ.get("OPENAI_API_KEY")
    if not api_key:
        raise RuntimeError("OPENAI_API_KEY is required")

    base = LLMBuilder("openai").api_key(api_key).model("gpt-4o-mini").build()
    guardrails = (
        GuardrailsBuilder()
        .input_guard(RegexPiiRedactionGuard())
        .enforcement_policy(EnforcementPolicy.SANITIZE)
        .build()
    )
    llm = (
        PipelineBuilder(base)
        .add_layer(CacheLayer(ttl_seconds=300))
        .add_layer(guardrails)
        .build()
    )

    agent = Agent(
        BasicAgent("guardrails_demo", "Summarize the input in one sentence."),
        llm=llm,
        memory=SlidingWindowMemory(window_size=20),
    )

    prompt = "My email is test@example.com. Summarize this request."

    started_at = perf_counter()
    result = await agent.run(prompt)
    first_elapsed = perf_counter() - started_at
    print(result["response"])
    print(f"first run: {first_elapsed:.3f}s")

    started_at = perf_counter()
    result = await agent.run(prompt)
    cached_elapsed = perf_counter() - started_at
    print(result["response"])
    print(f"cached run: {cached_elapsed:.3f}s")


if __name__ == "__main__":
    asyncio.run(main())
