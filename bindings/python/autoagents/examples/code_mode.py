"""Example: CodeAct/code mode execution with typed Python tools."""

import asyncio
import json
import os
import sys
from pathlib import Path
from typing import Optional

PACKAGE_ROOT = Path(__file__).resolve().parents[1]
if str(PACKAGE_ROOT) not in sys.path:
    sys.path.insert(0, str(PACKAGE_ROOT))

from autoagents_py import AgentBuilder, LLMBuilder, Task, tool
from autoagents_py.prebuilt import CodeActAgent, SlidingWindowMemory
from autoagents_py.types import AgentRunResult, JsonValue

DEFAULT_PROMPT = "What is the current time right now?"
AGENT_DESCRIPTION = (
    "You are a general-purpose CodeAct assistant. Solve user requests by "
    "writing one concise TypeScript script. Use standard JavaScript globals "
    "such as Date, Math, JSON, Array, and string utilities for generic "
    "tasks. Use the provided tools when they make the solution clearer, "
    "especially for arithmetic. Log important intermediate values with "
    "console.log when useful. Return a plain string answer, not JSON. The "
    "script must return its final value from the top level with `return ...;` "
    "or a trailing expression. Imports are not available in the sandbox."
)


@tool(
    name="AddNumbers",
    description="Add two integers and return the result",
)
def add_numbers(left: int, right: int) -> int:
    return left + right


@tool(
    name="MultiplyNumbers",
    description="Multiply two integers and return the result",
)
def multiply_numbers(left: int, right: int) -> int:
    return left * right


def format_json_value(value: Optional[JsonValue]) -> str:
    try:
        return json.dumps(value)
    except TypeError:
        return "<invalid json>"


def print_execution_trace(result: AgentRunResult) -> None:
    executions = result.get("executions", [])

    print("\nCodeAct Trace")
    print(f"  Sandbox executions: {len(executions)}")

    for execution in executions:
        tool_calls = execution.get("tool_calls", [])
        print(
            f"\n  - {execution['execution_id']} | success={execution['success']} "
            f"| tool_calls={len(tool_calls)} | duration={execution['duration_ms']}ms"
        )

        console = execution.get("console", [])
        if console:
            print("    Console:")
            for line in console:
                print(f"      {line}")

        if tool_calls:
            print("    Tool Calls:")
            for tool_call in tool_calls:
                print(
                    "      "
                    f"{tool_call.get('tool_name', '<unknown>')} "
                    f"| success={tool_call.get('success', False)} "
                    f"| args={format_json_value(tool_call.get('arguments'))} "
                    f"| result={format_json_value(tool_call.get('result'))}"
                )

        error = execution.get("error")
        if error:
            print(f"    Error: {error}")

        print("    Source:")
        for line in execution.get("source", "").splitlines():
            print(f"      {line}")


async def main() -> None:
    api_key = os.environ.get("OPENAI_API_KEY")
    if not api_key:
        raise RuntimeError("OPENAI_API_KEY is required")

    prompt_args = sys.argv[1:]
    prompt = " ".join(prompt_args) if prompt_args else DEFAULT_PROMPT

    llm = LLMBuilder("openai").api_key(api_key).model("gpt-4o").build()

    executor = (
        CodeActAgent("code_mode_agent", AGENT_DESCRIPTION)
        .tools([add_numbers, multiply_numbers])
        .max_turns(10)
    )

    handle = await (
        AgentBuilder(executor)
        .llm(llm)
        .memory(SlidingWindowMemory(window_size=8))
        .build()
    )

    print(f"Prompt:\n{prompt}\n")

    result = await handle.run(Task(prompt=prompt))

    print("Response")
    print(f"  {result['response']}")

    print_execution_trace(result)


if __name__ == "__main__":
    asyncio.run(main())
