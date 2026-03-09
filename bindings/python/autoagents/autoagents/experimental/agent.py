"""Experimental agent APIs that opt into Python-backed execution paths."""

from __future__ import annotations

from ..agent import ActorAgentHandle, AgentHandle, CustomExecutor
from ..agent import _BaseAgent, _BaseAgentBuilder


class ExperimentalAgentBuilder(_BaseAgentBuilder):
    def __init__(self, agent: object) -> None:
        super().__init__(agent, allow_experimental=True)


class Agent(_BaseAgent):
    def __init__(
        self,
        agent: object,
        *,
        llm,
        memory: object | None = None,
        output: type[object] | None = None,
    ) -> None:
        super().__init__(
            agent,
            llm=llm,
            memory=memory,
            output=output,
            allow_experimental=True,
        )


__all__ = [
    "Agent",
    "ExperimentalAgentBuilder",
    "AgentHandle",
    "ActorAgentHandle",
    "CustomExecutor",
]
