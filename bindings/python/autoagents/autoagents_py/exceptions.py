"""Structured exception types for the AutoAgents Python API."""

from __future__ import annotations


class AutoAgentsError(RuntimeError):
    """Base class for all AutoAgents errors."""


class AgentConfigError(AutoAgentsError):
    """Raised when user-supplied configuration or input is invalid."""


class AgentBuildError(AutoAgentsError):
    """Raised when agent construction fails (missing LLM, invalid config, …)."""


class AgentRunError(AutoAgentsError):
    """Raised when an agent run fails at execution time."""


class AgentTimeoutError(AgentRunError):
    """Raised when ``Agent.run()`` exceeds the specified timeout."""


class ExperimentalFeatureError(AgentConfigError):
    """Raised when an experimental extension path is used through the stable API."""


class ToolExecutionError(AgentRunError):
    """Raised when a tool call fails during execution."""


class LLMProviderError(AutoAgentsError):
    """Raised when the LLM provider is misconfigured or returns an error."""
