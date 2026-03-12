"""Standalone mistral-rs bindings for AutoAgents."""

from ._autoagents_mistral_rs import MistralRsBuilder, backend_build_info

__all__ = ["MistralRsBuilder", "backend_build_info"]
