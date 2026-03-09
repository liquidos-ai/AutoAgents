"""Guardrails package for AutoAgents Python bindings."""

from __future__ import annotations

from enum import Enum

from ._autoagents_guardrails import (
    Guardrails,
    GuardrailsBuilder,
    PromptInjectionGuard,
    RegexPiiRedactionGuard,
    ToxicityGuard,
)


class EnforcementPolicy(str, Enum):
    BLOCK = "block"
    SANITIZE = "sanitize"
    AUDIT = "audit"


__all__ = [
    "EnforcementPolicy",
    "Guardrails",
    "GuardrailsBuilder",
    "PromptInjectionGuard",
    "RegexPiiRedactionGuard",
    "ToxicityGuard",
]
