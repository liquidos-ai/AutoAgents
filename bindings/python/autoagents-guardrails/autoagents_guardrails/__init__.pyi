from enum import Enum

from autoagents_py import LLMProvider


class EnforcementPolicy(str, Enum):
    BLOCK = "block"
    SANITIZE = "sanitize"
    AUDIT = "audit"


class PromptInjectionGuard:
    def __init__(self) -> None: ...


class RegexPiiRedactionGuard:
    def __init__(self) -> None: ...


class ToxicityGuard:
    def __init__(self) -> None: ...


class Guardrails:
    def build(self, provider: LLMProvider) -> LLMProvider: ...
    def wrap(self, provider: LLMProvider) -> LLMProvider: ...


class GuardrailsBuilder:
    def __init__(self) -> None: ...
    def input_guard(
        self,
        guard: PromptInjectionGuard | RegexPiiRedactionGuard,
    ) -> GuardrailsBuilder: ...
    def output_guard(self, guard: ToxicityGuard) -> GuardrailsBuilder: ...
    def enforcement_policy(self, policy: EnforcementPolicy | str) -> GuardrailsBuilder: ...
    def build(self) -> Guardrails: ...
