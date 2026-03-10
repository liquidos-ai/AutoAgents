from __future__ import annotations

from enum import Enum

import pytest

import autoagents_py as aa
import autoagents_guardrails as aag


class _PolicyEnum(Enum):
    AUDIT = "audit"


def test_guardrails_python_package_exports():
    assert set(aag.__all__) == {
        "EnforcementPolicy",
        "Guardrails",
        "GuardrailsBuilder",
        "PromptInjectionGuard",
        "RegexPiiRedactionGuard",
        "ToxicityGuard",
    }
    assert aag.EnforcementPolicy.BLOCK.value == "block"


def test_guardrails_repr_and_builder_validation():
    prompt_guard = aag.PromptInjectionGuard()
    pii_guard = aag.RegexPiiRedactionGuard()
    toxicity_guard = aag.ToxicityGuard()

    assert repr(prompt_guard) == "PromptInjectionGuard()"
    assert repr(pii_guard) == "RegexPiiRedactionGuard()"
    assert repr(toxicity_guard) == "ToxicityGuard()"

    builder = (
        aag.GuardrailsBuilder()
        .input_guard(prompt_guard)
        .input_guard(pii_guard)
        .output_guard(toxicity_guard)
        .enforcement_policy(_PolicyEnum.AUDIT)
    )
    guardrails = builder.build()
    assert repr(guardrails) == "Guardrails(<configured>)"

    with pytest.raises(RuntimeError, match="expected one of: block, sanitize, audit"):
        aag.GuardrailsBuilder().enforcement_policy("invalid")

    with pytest.raises(RuntimeError, match="expects PromptInjectionGuard or RegexPiiRedactionGuard"):
        aag.GuardrailsBuilder().input_guard(object())

    with pytest.raises(RuntimeError, match="expects ToxicityGuard"):
        aag.GuardrailsBuilder().output_guard(object())


def test_guardrails_wrap_requires_llm_provider(built_llm):
    guardrails = aag.GuardrailsBuilder().input_guard(aag.PromptInjectionGuard()).build()
    wrapped = guardrails.wrap(built_llm)

    assert "LLMProvider" in repr(wrapped)

    with pytest.raises(RuntimeError, match="expects an AutoAgents LLMProvider"):
        guardrails.build(object())


def test_guardrails_can_build_inside_pipeline_with_cache_layer():
    base = aa.LLMBuilder("openai").api_key("test-key").model("gpt-4o-mini").build()
    guardrails = (
        aag.GuardrailsBuilder()
        .input_guard(aag.RegexPiiRedactionGuard())
        .enforcement_policy(aag.EnforcementPolicy.SANITIZE)
        .build()
    )

    llm = (
        aa.PipelineBuilder(base)
        .add_layer(aa.CacheLayer(ttl_seconds=300))
        .add_layer(guardrails)
        .build()
    )

    assert isinstance(llm, aa.LLMProvider)
    assert "LLMProvider" in repr(llm)
