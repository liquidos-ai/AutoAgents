from __future__ import annotations

import pytest

import autoagents_py as aa


@pytest.fixture
def built_llm() -> aa.LLMProvider:
    return aa.LLMBuilder("openai").api_key("test-key").model("gpt-4o-mini").build()
