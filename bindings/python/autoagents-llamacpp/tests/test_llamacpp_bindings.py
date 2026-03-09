from __future__ import annotations

import pytest

import autoagents_llamacpp as aal
from autoagents_llamacpp import _autoagents_llamacpp as native_llamacpp


def test_llamacpp_package_exports_and_build_info():
    assert set(aal.__all__) == {"LlamaCppBuilder", "backend_build_info"}

    info = aal.backend_build_info()
    assert info["backend"] == "llamacpp"
    assert info["mtmd"] is True
    assert {"cuda", "cuda_no_vmm", "metal", "vulkan"}.issubset(info.keys())


def test_llamacpp_builder_parses_extra_body_json():
    builder = aal.LlamaCppBuilder()
    assert builder.extra_body_json('{"temperature": 0.1}') is builder

    with pytest.raises(RuntimeError, match="invalid extra_body_json"):
        builder.extra_body_json("{")


def test_llamacpp_package_reexports_native_builder():
    assert aal.LlamaCppBuilder is native_llamacpp.LlamaCppBuilder


@pytest.mark.asyncio
async def test_llamacpp_builder_validation_branches():
    with pytest.raises(RuntimeError, match="either model_path .* or repo_id"):
        await aal.LlamaCppBuilder().build()

    with pytest.raises(RuntimeError, match="set only one source"):
        await aal.LlamaCppBuilder().model_path("model.gguf").repo_id("repo").build()

    with pytest.raises(RuntimeError, match="invalid reasoning_format"):
        await aal.LlamaCppBuilder().repo_id("repo").reasoning_format("bad-format").build()

    with pytest.raises(RuntimeError, match="invalid split_mode"):
        await aal.LlamaCppBuilder().repo_id("repo").split_mode("diagonal").build()
