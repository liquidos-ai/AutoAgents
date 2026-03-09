from __future__ import annotations

import pytest

import autoagents_mistral_rs as aam
from autoagents_mistral_rs import _autoagents_mistral_rs as native_mistralrs


def test_mistralrs_package_exports_and_build_info():
    assert set(aam.__all__) == {"MistralRsBuilder", "backend_build_info"}

    info = aam.backend_build_info()
    assert info["backend"] == "mistral-rs"
    assert {"cuda", "cudnn", "metal", "flash_attn", "accelerate", "mkl", "nccl", "ring"}.issubset(
        info.keys()
    )


def test_mistralrs_package_reexports_native_builder():
    assert aam.MistralRsBuilder is native_mistralrs.MistralRsBuilder


@pytest.mark.asyncio
async def test_mistralrs_builder_validation_branches():
    with pytest.raises(RuntimeError, match="invalid model_type"):
        await aam.MistralRsBuilder().model_type("audio").build()

    with pytest.raises(RuntimeError, match="for GGUF source set both model_dir and gguf_files"):
        await aam.MistralRsBuilder().model_dir("/models").build()

    with pytest.raises(RuntimeError, match="invalid isq_type"):
        await aam.MistralRsBuilder().repo_id("repo").isq_type("not-real").build()
