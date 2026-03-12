"""Metal variant import path for AutoAgents llama.cpp bindings."""

from ._autoagents_llamacpp_metal import LlamaCppBuilder, backend_build_info

__all__ = ["LlamaCppBuilder", "backend_build_info"]
