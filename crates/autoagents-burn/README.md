# LLM Inference engine for AutoAgents using Burn

---

### ⚠️ Experimental Feature Notice

**Disclaimer:** This project is currently in an **experimental** phase and is under active development. Features, APIs,
and internal logic are subject to change without notice. Stability and performance are not yet guaranteed.

We welcome feedback and contributions, but please use with caution in production environments. Expect breaking changes
and incomplete functionality as we iterate and improve the inference engine.
---

This is an inference engine implementing necessary traits for AutoAgents LLM Provider using the Burn.
This is currently experimental and plan is to mature this along with Burn

The package aims to be a cross-compile capable LLM provider to run LLM's on WebGpu, CUDA, RoCm etc.

Currently, the Burn Team is optimizing the Quantization support. Once we have that we can potentially use WebGPU
compilation for SLM's.

The code in this is taken from Burn-Lm (https://github.com/tracel-ai/burn-lm) and repurpsed to work with wasm-builds.
Later when Burn Team supports native WASM Build we should replace the models with Burn-LM.