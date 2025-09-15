# LLM Inference engine for AutoAgents using Burn

! Experimental

This is an inference engine implementing necessary traits for AutoAgents LLM Provider using the Burn.
This is currently experimental and plan is to mature this along with Burn

The package aims to be a cross-compile capable LLM provider to run LLM's on WebGpu, CUDA, RoCm etc.

Currently, the Burn Team is optimizing the Quantization support. Once we have that we can potentially use WebGPU
compilation for SLM's.

The code in this is taken from Burn-Lm (https://github.com/tracel-ai/burn-lm) and repurpsed to work with wasm-builds.
Later is Burn Team supports native WASSM Build we should replace the models.