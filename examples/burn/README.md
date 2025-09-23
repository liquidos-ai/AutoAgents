## Burn LLM Provider

This examples demonstrates the use of Burn for Running Local LLMs with AutoAgents.

##### For Running with Pretrained Model Weights AutoDownload

```shell
cargo run -p autoagents-burn-example --features cuda pretrained
```

```shell
cargo run -p autoagents-burn-example --features cuda pretrained --model llama3
```

##### For Running From Model Files (Need to Download them before Running this)

```shell
cargo run --package autoagents-burn-example --features cuda from-file
```
