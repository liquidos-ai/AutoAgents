# WASM Agent Execution

Run Agents on Browser using AutoAgents WASM support with Candle - The current TinyLLama model is chat completions hence
the results are not good. Candle currently does not support WebGPU.

```shell
wasm-pack build --release --target web --out-dir ./pkg
```

```shell
python3 -m http.server
```
