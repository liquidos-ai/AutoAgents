# Speech Examples

AutoAgents speech examples.

## Usage

Basic generation:

```
cargo run -p speech-examples --release -- --usecase basic --output
```

Real-time streaming:

```
cargo run -p speech-examples --release -- --usecase realtime --output
```

## Notes
- Set `HF_TOKEN` if you get a 401 from HuggingFace.
