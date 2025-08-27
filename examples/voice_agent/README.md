## Voice Agent

This example uses Kokoros for TTS and Whisper for STT and AutoAgents for LLM agent

```shell
cargo run --package voice-agent-example -- --text -o ./examples/voice_agent/data/output.wav
```

## DOwnlaod the base model and move it into models folder in the voice_agent example

```shell
wget https://huggingface.co/ggerganov/whisper.cpp/resolve/main/ggml-base.en.bin
```

## Run audio file input and Speaker output

```shell
cargo run --package voice-agent-example file --input examples/voice_agent/data/input.wav
```

## Realtime audio input with microphone and speaker output from agent

```shell
cargo run --package voice-agent-example realtime
```

#### Note

This is not production ready code at all, Its mostly written using Claude for rapid testing of voice agents.