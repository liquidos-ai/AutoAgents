# Speech Examples

Four runnable examples in one binary, selected at runtime with `--usecase`:

| Usecase | Description |
|---------|-------------|
| `vad`   | VAD-driven Speech-to-Text (Silero VAD + Parakeet Nemotron) |
| `stt`   | Transcribe a WAV file (Parakeet Nemotron) |
| `tts`   | Synthesise speech from text (Pocket-TTS) |
| `agent` | Full voice loop: VAD → STT → LLM Agent → TTS |

## Prerequisites

Set `HF_TOKEN` or `HUGGINGFACE_HUB_TOKEN` if the repository requires authentication.

## Building

No feature flags are needed — all examples are always compiled:

```bash
cargo build -p speech-examples --release
```

Add `--features cuda` to enable CUDA/TensorRT acceleration for Parakeet.

## Usage

### VAD + STT — microphone

```bash
cargo run -p speech-examples --release -- --usecase vad --input mic --max-seconds 30
```

### VAD + STT — WAV file

```bash
cargo run -p speech-examples --release -- --usecase vad --input file --audio-file /path/to/audio.wav
```

### STT only — WAV file (Nemotron)

```bash
cargo run -p speech-examples --release -- --usecase stt --audio-file /path/to/audio.wav
```

### TTS only

```bash
cargo run -p speech-examples --release -- --usecase tts --text "Hello from AutoAgents"
```

To download sample audio
```bash
wget https://dldata-public.s3.us-east-2.amazonaws.com/2086-149220-0033.wav -O sample.wav
```

Use `--voice` to select a Pocket-TTS voice (default: `alba`) and `--output-file` to save the audio to a WAV file.

### Agent loop (VAD → STT → LLM → TTS)

```bash
export OPENAI_API_KEY=sk...
cargo run -p speech-examples --release -- --usecase agent --input mic --max-seconds 60
```

| Flag | Default | Description |
|------|---------|-------------|
| `--input` | `mic` | `mic` or `file` |
| `--audio-file` | — | WAV file path (required for `--input file` and `stt`) |
| `--max-seconds` | `30` | Stop mic capture after this many seconds (0 = run until Ctrl+C) |
| `--language` | — | Language hint passed to the STT model |
| `--text` | built-in | Text to synthesise (TTS only) |
| `--output-file` | — | Save TTS output to a WAV file |
| `--agent-model` | `gpt-4o-mini` | OpenAI model used by the agent |
| `--voice` | `alba` | Pocket-TTS voice identifier |

## Notes

- Input audio is automatically resampled to 16 kHz mono before VAD/STT processing.
- The segmenter is tuned for low-latency real-time use: 30 ms VAD windows, 450 ms silence timeout, conservative speech/silence thresholds.
- Agent mic input uses Parakeet Nemotron streaming with VAD end-of-utterance detection (English only). Ensure the `nemotron-speech-streaming-en-0.6b` model directory is available in the HuggingFace repo.
- The agent example requires `OPENAI_API_KEY` to be set.
- Say "goodbye" or "stop" to trigger the agent's `exit_conversation` tool and end the session cleanly.

FYI: The Parakeet ONNX models (downloaded separately from HuggingFace) by NVIDIA. This library does not distribute the models.
