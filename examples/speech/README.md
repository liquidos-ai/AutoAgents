# Speech Examples

AutoAgents speech examples demonstrating Text-to-Speech (TTS) and Speech-to-Text (STT) capabilities.

## Usage

### Text-to-Speech (TTS)

Basic generation:

```
cargo run -p speech-examples --release -- --usecase basic --output
```

Real-time streaming:

```
cargo run -p speech-examples --release -- --usecase realtime --output
```

### Speech-to-Text (STT)

Speech-to-text with Parakeet:

```bash
# Set the model path (required)
export PARAKEET_MODEL_PATH=/path/to/parakeet-tdt-0.6b

# Optional: Set an audio file to transcribe
export AUDIO_FILE_PATH=/path/to/audio.wav

# Run the example
cargo run -p speech-examples --release --features parakeet -- --usecase stt
```

**Note:** You need to download the Parakeet model files first. See `crates/autoagents-speech/PARAKEET_STT.md` for detailed instructions.

## Notes
- Set `HF_TOKEN` if you get a 401 from HuggingFace.
- STT examples require the `parakeet` feature flag and model files to be downloaded.
- Audio files for STT must be 16kHz, mono WAV format.
