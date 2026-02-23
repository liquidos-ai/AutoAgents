# Speech Examples

AutoAgents speech examples demonstrating Text-to-Speech (TTS) and Speech-to-Text (STT) capabilities.

## Usage

### Text-to-Speech (TTS)

Basic generation:

```bash
cargo run -p speech-examples --release -- --usecase basic --output
```

Real-time streaming:

```bash
cargo run -p speech-examples --release -- --usecase realtime --output
```

### Speech-to-Text (STT) - Parakeet Provider

The Parakeet provider supports multiple transcription modes with different model variants:

#### Prerequisites

Download the Parakeet models from altunenes/parakeet-rs Huggingface repository:
```bash
huggingface-cli download altunenes/parakeet-rs --local-dir path_to_parakeet_models
```

#### Example Audio File

For testing, you can download the sample audio:

```bash
wget https://github.com/thewh1teagle/pyannote-rs/releases/download/v0.1.0/6_speakers.wav
export AUDIO_FILE_PATH=path_to_6_speakers.wav
```

#### Available Modes

**1. File Transcription (Non-Streaming with Timestamps)**

Uses TDT model to transcribe audio files with word-level timestamps:

```bash
export PARAKEET_TDT_PATH=path_to_parakeet_models/tdt
export AUDIO_FILE_PATH=path_to_6_speakers.wav

cargo run -p speech-examples --release --features parakeet -- \
  --usecase parakeet --mode file
```

**2. File Transcription (Streaming)**

Uses Nemotron model for streaming transcription with punctuation:

```bash
export PARAKEET_NEMOTRON_PATH=path_to_parakeet_models/nemotron-speech-streaming-en-0.6b
export AUDIO_FILE_PATH=path_to_6_speakers.wav

cargo run -p speech-examples --release --features parakeet -- \
  --usecase parakeet --mode file-stream
```

**3. Microphone Transcription (Non-Streaming)**

Captures 5 seconds of audio from microphone and transcribes with TDT:

```bash
export PARAKEET_TDT_PATH=path_to_parakeet_models/tdt

cargo run -p speech-examples --release --features parakeet -- \
  --usecase parakeet --mode mic
```

**4. Microphone Transcription (Streaming with EOU Detection)**

Real-time transcription from microphone with automatic end-of-utterance detection:

```bash
export PARAKEET_EOU_PATH=path_to_parakeet_models/realtime_eou_120m-v1-onnx

cargo run -p speech-examples --release --features parakeet -- \
  --usecase parakeet --mode mic-stream

# Enable verbose mode to see detailed processing info:
VERBOSE=1 cargo run -p speech-examples --release --features parakeet -- \
  --usecase parakeet --mode mic-stream
```

**Note**: The streaming mode shows visual feedback:
- **Normal mode**: Displays dots (`.`) to indicate audio is being processed, then shows transcription as it's recognized
- **Verbose mode** (`VERBOSE=1` or `DEBUG=1`): Shows detailed information for each audio chunk including sample count, text output, confidence scores, and EOU detection

**5. Run All Examples**

Run all four transcription modes sequentially:

```bash
export PARAKEET_TDT_PATH=path_to_parakeet_models/tdt
export PARAKEET_NEMOTRON_PATH=path_to_parakeet_models/nemotron-speech-streaming-en-0.6b
export PARAKEET_EOU_PATH=path_to_parakeet_models/realtime_eou_120m-v1-onnx
export AUDIO_FILE_PATH=path_to_6_speakers.wav

cargo run -p speech-examples --release --features parakeet -- \
  --usecase parakeet --mode all
```

#### Model Capabilities Comparison

| Model | Streaming | Timestamps | Languages | EOU Detection | Chunk Size |
|-------|-----------|------------|-----------|---------------|------------|
| TDT | ✗ | ✓ | 25 | ✗ | N/A |
| Nemotron | ✓ | ✗ | English | ✗ | 8960 samples (560ms) |
| EOU | ✓ | ✗ | English | ✓ | 2560 samples (160ms) |

## Notes

### TTS Notes
- Set `HF_TOKEN` if you get a 401 from HuggingFace
- First run downloads ~100MB model from HuggingFace

### STT Notes
- STT examples require the `parakeet` feature flag
- All models require 16kHz mono audio input
- Models must be downloaded separately from altunenes/parakeet-rs huggingface repository
- GPU acceleration supported via `--execution-provider` (cpu, cuda, tensorrt, directml, coreml)
- Microphone examples require a working audio input device
