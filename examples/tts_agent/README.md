# TTS Integration Example

This example demonstrates how to integrate Text-to-Speech (TTS) capabilities into an AutoAgents agent using the Pocket-TTS provider.

## Prerequisites

Before running this example, you need:

1. **Internet connection** - To download the TTS model from HuggingFace (~100MB)
2. **Disk space** - At least 500MB free for model and cache
3. **HuggingFace Access** - Models are public but may require authentication

### Setting up HuggingFace Token (if needed)

If you get a 401 error, you may need a HuggingFace token:

```bash
# Get your token from https://huggingface.co/settings/tokens
export HF_TOKEN=your_huggingface_token_here
```

## Running the Example

```bash
cargo run -p tts-agent-example
```

On first run, the model will be downloaded and cached at `~/.cache/pocket-tts/`. Subsequent runs will use the cached model.

## What It Does

1. Initializes a Pocket-TTS provider
2. Creates an agent with TTS capabilities enabled
3. Configures the agent with:
   - TTS Mode: TextAndAudio (generates both text and audio)
   - Storage Policy: OutputOnly (only stores generated audio)
   - Default Voice: Alba (female, French accent)
4. Sends a message to the agent
5. Agent processes the message and generates an audio response
6. Saves the audio to `agent_response.wav`

## TTS Configuration Options

### TTS Modes

- `Disabled`: No TTS generation
- `TextAndAudio`: Generate both text and audio responses
- `AudioOnly`: Only generate audio (text still in history)
- `OnDemand`: Generate audio only when explicitly requested

### Audio Storage Policies

- `None`: Don't store audio in message history
- `OutputOnly`: Store only agent-generated audio
- `HistoryOnly`: Store all audio but not in outputs
- `Full`: Store all audio everywhere

### Predefined Voices

Available voices in Pocket-TTS:
- `alba` - Female, French
- `marius` - Male, French
- `javert` - Male, dramatic
- `jean` - Male, warm
- `fantine` - Female, gentle
- `cosette` - Female, young
- `eponine` - Female, expressive
- `azelma` - Female, soft

## Requirements

- First run will download the TTS model from HuggingFace (~100MB)
- Model is cached in `~/.cache/pocket-tts/` for subsequent runs
- Internet connection required for initial download
- May require HuggingFace authentication (set HF_TOKEN environment variable)

## Troubleshooting

**401 Error**: If you see a "status code 401" error:
```bash
export HF_TOKEN=your_huggingface_token
cargo run -p tts-agent-example
```

Get your token from: https://huggingface.co/settings/tokens

**Network Error**: Ensure you have internet connectivity for the initial model download.

## Notes

- The example uses the library backend (local model execution)
- Audio is generated at 24kHz sample rate
- Generated audio files are in WAV format (32-bit float PCM)
