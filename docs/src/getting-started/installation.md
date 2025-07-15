# Installation

This guide will help you install AutoAgents and set up your development environment.

## Prerequisites

Before installing AutoAgents, ensure you have the following prerequisites:

### Rust Toolchain
AutoAgents requires Rust 1.70 or later. Install Rust using [rustup](https://rustup.rs/):

```bash
curl --proto '=https' --tlsv1.2 -sSf https://sh.rustup.rs | sh
source $HOME/.cargo/env
```

Verify your installation:
```bash
rustc --version
cargo --version
```

### System Dependencies
Depending on your operating system, you may need additional dependencies:

#### Ubuntu/Debian
```bash
sudo apt update
sudo apt install build-essential pkg-config libssl-dev
```

#### macOS
```bash
# Install Xcode command line tools
xcode-select --install

# Or install via Homebrew
brew install pkg-config openssl
```

#### Windows
- Install [Visual Studio Build Tools](https://visualstudio.microsoft.com/downloads/#build-tools-for-visual-studio-2022)
- Or use [chocolatey](https://chocolatey.org/): `choco install visualstudio2022buildtools`

## Installation Methods

### Method 1: Using Cargo (Recommended)

Add AutoAgents to your `Cargo.toml`:

```toml
[dependencies]
autoagents = "0.2.0-alpha.0"
tokio = { version = "1.0", features = ["full"] }
```

For specific LLM provider support, use feature flags:

```toml
[dependencies]
autoagents = { version = "0.2.0-alpha.0", features = ["openai", "anthropic"] }
```

### Method 2: From Source

Clone the repository and build from source:

```bash
git clone https://github.com/liquidos-ai/AutoAgents.git
cd AutoAgents
cargo build --release
```

### Method 3: Using Cargo Install

Install the latest version directly:

```bash
cargo install autoagents
```

## Feature Flags

AutoAgents uses feature flags to enable specific LLM providers and functionality:

### LLM Provider Features
- `openai` - OpenAI GPT models
- `anthropic` - Anthropic Claude models
- `ollama` - Local Ollama models
- `google` - Google Gemini models
- `groq` - Groq inference API
- `deepseek` - DeepSeek models
- `xai` - xAI Grok models
- `phind` - Phind developer models
- `azure_openai` - Azure OpenAI service

### Utility Features
- `full` - All available features
- `logging` - Enhanced logging support

### Example Configuration

```toml
[dependencies]
autoagents = { 
    version = "0.2.0-alpha.0", 
    features = ["openai", "anthropic", "logging"] 
}
```

## Environment Setup

### API Keys
Most LLM providers require API keys. Set them as environment variables:

```bash
# OpenAI
export OPENAI_API_KEY="your-openai-key"

# Anthropic
export ANTHROPIC_API_KEY="your-anthropic-key"

# Google
export GOOGLE_API_KEY="your-google-key"

# Groq
export GROQ_API_KEY="your-groq-key"
```

### Configuration File
Create a `.env` file in your project root:

```env
OPENAI_API_KEY=your-openai-key
ANTHROPIC_API_KEY=your-anthropic-key
RUST_LOG=info
```

## Verification

Create a simple test file to verify your installation:

```rust
// test_installation.rs
use autoagents::prelude::*;

#[tokio::main]
async fn main() -> Result<(), Box<dyn std::error::Error>> {
    println!("AutoAgents installed successfully!");
    
    // Test with a simple agent (requires OpenAI API key)
    #[cfg(feature = "openai")]
    {
        use autoagents::llm::OpenAIProvider;
        
        if let Ok(api_key) = std::env::var("OPENAI_API_KEY") {
            let llm = OpenAIProvider::new(&api_key)?;
            println!("OpenAI provider initialized successfully!");
        }
    }
    
    Ok(())
}
```

Run the test:
```bash
cargo run --bin test_installation
```

## Development Environment

### Recommended Tools
- **IDE**: VS Code with rust-analyzer extension
- **Debugging**: LLDB or GDB for debugging
- **Testing**: Built-in `cargo test`
- **Documentation**: `cargo doc --open`

### VS Code Extensions
- `rust-lang.rust-analyzer` - Rust language support
- `vadimcn.vscode-lldb` - Debugging support
- `serayuzgur.crates` - Cargo.toml management

## Common Issues

### SSL Certificate Issues
If you encounter SSL certificate errors:

```bash
# Update certificates (Ubuntu/Debian)
sudo apt-get update && sudo apt-get install ca-certificates

# macOS
brew install ca-certificates
```

### Compilation Errors
If you see compilation errors:

1. Update Rust: `rustup update`
2. Clear cargo cache: `cargo clean`
3. Check feature flags are correctly specified

### API Key Issues
- Ensure API keys are properly set in environment variables
- Check API key permissions and quotas
- Verify the correct provider feature is enabled

## Docker Setup

For containerized development:

```dockerfile
FROM rust:1.70

WORKDIR /app
COPY . .

RUN cargo build --release

ENV OPENAI_API_KEY=""
ENV ANTHROPIC_API_KEY=""

CMD ["./target/release/your-app"]
```

## Next Steps

Now that you have AutoAgents installed, you can:

1. Follow the [Quick Start](./quick-start.md) guide
2. Create [Your First Agent](./first-agent.md)
3. Explore the [Examples](../examples/basic-weather.md)

## Getting Help

If you encounter issues during installation:

- Check the [FAQ](../appendices/faq.md)
- Visit our [Discord community](https://discord.gg/Ghau8xYn)
- Create an issue on [GitHub](https://github.com/liquidos-ai/AutoAgents/issues)