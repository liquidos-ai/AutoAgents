# Providers Examples

This directory contains examples demonstrating how to use different LLM providers with AutoAgents.

## Available providers In This Example

- **OpenAI**: GPT models from OpenAI
- **OpenAI Responses**: OpenAI using the Responses API explicitly
- **Anthropic**: Claude models from Anthropic
- **Groq**: Fast inference with various open-source models
- **OpenRouter**: Access to multiple models through one API
- **Requesty**: Access to multiple models through one API
- **Opper**: Access to multiple models through one API
- **Ollama**: Local inference with open-source models

## Running the Examples

Each backend requires different setup:

### OpenAI

```bash
export OPENAI_API_KEY=your_api_key_here
cargo run --package providers -- --backend open-ai
```

### OpenAI Responses

```bash
export OPENAI_API_KEY=your_api_key_here
cargo run --package providers -- --backend open-ai-responses
```

### Anthropic

```bash
export ANTHROPIC_API_KEY=your_api_key_here
cargo run --package providers -- --backend anthropic
```

### Groq

```bash
export GROQ_API_KEY=your_api_key_here
cargo run --package providers -- --backend groq
```

### OpenRouter

```bash
export OPENROUTER_API_KEY=your_api_key_here
cargo run --package providers -- --backend open-router
```

### Requesty

```bash
export REQUESTY_API_KEY=your_api_key_here
cargo run --package providers -- --backend requesty
```

### Opper

```bash
export OPPER_API_KEY=your_api_key_here
cargo run --package providers -- --backend opper
```

### Ollama

First, install and start Ollama locally:

```bash
# Install Ollama (see https://ollama.ai for installation instructions)
ollama pull llama3.2:3b  # Pull the model used in the example
ollama serve              # Start the Ollama server

# Then run the example
cargo run --package providers -- --backend ollama
```

## Backend-Specific Notes

- **OpenAI**: Uses the OpenAI backend defaults
- **OpenAI Responses**: Explicitly selects `OpenAIApiMode::Responses`
- **Anthropic**: Uses Claude 3.5 Sonnet model by default
- **Groq**: Uses Llama 3.3 70B model for fast inference
- **OpenRouter**: Uses a free Gemini model by default
- **Requesty**: Uses `openai/gpt-4o-mini` by default; set `.base_url("https://router.eu.requesty.ai/v1")` for EU routing
- **Opper**: Uses `claude-sonnet-4-6` by default; a `provider/model` id such as `anthropic/claude-sonnet-4-6` pins one route
- **Ollama**: Requires local installation and model download

You can modify the model names and other parameters in each backend's source file.
