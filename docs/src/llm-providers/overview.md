# LLM Providers Overview

AutoAgents supports a wide range of Large Language Model (LLM) providers, allowing you to choose the best fit for your specific use case. This document provides an overview of the supported providers and how to use them.

## Supported Providers

AutoAgents currently supports the following LLM providers:

| Provider | Status | Models | Features |
|----------|--------|--------|----------|
| **OpenAI** | ✅ | GPT-4, GPT-3.5-turbo | Function calling, streaming, vision |
| **Anthropic** | ✅ | Claude 3.5 Sonnet, Claude 3 Opus/Haiku | Tool use, long context |
| **Ollama** | ✅ | Llama 3, Mistral, CodeLlama | Local inference, custom models |
| **Google** | ✅ | Gemini Pro, Gemini Flash | Multimodal, fast inference |
| **Groq** | ✅ | Llama 3, Mixtral | Ultra-fast inference |
| **DeepSeek** | ✅ | DeepSeek Coder, DeepSeek Chat | Code-specialized models |
| **xAI** | ✅ | Grok-1, Grok-2 | Real-time information |
| **Phind** | ✅ | CodeLlama variants | Developer-focused |
| **Azure OpenAI** | ✅ | GPT-4, GPT-3.5 | Enterprise features |

## Common Interface

All LLM providers implement the `LLMProvider` trait, providing a consistent interface:

```rust
#[async_trait]
pub trait LLMProvider: Send + Sync {
    /// Generate a response from the language model
    async fn generate(
        &self,
        messages: Vec<ChatMessage>,
        tools: Option<Vec<Tool>>,
        temperature: Option<f32>,
    ) -> Result<LLMResponse, LLMError>;
    
    /// Check if the provider is available
    fn is_available(&self) -> bool;
    
    /// Get provider information
    fn get_info(&self) -> ProviderInfo;
    
    /// Get supported models
    fn get_models(&self) -> Vec<String>;
}
```

## Provider Configuration

### Feature Flags

Enable specific providers using Cargo features:

```toml
[dependencies]
autoagents = { 
    version = "0.2.0-alpha.0", 
    features = ["openai", "anthropic", "ollama"]
}
```

### Environment Variables

Most providers require API keys set as environment variables:

```bash
# OpenAI
export OPENAI_API_KEY="your-openai-key"

# Anthropic
export ANTHROPIC_API_KEY="your-anthropic-key"

# Google
export GOOGLE_API_KEY="your-google-key"

# Groq
export GROQ_API_KEY="your-groq-key"

# DeepSeek
export DEEPSEEK_API_KEY="your-deepseek-key"

# xAI
export XAI_API_KEY="your-xai-key"

# Azure OpenAI
export AZURE_OPENAI_API_KEY="your-azure-key"
export AZURE_OPENAI_ENDPOINT="https://your-resource.openai.azure.com"
```

## Basic Usage

### Creating a Provider

```rust
use autoagents::llm::{OpenAIProvider, AnthropicProvider, OllamaProvider};
use std::sync::Arc;

// OpenAI
let openai = Arc::new(OpenAIProvider::new("your-api-key")?);

// Anthropic
let anthropic = Arc::new(AnthropicProvider::new("your-api-key")?);

// Ollama (local)
let ollama = Arc::new(OllamaProvider::new("http://localhost:11434")?);
```

### Using with Agents

```rust
use autoagents::core::agent::base::AgentBuilder;

let agent = AgentBuilder::new(MyAgent {})
    .with_llm(openai)  // or anthropic, ollama, etc.
    .build()?;
```

## Provider-Specific Features

### OpenAI

```rust
let openai = OpenAIProvider::builder()
    .api_key("your-key")
    .model("gpt-4")
    .temperature(0.7)
    .max_tokens(1000)
    .build()?;
```

**Features:**
- Function calling for tool integration
- Streaming responses
- Vision capabilities (GPT-4V)
- Fine-tuned models support

### Anthropic

```rust
let anthropic = AnthropicProvider::builder()
    .api_key("your-key")
    .model("claude-3-5-sonnet-20241022")
    .max_tokens(4000)
    .build()?;
```

**Features:**
- Tool use for function calling
- Long context windows (up to 200K tokens)
- Constitutional AI training
- Safety-focused responses

### Ollama

```rust
let ollama = OllamaProvider::builder()
    .base_url("http://localhost:11434")
    .model("llama3")
    .temperature(0.8)
    .build()?;
```

**Features:**
- Local inference (no API calls)
- Custom model support
- Privacy-focused
- Offline capabilities

### Google

```rust
let google = GoogleProvider::builder()
    .api_key("your-key")
    .model("gemini-pro")
    .build()?;
```

**Features:**
- Multimodal capabilities
- Fast inference
- Integrated search
- Safety filters

## Advanced Configuration

### Custom Model Settings

```rust
let provider = OpenAIProvider::builder()
    .api_key("your-key")
    .model("gpt-4")
    .temperature(0.7)
    .max_tokens(2000)
    .top_p(0.9)
    .frequency_penalty(0.0)
    .presence_penalty(0.0)
    .build()?;
```

### Timeout Configuration

```rust
let provider = OpenAIProvider::builder()
    .api_key("your-key")
    .timeout(Duration::from_secs(30))
    .build()?;
```

### Retry Logic

```rust
let provider = OpenAIProvider::builder()
    .api_key("your-key")
    .max_retries(3)
    .retry_delay(Duration::from_secs(1))
    .build()?;
```

## Error Handling

### Common Errors

```rust
use autoagents::llm::LLMError;

match provider.generate(messages, None, None).await {
    Ok(response) => println!("Success: {}", response.content),
    Err(LLMError::ApiKey) => eprintln!("Invalid API key"),
    Err(LLMError::RateLimit) => eprintln!("Rate limit exceeded"),
    Err(LLMError::NetworkError(e)) => eprintln!("Network error: {}", e),
    Err(LLMError::InvalidModel(model)) => eprintln!("Invalid model: {}", model),
    Err(e) => eprintln!("Other error: {}", e),
}
```

### Graceful Degradation

```rust
pub struct FallbackProvider {
    primary: Arc<dyn LLMProvider>,
    fallback: Arc<dyn LLMProvider>,
}

impl FallbackProvider {
    async fn generate(&self, messages: Vec<ChatMessage>) -> Result<LLMResponse, LLMError> {
        match self.primary.generate(messages.clone(), None, None).await {
            Ok(response) => Ok(response),
            Err(_) => self.fallback.generate(messages, None, None).await,
        }
    }
}
```

## Provider Comparison

### Performance

| Provider | Latency | Throughput | Cost |
|----------|---------|------------|------|
| Groq | Very Low | Very High | Medium |
| OpenAI | Low | High | High |
| Anthropic | Medium | Medium | High |
| Ollama | Variable | Variable | Free |
| Google | Low | High | Low |

### Use Cases

**OpenAI**: Best for general-purpose applications requiring high-quality responses and function calling.

**Anthropic**: Ideal for safety-critical applications and tasks requiring long context.

**Ollama**: Perfect for privacy-sensitive applications and offline usage.

**Google**: Great for multimodal applications and cost-effective solutions.

**Groq**: Excellent for real-time applications requiring ultra-fast responses.

**DeepSeek**: Specialized for coding and development tasks.

## Best Practices

### 1. Provider Selection

Choose providers based on:
- **Latency requirements**: Groq for speed, others for quality
- **Cost constraints**: Google and Ollama for cost-effectiveness
- **Privacy needs**: Ollama for complete privacy
- **Feature requirements**: OpenAI for function calling

### 2. Configuration Management

```rust
use serde::{Deserialize, Serialize};

#[derive(Serialize, Deserialize)]
pub struct LLMConfig {
    pub provider: String,
    pub model: String,
    pub temperature: f32,
    pub max_tokens: u32,
    pub api_key: String,
}

impl LLMConfig {
    pub fn create_provider(&self) -> Result<Arc<dyn LLMProvider>, LLMError> {
        match self.provider.as_str() {
            "openai" => Ok(Arc::new(OpenAIProvider::new(&self.api_key)?)),
            "anthropic" => Ok(Arc::new(AnthropicProvider::new(&self.api_key)?)),
            "ollama" => Ok(Arc::new(OllamaProvider::new("http://localhost:11434")?)),
            _ => Err(LLMError::InvalidProvider(self.provider.clone())),
        }
    }
}
```

### 3. Monitoring and Logging

```rust
use log::{info, warn, error};

pub struct MonitoredProvider {
    inner: Arc<dyn LLMProvider>,
    metrics: Arc<Mutex<ProviderMetrics>>,
}

impl MonitoredProvider {
    async fn generate(&self, messages: Vec<ChatMessage>) -> Result<LLMResponse, LLMError> {
        let start = Instant::now();
        
        match self.inner.generate(messages, None, None).await {
            Ok(response) => {
                let duration = start.elapsed();
                info!("LLM call succeeded in {:?}", duration);
                self.metrics.lock().unwrap().record_success(duration);
                Ok(response)
            }
            Err(e) => {
                error!("LLM call failed: {}", e);
                self.metrics.lock().unwrap().record_error();
                Err(e)
            }
        }
    }
}
```

### 4. Rate Limiting

```rust
use tokio::time::{sleep, Duration};

pub struct RateLimitedProvider {
    inner: Arc<dyn LLMProvider>,
    rate_limiter: Arc<Mutex<RateLimiter>>,
}

impl RateLimitedProvider {
    async fn generate(&self, messages: Vec<ChatMessage>) -> Result<LLMResponse, LLMError> {
        self.rate_limiter.lock().await.acquire().await;
        self.inner.generate(messages, None, None).await
    }
}
```

## Testing with Providers

### Mock Provider

```rust
pub struct MockLLMProvider {
    responses: Vec<String>,
    call_count: Arc<Mutex<usize>>,
}

impl MockLLMProvider {
    pub fn new(responses: Vec<String>) -> Self {
        Self {
            responses,
            call_count: Arc::new(Mutex::new(0)),
        }
    }
}

#[async_trait]
impl LLMProvider for MockLLMProvider {
    async fn generate(
        &self,
        _messages: Vec<ChatMessage>,
        _tools: Option<Vec<Tool>>,
        _temperature: Option<f32>,
    ) -> Result<LLMResponse, LLMError> {
        let mut count = self.call_count.lock().unwrap();
        let response = self.responses.get(*count).unwrap_or(&"Mock response".to_string());
        *count += 1;
        
        Ok(LLMResponse {
            content: response.clone(),
            usage: None,
            model: "mock".to_string(),
        })
    }
    
    fn is_available(&self) -> bool {
        true
    }
    
    fn get_info(&self) -> ProviderInfo {
        ProviderInfo {
            name: "Mock".to_string(),
            version: "1.0.0".to_string(),
        }
    }
    
    fn get_models(&self) -> Vec<String> {
        vec!["mock".to_string()]
    }
}
```

## Future Providers

Planned additions:
- **Cohere**: Command and Embed models
- **Hugging Face**: Inference API integration
- **Replicate**: Community models
- **Together AI**: Open-source models
- **Perplexity**: Search-augmented responses

## Contributing New Providers

To add a new provider:

1. Implement the `LLMProvider` trait
2. Add feature flag to `Cargo.toml`
3. Create provider-specific error types
4. Add comprehensive tests
5. Update documentation

See the [Contributing Guide](../contributing/development-setup.md) for detailed instructions.

The unified provider interface ensures that switching between different LLM providers is seamless, allowing you to choose the best provider for each specific use case while maintaining code compatibility.