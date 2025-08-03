//! Main entry point for liquid-edge chat inference demonstration
//!
//! This example demonstrates loading a TinyLlama model and performing
//! proper chat-based text generation with Jinja templates, system prompts,
//! and conservative sampling using the ONNX runtime backend.

use liquid_edge::{
    runtime::{OnnxInput, OnnxRuntime},
    sampling::TopPSampler,
    tokenizer::Tokenizer,
    traits::{GenerationOptions, InferenceRuntime, SamplingStrategy, TokenizerTrait},
    EdgeError, EdgeResult,
};
use minijinja::{context, Environment};
use serde::{Deserialize, Serialize};
use serde_json;
use std::env;
use std::fs;
use std::io::{self, IsTerminal, Read, Write};
use std::path::Path;

/// Model configuration structure
#[derive(Debug, Deserialize, Serialize)]
struct ModelConfig {
    #[serde(default = "default_eos_token_id")]
    eos_token_id: u32,
    #[serde(default = "default_bos_token_id")]
    bos_token_id: u32,
    #[serde(default = "default_max_position_embeddings")]
    max_position_embeddings: usize,
    #[serde(default = "default_max_position_embeddings")]
    n_positions: usize, // GPT2 style naming
    #[serde(default = "default_vocab_size")]
    vocab_size: usize,
}

/// Tokenizer configuration structure
#[derive(Debug, Deserialize, Serialize)]
struct TokenizerConfig {
    #[serde(default)]
    add_bos_token: bool,
    #[serde(default)]
    add_eos_token: bool,
    #[serde(default)]
    bos_token: Option<String>,
    #[serde(default)]
    eos_token: Option<String>,
    #[serde(default = "default_model_max_length")]
    model_max_length: usize,
    #[serde(default)]
    pad_token: Option<String>,
}

/// Special token representation
#[derive(Debug, Clone)]
struct SpecialToken {
    content: String,
    lstrip: bool,
    normalized: bool,
    rstrip: bool,
    single_word: bool,
}

impl SpecialToken {
    fn from_value(value: &serde_json::Value) -> Option<Self> {
        match value {
            serde_json::Value::String(s) => Some(SpecialToken {
                content: s.clone(),
                lstrip: false,
                normalized: false,
                rstrip: false,
                single_word: false,
            }),
            serde_json::Value::Object(obj) => Some(SpecialToken {
                content: obj.get("content")?.as_str()?.to_string(),
                lstrip: obj.get("lstrip").and_then(|v| v.as_bool()).unwrap_or(false),
                normalized: obj
                    .get("normalized")
                    .and_then(|v| v.as_bool())
                    .unwrap_or(false),
                rstrip: obj.get("rstrip").and_then(|v| v.as_bool()).unwrap_or(false),
                single_word: obj
                    .get("single_word")
                    .and_then(|v| v.as_bool())
                    .unwrap_or(false),
            }),
            _ => None,
        }
    }
}

/// Special tokens mapping
#[derive(Debug, Deserialize, Serialize)]
struct SpecialTokensMap {
    #[serde(default)]
    bos_token: Option<serde_json::Value>,
    #[serde(default)]
    eos_token: Option<serde_json::Value>,
    #[serde(default)]
    unk_token: Option<serde_json::Value>,
    #[serde(default)]
    pad_token: Option<serde_json::Value>,
}

// Configuration defaults
fn default_model_max_length() -> usize {
    1024
}
fn default_eos_token_id() -> u32 {
    2
}
fn default_bos_token_id() -> u32 {
    1
}
fn default_max_position_embeddings() -> usize {
    2048
}
fn default_vocab_size() -> usize {
    32000
}

/// Enhanced chat bot with proper chat formatting
pub struct ChatBot {
    runtime: OnnxRuntime,
    tokenizer: Tokenizer,
    sampler: TopPSampler,
    eos_token_id: u32,
    bos_token_id: u32,
    conversation_history: Vec<(String, String)>, // (user, assistant) pairs
    chat_template: String,
    config: ModelConfig,
    tokenizer_config: TokenizerConfig,
    special_tokens_map: SpecialTokensMap,
    eos_token_str: String,
    bos_token_str: String,
    model_name: String,
}

impl ChatBot {
    /// Create a new chat bot instance
    pub fn new<P: AsRef<Path>>(model_dir: P, model_name: String) -> EdgeResult<Self> {
        let model_dir = model_dir.as_ref();
        println!("Initializing {} Chat Bot...", model_name);

        // Define paths
        let tokenizer_path = model_dir.join("tokenizer.json");
        let model_path = model_dir.join("model.onnx");
        let config_path = model_dir.join("config.json");
        let tokenizer_config_path = model_dir.join("tokenizer_config.json");
        let special_tokens_map_path = model_dir.join("special_tokens_map.json");
        let template_path = model_dir.join("chat_template.jinja");

        // Load tokenizer
        println!("Loading tokenizer for {}...", model_name);
        let tokenizer = Tokenizer::from_file(&tokenizer_path)
            .map_err(|e| EdgeError::model(format!("Failed to load tokenizer: {}", e)))?;
        println!("✅ Tokenizer loaded successfully!");

        // Load ONNX model
        println!("Loading {} ONNX model...", model_name);
        let runtime = OnnxRuntime::new(&model_path, model_name.clone())
            .map_err(|e| EdgeError::model(format!("Failed to load ONNX model: {}", e)))?;
        println!("✅ {} ONNX model loaded successfully!", model_name);

        // Load model configuration
        println!("Loading model configuration...");
        let config_str = fs::read_to_string(&config_path)
            .map_err(|e| EdgeError::configuration(format!("Failed to load config.json: {}", e)))?;
        let mut config: ModelConfig = serde_json::from_str(&config_str)
            .map_err(|e| EdgeError::configuration(format!("Failed to parse config.json: {}", e)))?;

        // Handle different naming conventions
        if config.max_position_embeddings == 2048 && config.n_positions != 2048 {
            config.max_position_embeddings = config.n_positions;
        }

        println!(
            "📋 Model config loaded: max_position_embeddings={}, eos_token_id={}",
            config.max_position_embeddings, config.eos_token_id
        );

        // Load tokenizer config
        println!("Loading tokenizer configuration...");
        let tokenizer_config_str = fs::read_to_string(&tokenizer_config_path).map_err(|e| {
            EdgeError::configuration(format!("Failed to load tokenizer_config.json: {}", e))
        })?;
        let tokenizer_config: TokenizerConfig = serde_json::from_str(&tokenizer_config_str)
            .map_err(|e| {
                EdgeError::configuration(format!("Failed to parse tokenizer_config.json: {}", e))
            })?;
        println!(
            "📖 Tokenizer config loaded: add_bos_token={}, model_max_length={}",
            tokenizer_config.add_bos_token, tokenizer_config.model_max_length
        );

        // Load special tokens map
        println!("Loading special tokens map...");
        let special_tokens_map_str = fs::read_to_string(&special_tokens_map_path).map_err(|e| {
            EdgeError::configuration(format!("Failed to load special_tokens_map.json: {}", e))
        })?;
        let special_tokens_map: SpecialTokensMap = serde_json::from_str(&special_tokens_map_str)
            .map_err(|e| {
                EdgeError::configuration(format!("Failed to parse special_tokens_map.json: {}", e))
            })?;

        // Get actual token strings from special_tokens_map
        let eos_token_str = special_tokens_map
            .eos_token
            .as_ref()
            .and_then(|v| SpecialToken::from_value(v))
            .map(|t| t.content)
            .unwrap_or_else(|| "</s>".to_string());

        let bos_token_str = special_tokens_map
            .bos_token
            .as_ref()
            .and_then(|v| SpecialToken::from_value(v))
            .map(|t| t.content)
            .unwrap_or_else(|| "<s>".to_string());

        println!(
            "📑 Special tokens loaded: BOS='{}', EOS='{}'",
            bos_token_str, eos_token_str
        );

        // Get token IDs from config
        let eos_token_id = config.eos_token_id;
        let bos_token_id = config.bos_token_id;
        println!("🔚 Token IDs: EOS={}, BOS={}", eos_token_id, bos_token_id);

        // Load Jinja template (optional)
        let chat_template = if template_path.exists() {
            println!("📝 Loading chat template...");
            fs::read_to_string(&template_path).map_err(|e| {
                EdgeError::configuration(format!("Failed to load chat template: {}", e))
            })?
        } else {
            println!("📝 No chat template found, using default");
            // Default template for models without chat_template.jinja
            String::from("{% for message in messages %}{{ message['content'] }}{% if not loop.last %} {% endif %}{% endfor %}{% if add_generation_prompt %}{% endif %}")
        };

        // Create conservative sampler for chat (much lower temperature and top_p)
        let sampler = TopPSampler::new(0.1)?; // Very focused sampling

        println!("🎯 Full AI chat inference ready!");

        Ok(Self {
            runtime,
            tokenizer,
            sampler,
            eos_token_id,
            bos_token_id,
            conversation_history: Vec::new(),
            chat_template,
            config,
            tokenizer_config,
            special_tokens_map,
            eos_token_str,
            bos_token_str,
            model_name,
        })
    }

    /// Generate a chat response
    pub fn generate_response(&mut self, input: &str) -> EdgeResult<String> {
        // Special commands
        match input.to_lowercase().as_str() {
            s if s.contains("model") || s.contains("onnx") => {
                return Ok(format!(
                    "✅ {} ONNX model loaded! Using full AI chat inference.",
                    self.model_name
                ));
            }
            "clear" | "reset" => {
                self.conversation_history.clear();
                return Ok("🔄 Conversation history cleared!".to_string());
            }
            _ => {}
        }

        self.run_chat_inference(input)
    }

    /// Run chat inference with proper formatting
    fn run_chat_inference(&mut self, input: &str) -> EdgeResult<String> {
        // Build messages array for Jinja template
        let mut messages = vec![];

        // Check if this is a chat model by examining the template
        let is_chat_model = self.chat_template != String::from("{% for message in messages %}{{ message['content'] }}{% if not loop.last %} {% endif %}{% endfor %}{% if add_generation_prompt %}{% endif %}");

        // Only add system message for chat models that support the 'system' role
        if is_chat_model && self.chat_template.contains("system") {
            messages.push(serde_json::json!({
                "role": "system",
                "content": "You are ChatBOT, a helpful, friendly, and knowledgeable AI assistant. Your name is ChatBOT. When introducing yourself, always use the name ChatBOT. Provide clear, detailed, and conversational responses. Be helpful and engaging while staying informative."
            }));
        }

        // Add conversation history for context (last 3 exchanges to avoid token limits)
        if is_chat_model {
            let recent_history = self.conversation_history.iter().rev().take(3).rev();

            for (user_msg, assistant_msg) in recent_history {
                messages.push(serde_json::json!({
                    "role": "user",
                    "content": user_msg
                }));
                messages.push(serde_json::json!({
                    "role": "assistant",
                    "content": assistant_msg
                }));
            }

            // Add current user message
            messages.push(serde_json::json!({
                "role": "user",
                "content": input.trim()
            }));
        } else {
            // For non-chat models, just use the input as a single message
            messages.push(serde_json::json!({
                "role": "user",
                "content": input.trim()
            }));
        }

        // Render the conversation using Jinja template
        let mut jinja_env = Environment::new();
        jinja_env
            .add_template("chat", &self.chat_template)
            .map_err(|e| EdgeError::template(format!("Failed to parse Jinja template: {}", e)))?;

        let tmpl = jinja_env
            .get_template("chat")
            .map_err(|e| EdgeError::template(format!("Failed to get template: {}", e)))?;

        let conversation = tmpl
            .render(context! {
                messages => messages,
                eos_token => &self.eos_token_str,
                add_generation_prompt => true
            })
            .map_err(|e| EdgeError::template(format!("Failed to render template: {}", e)))?;

        println!(
            "🗨️ Building conversation with {} previous exchanges",
            self.conversation_history.len().min(3)
        );
        println!("📝 Rendered template:\n{}", conversation);

        // Tokenize input with proper BOS handling
        let add_special_tokens = self.tokenizer_config.add_bos_token;
        let input_ids = self
            .tokenizer
            .encode(&conversation, add_special_tokens)?
            .into_iter()
            .map(|id| id as i64)
            .collect::<Vec<_>>();

        let mut current_tokens = input_ids;

        // If tokenizer didn't add BOS but config says we should, add it manually
        if self.tokenizer_config.add_bos_token
            && !current_tokens.is_empty()
            && current_tokens[0] != self.bos_token_id as i64
        {
            current_tokens.insert(0, self.bos_token_id as i64);
        }

        println!(
            "🔤 Tokenizing conversation → {} tokens (BOS: {})",
            current_tokens.len(),
            if self.tokenizer_config.add_bos_token {
                "added"
            } else {
                "not added"
            }
        );

        let mut generated_tokens = Vec::new();
        let max_new_tokens = 100;

        // Conservative generation options for consistent chat responses
        let gen_options = GenerationOptions {
            max_new_tokens,
            temperature: 0.1, // Much lower temperature for deterministic output
            top_p: 0.1,       // Very focused sampling for consistency
            do_sample: true,
            ..Default::default()
        };

        println!("🧠 Running {} chat inference...", self.model_name);

        // Generate tokens one by one
        for _step in 0..max_new_tokens {
            let seq_len = current_tokens.len();

            // Check token limit from config
            if seq_len > self.config.max_position_embeddings - 200 {
                println!(
                    "⚠️ Approaching token limit ({}/{}), stopping generation",
                    seq_len, self.config.max_position_embeddings
                );
                break;
            }

            // Prepare input for inference
            let onnx_input = OnnxInput::new(current_tokens.clone());

            // Run inference
            let output = self.runtime.infer(onnx_input)?;

            // Get logits for the last token
            let last_logits = output.last_token_logits()?;

            // Sample next token using conservative top-p sampling
            let next_token = self.sampler.sample(last_logits, &gen_options)?;

            // Check for end-of-sequence token
            if next_token == self.eos_token_id {
                println!("🔚 Hit EOS token, stopping generation");
                break;
            }

            // Add the new token
            generated_tokens.push(next_token);
            current_tokens.push(next_token as i64);

            // Improved early stopping - look for natural conversation endings
            if generated_tokens.len() >= 5 {
                let partial_text = self
                    .tokenizer
                    .decode(&generated_tokens, true)
                    .unwrap_or_default();

                // Stop on natural endings, but require minimum length
                if generated_tokens.len() >= 10
                    && (
                        partial_text.trim().ends_with('.')
                            || partial_text.trim().ends_with('!')
                            || partial_text.trim().ends_with('?')
                            || partial_text.contains("<|")
                        // Stop if we hit chat markers
                    )
                {
                    break;
                }

                // Stop on repeated patterns (avoid loops)
                if generated_tokens.len() >= 20 {
                    let last_10: Vec<u32> =
                        generated_tokens.iter().rev().take(10).cloned().collect();
                    let prev_10: Vec<u32> = generated_tokens
                        .iter()
                        .rev()
                        .skip(10)
                        .take(10)
                        .cloned()
                        .collect();
                    if last_10 == prev_10 {
                        println!("🔁 Detected repetition, stopping");
                        break;
                    }
                }
            }
        }

        // Decode the generated tokens
        let generated_text = self
            .tokenizer
            .decode(&generated_tokens, true)
            .map_err(|e| EdgeError::inference(format!("Failed to decode: {}", e)))?;

        let response = generated_text.trim();

        // Clean up any leftover chat markers or unwanted text
        let without_eos = response.replace("</s>", ""); // Remove EOS tokens
        let cleaned_response = without_eos
            .split("<|")
            .next()
            .unwrap_or(&without_eos) // Remove any chat markers
            .trim();

        println!(
            "🎯 Generated {} tokens: '{}'",
            generated_tokens.len(),
            cleaned_response
        );

        let final_response = if cleaned_response.is_empty() {
            "I'm here to help! What would you like to know?".to_string()
        } else {
            cleaned_response.to_string()
        };

        // Store this exchange in conversation history
        self.conversation_history
            .push((input.to_string(), final_response.clone()));

        // Keep only last 5 exchanges to manage memory
        if self.conversation_history.len() > 5 {
            self.conversation_history
                .drain(0..self.conversation_history.len() - 5);
        }

        Ok(final_response)
    }
}

/// Interactive chat mode
fn interactive_mode(chat_bot: &mut ChatBot) -> EdgeResult<()> {
    println!("🔍 Debug: interactive_mode called");
    println!("\n🤖 Interactive Chat Mode - Type 'quit' to exit");
    println!("🎯 Using conservative sampling for consistent responses\n");

    loop {
        print!("\n> ");
        io::stdout().flush().unwrap();

        let mut input = String::new();
        io::stdin()
            .read_line(&mut input)
            .map_err(|e| EdgeError::runtime(format!("Failed to read input: {}", e)))?;

        let input = input.trim();

        if input.is_empty() {
            continue;
        }

        if input.to_lowercase() == "quit" || input.to_lowercase() == "exit" {
            println!("👋 Goodbye!");
            break;
        }

        match chat_bot.generate_response(input) {
            Ok(response) => {
                println!("🤖: {}", response);
            }
            Err(e) => {
                eprintln!("❌ Generation error: {}", e);
            }
        }
    }

    Ok(())
}

/// Single prompt mode
fn single_prompt_mode(chat_bot: &mut ChatBot, prompt: &str) -> EdgeResult<()> {
    println!(
        "🔍 Debug: single_prompt_mode called with prompt: '{}'",
        prompt
    );
    println!("📝 Input prompt: {}\n", prompt);

    match chat_bot.generate_response(prompt) {
        Ok(response) => {
            println!("🔍 Debug: Response generated successfully");
            println!("{}", response);
        }
        Err(e) => {
            eprintln!("❌ Generation failed: {}", e);
            return Err(e);
        }
    }

    Ok(())
}

fn main() -> EdgeResult<()> {
    // Initialize logging
    liquid_edge::init_with_level(log::LevelFilter::Info)?;

    println!("🚀 Liquid Edge - Enhanced Chat Bot with Jinja Templates");
    println!("====================================================\n");

    // Parse command line arguments
    let args: Vec<String> = env::args().collect();
    println!("🔍 Debug: Command line args: {:?}", args);

    // Get model name from command line args or default to tinyllama
    let model_name = args.get(1).map(|s| s.as_str()).unwrap_or("tinyllama");
    println!("🔍 Debug: Model name: {}", model_name);

    // Determine model directory - try multiple possible locations
    let possible_paths = vec![
        format!("models/{}", model_name),
        format!("../../models/{}", model_name),
        format!("../models/{}", model_name),
    ];

    let mut model_dir = String::new();
    let mut found = false;

    for path in possible_paths {
        let test_path = Path::new(&path);
        if test_path.exists() {
            model_dir = path;
            found = true;
            break;
        }
    }

    if !found {
        return Err(EdgeError::model(format!(
            "Model directory not found in any of the expected locations. Please ensure the {} model is downloaded to a 'models/{}' directory.",
            model_name, model_name
        )));
    }

    let model_path = Path::new(&model_dir);

    println!("📁 Loading model from: {}", model_dir);

    // Create chat bot
    let mut chat_bot = match ChatBot::new(model_path, model_name.to_string()) {
        Ok(bot) => bot,
        Err(e) => {
            eprintln!("❌ Failed to initialize chat bot: {}", e);
            eprintln!("\n💡 Troubleshooting tips:");
            eprintln!("   1. Ensure model.onnx exists in the model directory");
            eprintln!("   2. Ensure tokenizer.json exists in the model directory");
            eprintln!("   3. Ensure config files (config.json, tokenizer_config.json, special_tokens_map.json) exist");
            eprintln!("   4. Check that the ONNX model is compatible");
            return Err(e);
        }
    };

    // Determine execution mode based on arguments and input type
    let is_terminal = io::stdin().is_terminal();
    println!(
        "🔍 Debug: Is terminal: {}, args.len(): {}",
        is_terminal,
        args.len()
    );

    // Check if a prompt was provided as command line argument (priority)
    if args.len() > 2 {
        // Single prompt mode
        let prompt = args[2..].join(" ");
        println!("🔍 Debug: Entering single prompt mode with: '{}'", prompt);
        single_prompt_mode(&mut chat_bot, &prompt)?;
    } else if is_terminal {
        // Interactive mode
        println!("🔍 Debug: Entering interactive mode");
        interactive_mode(&mut chat_bot)?;
    } else {
        // Piped input mode
        println!("🔍 Debug: Entering piped input mode");
        let mut input = String::new();
        io::stdin()
            .read_to_string(&mut input)
            .map_err(|e| EdgeError::runtime(format!("Failed to read piped input: {}", e)))?;
        let input = input.trim();

        if !input.is_empty() {
            let response = chat_bot.generate_response(input)?;
            println!("{}", response);
        }
    }

    println!("🎉 Chat session completed successfully!");
    Ok(())
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_model_config_defaults() {
        let config = ModelConfig {
            eos_token_id: default_eos_token_id(),
            bos_token_id: default_bos_token_id(),
            max_position_embeddings: default_max_position_embeddings(),
            n_positions: default_max_position_embeddings(),
            vocab_size: default_vocab_size(),
        };

        assert_eq!(config.eos_token_id, 2);
        assert_eq!(config.bos_token_id, 1);
        assert_eq!(config.max_position_embeddings, 2048);
        assert_eq!(config.vocab_size, 32000);
    }

    #[test]
    fn test_special_token_parsing() {
        let json_str = r#"{"content": "</s>", "lstrip": false, "normalized": false, "rstrip": false, "single_word": false}"#;
        let value: serde_json::Value = serde_json::from_str(json_str).unwrap();
        let token = SpecialToken::from_value(&value).unwrap();

        assert_eq!(token.content, "</s>");
        assert_eq!(token.lstrip, false);
    }
}
