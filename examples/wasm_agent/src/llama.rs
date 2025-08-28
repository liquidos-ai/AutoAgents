use crate::console_log;
use candle_core::quantized::gguf_file;
use candle_core::{DType, Device, Tensor};
use candle_transformers::generation::LogitsProcessor;
use candle_transformers::models::quantized_llama::ModelWeights as QLlamaModel;
use js_sys::Date;
use serde::Deserialize;
use tokenizers::Tokenizer;
use wasm_bindgen::prelude::*;

enum SelectedModel {
    Quantized(QLlamaModel),
}

#[wasm_bindgen]
pub struct Model {
    model: SelectedModel,
    tokenizer: Tokenizer,
    logits_processor: LogitsProcessor,
    tokens: Vec<u32>,
    repeat_penalty: f32,
    repeat_last_n: usize,
    previous_text_length: usize,
    stop_tokens: Vec<String>,
}

#[derive(Debug, Clone, PartialEq, Deserialize)]
#[serde(default)]
pub struct ModelName {
    pub _name_or_path: Option<String>,
    pub model_type: Option<String>,
    pub architectures: Option<Vec<String>>,
}

impl Default for ModelName {
    fn default() -> Self {
        Self {
            _name_or_path: Some("TinyLlama".to_string()),
            model_type: Some("llama".to_string()),
            architectures: Some(vec!["LlamaForCausalLM".to_string()]),
        }
    }
}

// Helper function to clean up invalid UTF-8 bytes at the end of tokenizer data
fn clean_tokenizer_bytes(bytes: &[u8]) -> Vec<u8> {
    // First, find where valid UTF-8 ends
    let mut valid_end = bytes.len();
    while valid_end > 0 {
        if std::str::from_utf8(&bytes[..valid_end]).is_ok() {
            break;
        }
        valid_end -= 1;
    }

    if valid_end == 0 {
        console_log!("No valid UTF-8 found, returning original bytes");
        return bytes.to_vec();
    }

    if valid_end < bytes.len() {
        console_log!(
            "Found invalid UTF-8 at position {}, truncating to {}",
            valid_end,
            valid_end
        );
    }

    // Convert the valid portion to string
    let s = match std::str::from_utf8(&bytes[..valid_end]) {
        Ok(s) => s,
        Err(_) => {
            console_log!("Even truncated bytes are invalid UTF-8, using lossy conversion");
            return String::from_utf8_lossy(&bytes[..valid_end])
                .as_bytes()
                .to_vec();
        }
    };

    // The tokenizer JSON might have invalid trailing content after the main structure
    // Look for the pattern "}}" which typically indicates the end of the tokenizer structure
    if let Some(double_brace_pos) = s.find("}}") {
        // Find the position right after the double brace
        let end_pos = double_brace_pos + 2;
        let clean_json = &s[..end_pos];
        console_log!(
            "Found tokenizer end pattern at position {}, trimming to length: {}",
            double_brace_pos,
            clean_json.len()
        );

        // Verify this creates balanced JSON
        let open_braces = clean_json.chars().filter(|&c| c == '{').count();
        let close_braces = clean_json.chars().filter(|&c| c == '}').count();
        console_log!(
            "Brace count after trimming: {} open, {} close",
            open_braces,
            close_braces
        );

        if open_braces == close_braces {
            console_log!("JSON is now balanced!");
            return clean_json.as_bytes().to_vec();
        }
    }

    // Fallback: Find the last valid closing brace
    if let Some(last_brace) = s.rfind('}') {
        let clean_json = &s[..=last_brace];
        console_log!(
            "Fallback: Found last closing brace at position {}, final length: {}",
            last_brace,
            clean_json.len()
        );

        // Basic JSON validation - count braces
        let open_braces = clean_json.chars().filter(|&c| c == '{').count();
        let close_braces = clean_json.chars().filter(|&c| c == '}').count();
        console_log!("Brace count: {} open, {} close", open_braces, close_braces);

        clean_json.as_bytes().to_vec()
    } else {
        console_log!("No closing brace found, returning truncated UTF-8");
        s.as_bytes().to_vec()
    }
}

#[wasm_bindgen]
impl Model {
    #[wasm_bindgen(constructor)]
    pub fn load(
        weights: Vec<u8>,
        _tokenizer: Vec<u8>, // Unused - we use embedded tokenizer
        _config: Vec<u8>,    // Unused - we skip config parsing
        quantized: bool,
    ) -> Result<Model, JsError> {
        console_error_panic_hook::set_once();
        console_log!("loading TinyLlama model");
        let device = Device::Cpu;
        // Simply assume it's a TinyLlama model for now - no complex config parsing
        console_log!("Skipping config parsing to avoid interference with tokenizer");

        // Use the embedded tokenizer file instead of downloading
        console_log!("Using embedded tokenizer from models folder...");
        let embedded_tokenizer = include_bytes!("../models/tokenizer.json");
        console_log!("Embedded tokenizer length: {}", embedded_tokenizer.len());

        let tokenizer = match Tokenizer::from_bytes(embedded_tokenizer) {
            Ok(t) => {
                console_log!("Embedded tokenizer loaded successfully");
                t
            }
            Err(e) => {
                console_log!(
                    "Embedded tokenizer failed: {}, trying downloaded tokenizer...",
                    e
                );
                // Fall back to the downloaded tokenizer parameter
                Tokenizer::from_bytes(&_tokenizer)
                    .map_err(|m| JsError::new(&format!("Both embedded and downloaded tokenizer failed. Embedded: {}, Downloaded: {}", e, m)))?
            }
        };
        let start = Date::now();
        console_log!("weights len: {:?}", weights.len());

        if !quantized {
            return Err(JsError::new(
                "Only quantized TinyLlama models are supported",
            ));
        }

        console_log!("Loading quantized TinyLlama model from GGUF");
        // Parse GGUF content for quantized models
        let mut reader = std::io::Cursor::new(&weights);
        let content = gguf_file::Content::read(&mut reader)
            .map_err(|e| JsError::new(&format!("Failed to read GGUF content: {}", e)))?;

        // Load quantized llama model - compatible with TinyLlama
        let model = QLlamaModel::from_gguf(content, &mut reader, &device)
            .map_err(|e| JsError::new(&format!("Failed to load quantized TinyLlama: {}", e)))?;
        console_log!("Quantized TinyLlama model loaded successfully");
        let selected_model = SelectedModel::Quantized(model);

        console_log!("model loaded in {:?}s", (Date::now() - start) / 1000.);
        let logits_processor = LogitsProcessor::new(299792458, None, None);

        // Define stop tokens for TinyLlama
        let stop_tokens = vec![
            "</s>".to_string(),
            "<|endoftext|>".to_string(),
            "<|user|>".to_string(),
            "<|system|>".to_string(),
            "<|assistant|>".to_string(),
            "[INST]".to_string(),
            "[/INST]".to_string(),
            "Human:".to_string(),
            "Assistant:".to_string(),
        ];

        Ok(Self {
            model: selected_model,
            tokenizer,
            tokens: vec![],
            logits_processor,
            repeat_penalty: 1.,
            repeat_last_n: 64,
            previous_text_length: 0,
            stop_tokens,
        })
    }

    #[wasm_bindgen]
    pub fn init_with_prompt(
        &mut self,
        prompt: String,
        temp: f64,
        top_p: f64,
        repeat_penalty: f32,
        repeat_last_n: usize,
        seed: u64,
    ) -> Result<String, JsError> {
        // Clear cache - not implemented for quantized models yet
        match &mut self.model {
            SelectedModel::Quantized(_) => {} // Cache clearing not available
        };

        let temp = if temp <= 0. { None } else { Some(temp) };
        let top_p = if top_p <= 0. || top_p >= 1. {
            None
        } else {
            Some(top_p)
        };
        self.logits_processor = LogitsProcessor::new(seed, temp, top_p);
        self.repeat_penalty = repeat_penalty;
        self.repeat_last_n = repeat_last_n;
        self.tokens.clear();
        self.previous_text_length = 0;
        let tokens = self
            .tokenizer
            .encode(prompt, true)
            .map_err(|m| JsError::new(&m.to_string()))?
            .get_ids()
            .to_vec();
        let text = self
            .process(&tokens)
            .map_err(|m| JsError::new(&m.to_string()))?;
        Ok(text)
    }

    #[wasm_bindgen]
    pub fn next_token(&mut self) -> Result<String, JsError> {
        let last_token = *self.tokens.last().unwrap();
        let text = self
            .process(&[last_token])
            .map_err(|m| JsError::new(&m.to_string()))?;
        Ok(text)
    }
}

impl Model {
    fn process(&mut self, tokens: &[u32]) -> candle_core::Result<String> {
        let dev = Device::Cpu;
        let input = Tensor::new(tokens, &dev)?.unsqueeze(0)?;
        let logits = match &mut self.model {
            SelectedModel::Quantized(m) => m.forward(&input, self.tokens.len())?,
        };
        let logits = logits.squeeze(0)?.to_dtype(DType::F32)?;
        let logits = if self.repeat_penalty == 1. {
            logits
        } else {
            let start_at = tokens.len().saturating_sub(self.repeat_last_n);
            candle_transformers::utils::apply_repeat_penalty(
                &logits,
                self.repeat_penalty,
                &tokens[start_at..],
            )?
        };

        let next_token = self.logits_processor.sample(&logits)?;
        self.tokens.push(next_token);

        // Decode the entire sequence to get proper spacing, then extract the last token
        let full_text = self
            .tokenizer
            .decode(&self.tokens, true)
            .unwrap_or_else(|e| {
                console_log!("error decoding full sequence: {:?}", e);
                "".to_string()
            });

        // Check if the full text contains any stop tokens
        for stop_token in &self.stop_tokens {
            if full_text.contains(stop_token) {
                console_log!("Stop token detected: {}", stop_token);
                // Return the text up to the stop token
                if let Some(stop_pos) = full_text.find(stop_token) {
                    let clean_text = &full_text[..stop_pos];
                    let token = if clean_text.len() > self.previous_text_length {
                        let new_text = &clean_text[self.previous_text_length..];
                        self.previous_text_length = clean_text.len();
                        new_text.to_string()
                    } else {
                        String::new()
                    };
                    console_log!("Final token before stop: '{}'", token);
                    return Ok(token);
                }
            }
        }

        // For streaming, we need to return only the new part
        let current_length = full_text.len();
        let token = if current_length > self.previous_text_length {
            let new_text = &full_text[self.previous_text_length..];
            self.previous_text_length = current_length;
            new_text.to_string()
        } else {
            String::new()
        };

        console_log!("Decoded token: '{}'", token);
        Ok(token)
    }
}
