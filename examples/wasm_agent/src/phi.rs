use crate::console_log;
use candle_core::quantized::gguf_file;
use candle_core::{DType, Device, Tensor};
use candle_nn::VarBuilder;
use candle_transformers::generation::LogitsProcessor;
use candle_transformers::models::phi3::{Config as Phi3Config, Model as Phi3Model};
use candle_transformers::models::quantized_phi3::ModelWeights as QPhi3Model;
use js_sys::Date;
use js_sys::Math::log;
use serde::Deserialize;
use tokenizers::Tokenizer;
use wasm_bindgen::prelude::*;

enum SelectedModel {
    Phi3(Phi3Model),
    QuantizedPhi3(QPhi3Model),
}

#[wasm_bindgen]
pub struct Model {
    model: SelectedModel,
    tokenizer: Tokenizer,
    logits_processor: LogitsProcessor,
    tokens: Vec<u32>,
    repeat_penalty: f32,
    repeat_last_n: usize,
}

#[derive(Debug, Clone, PartialEq, Deserialize)]

pub struct ModelName {
    pub _name_or_path: String,
}

#[wasm_bindgen]
impl Model {
    #[wasm_bindgen(constructor)]
    pub fn load(
        weights: Vec<u8>,
        tokenizer: Vec<u8>,
        config: Vec<u8>,
        quantized: bool,
    ) -> Result<Model, JsError> {
        console_error_panic_hook::set_once();
        console_log!("loading model");
        let device = Device::Cpu;
        let name: ModelName = serde_json::from_slice(&config)?;

        console_log!("config loaded {:?}", name);
        let tokenizer =
            Tokenizer::from_bytes(&tokenizer).map_err(|m| JsError::new(&m.to_string()))?;
        let start = Date::now();
        console_log!("weights len: {:?}", weights.len());

        // Load Phi-3 model
        let phi3_config: Phi3Config = serde_json::from_slice(&config)?;
        let model = if quantized {
            console_log!("Loading quantized Phi-3 model from GGUF");
            // Parse GGUF content for quantized models
            let mut reader = std::io::Cursor::new(&weights);
            let content = gguf_file::Content::read(&mut reader)
                .map_err(|e| JsError::new(&format!("Failed to read GGUF content: {}", e)))?;

            // QPhi3Model requires from_gguf method with flash_attn flag
            let model = QPhi3Model::from_gguf(false, content, &mut reader, &device)
                .map_err(|e| JsError::new(&format!("Failed to load quantized Phi-3: {}", e)))?;
            console_log!("Quantized Phi-3 model loaded successfully");
            SelectedModel::QuantizedPhi3(model)
        } else {
            console_log!("Loading non-quantized Phi-3 model");
            let vb = VarBuilder::from_buffered_safetensors(weights, DType::F32, &device)?;
            let model = Phi3Model::new(&phi3_config, vb)?;
            SelectedModel::Phi3(model)
        };
        console_log!("model loaded in {:?}s", (Date::now() - start) / 1000.);
        let logits_processor = LogitsProcessor::new(299792458, None, None);
        Ok(Self {
            model,
            tokenizer,
            tokens: vec![],
            logits_processor,
            repeat_penalty: 1.,
            repeat_last_n: 64,
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
        match &mut self.model {
            SelectedModel::Phi3(m) => m.clear_kv_cache(),
            SelectedModel::QuantizedPhi3(_) => {} // Not implemented yet
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
            SelectedModel::Phi3(m) => m.forward(&input, self.tokens.len())?,
            SelectedModel::QuantizedPhi3(m) => m.forward(&input, self.tokens.len())?,
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
        let token = self
            .tokenizer
            .decode(&[next_token], false)
            .unwrap_or_else(|e| {
                console_log!("error decoding token: {:?}", e);
                "".to_string()
            });
        // console_log!("token: {:?}: {:?}", token, next_token);
        Ok(token)
    }
}
