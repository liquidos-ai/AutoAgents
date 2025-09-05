/// This module is taken from the Candle Wasm Examples
use crate::console_log;
use candle::{DType, Device, Tensor};
use candle_nn::VarBuilder;
use candle_transformers::generation::LogitsProcessor;
use candle_transformers::models::mixformer::{Config, MixFormerSequentialForCausalLM as MixFormer};
use candle_transformers::models::quantized_mixformer::MixFormerSequentialForCausalLM as QMixFormer;
use js_sys::Date;
use serde::{Deserialize, Serialize};
use tokenizers::Tokenizer;
use wasm_bindgen::prelude::*;

#[derive(Serialize, Deserialize)]
pub struct PhiTokenOutput {
    pub token: String,
}

#[derive(Serialize, Deserialize)]
pub struct PhiInitInput {
    pub prompt: String,
    pub temp: f64,
    pub top_p: f64,
    pub repeat_penalty: f32,
    pub repeat_last_n: usize,
    pub seed: u64,
}

#[derive(Debug, Clone, PartialEq, Deserialize)]
pub struct ModelName {
    pub _name_or_path: String,
}

enum SelectedModel {
    MixFormer(MixFormer),
    Quantized(QMixFormer),
}

#[wasm_bindgen]
pub struct PhiModel {
    model: SelectedModel,
    tokenizer: Tokenizer,
    logits_processor: LogitsProcessor,
    tokens: Vec<u32>,
    repeat_penalty: f32,
    repeat_last_n: usize,
}

#[wasm_bindgen]
impl PhiModel {
    #[wasm_bindgen(constructor)]
    pub fn load(
        weights: Vec<u8>,
        tokenizer: Vec<u8>,
        config: Vec<u8>,
        quantized: bool,
    ) -> Result<PhiModel, JsError> {
        console_error_panic_hook::set_once();
        console_log!("Loading Phi model");

        let device = Device::Cpu;
        let name: ModelName = serde_json::from_slice(&config)?;
        let config: Config = serde_json::from_slice(&config)?;

        console_log!("Config loaded {:?}", name);
        let tokenizer =
            Tokenizer::from_bytes(&tokenizer).map_err(|m| JsError::new(&m.to_string()))?;

        let start = Date::now();
        console_log!("Weights len: {:?}", weights.len());

        let model = if quantized {
            let vb = candle_transformers::quantized_var_builder::VarBuilder::from_gguf_buffer(
                &weights, &device,
            )?;
            console_log!("Quantized weights loaded");
            if name._name_or_path == "microsoft/phi-2" {
                let model = QMixFormer::new_v2(&config, vb)?;
                SelectedModel::Quantized(model)
            } else {
                let model = QMixFormer::new(&config, vb)?;
                SelectedModel::Quantized(model)
            }
        } else {
            let device = &Device::Cpu;
            let vb = VarBuilder::from_buffered_safetensors(weights, DType::F32, device)?;
            let model = MixFormer::new(&config, vb)?;
            SelectedModel::MixFormer(model)
        };

        console_log!("Phi model loaded in {:?}s", (Date::now() - start) / 1000.0);
        let logits_processor = LogitsProcessor::new(299792458, None, None);

        Ok(Self {
            model,
            tokenizer,
            tokens: vec![],
            logits_processor,
            repeat_penalty: 1.0,
            repeat_last_n: 64,
        })
    }

    #[wasm_bindgen]
    pub fn init_with_prompt(&mut self, input: JsValue) -> Result<JsValue, JsError> {
        let PhiInitInput {
            prompt,
            temp,
            top_p,
            repeat_penalty,
            repeat_last_n,
            seed,
        } = serde_wasm_bindgen::from_value(input).map_err(|m| JsError::new(&m.to_string()))?;

        match &mut self.model {
            SelectedModel::MixFormer(m) => m.clear_kv_cache(),
            SelectedModel::Quantized(m) => m.clear_kv_cache(),
        };

        let temp = if temp <= 0.0 { None } else { Some(temp) };
        let top_p = if top_p <= 0.0 || top_p >= 1.0 {
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

        let text = match self.process(&tokens) {
            Ok(text) => text,
            Err(_e) => {
                console_log!("Error processing tokens");
                "".to_string()
            }
        };

        let output = PhiTokenOutput { token: text };
        Ok(serde_wasm_bindgen::to_value(&output)?)
    }

    #[wasm_bindgen]
    pub fn next_token(&mut self) -> Result<JsValue, JsError> {
        let last_token = *self.tokens.last().unwrap();
        let text = match self.process(&[last_token]) {
            Ok(text) => text,
            Err(_e) => {
                console_log!("Error decoding token");
                "".to_string()
            }
        };

        let output = PhiTokenOutput { token: text };
        Ok(serde_wasm_bindgen::to_value(&output)?)
    }

    fn process(&mut self, tokens: &[u32]) -> Result<String, JsError> {
        let dev = Device::Cpu;
        let input = Tensor::new(tokens, &dev)?.unsqueeze(0)?;
        let logits = match &mut self.model {
            SelectedModel::MixFormer(m) => m.forward(&input)?,
            SelectedModel::Quantized(m) => m.forward(&input)?,
        };

        let logits = logits.squeeze(0)?.to_dtype(DType::F32)?;
        let logits = if self.repeat_penalty == 1.0 {
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
        let token = match self.tokenizer.decode(&[next_token], false) {
            Ok(token) => token,
            Err(e) => {
                console_log!("Error decoding token: {:?}", e);
                "".to_string()
            }
        };

        Ok(token)
    }
}
