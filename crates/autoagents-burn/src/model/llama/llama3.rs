use crate::backend::burn_backend_types::InferenceBackend;
use crate::model::llama::chat::{GenerationConfig, LLamaModel, LlamaChat};
use crate::model::llama::tokenizer::Tiktoken;
use crate::model::llama::{Llama, LlamaConfig};
use crate::utils::{spawn_blocking, CustomMutex};
use autoagents_llm::error::LLMError;
use log::info;
use std::marker::PhantomData;
use std::path::PathBuf;
use std::sync::Arc;

/// Supported Llama3 model variants
#[derive(Debug, Clone, Copy, Default)]
pub enum Llama3Model {
    /// Llama-3-8B-Instruct model
    #[default]
    Llama3_8B,
    /// Llama-3.1-8B-Instruct model
    Llama3_1_8B,
    /// Llama-3.2-3B-Instruct model
    Llama3_2_3B,
    /// Llama-3.2-1B-Instruct model
    Llama3_2_1B,
    /// Llama-3.2-1B-Instruct quantized model
    Llama3_2_1bQ4,
}

pub struct LLama3ModelConfig {
    pub model_path: PathBuf,
    pub tokenizer_path: PathBuf,
    pub max_seq_len: usize,
    pub generation_config: GenerationConfig,
    pub model_variant: Llama3Model,
    pub import: bool,
    pub model_bytes: Option<Vec<u8>>,
    pub tokenizer_bytes: Option<Vec<u8>>,
}

impl Default for LLama3ModelConfig {
    fn default() -> Self {
        Self {
            model_path: PathBuf::from("models/llama3.mpk"),
            tokenizer_path: PathBuf::from("models/tokenizer.model"),
            max_seq_len: 8192,
            generation_config: GenerationConfig::default(),
            model_variant: Llama3Model::default(),
            import: false,
            model_bytes: None,
            tokenizer_bytes: None,
        }
    }
}

#[derive(Default)]
pub struct Llama3Builder {
    config: LLama3ModelConfig,
}

impl Llama3Builder {
    pub fn new() -> Self {
        Self::default()
    }

    #[cfg(not(target_arch = "wasm32"))]
    pub fn model_path(mut self, path: impl Into<PathBuf>) -> Self {
        self.config.model_path = path.into();
        self
    }

    #[cfg(not(target_arch = "wasm32"))]
    pub fn tokenizer_path(mut self, path: impl Into<PathBuf>) -> Self {
        self.config.tokenizer_path = path.into();
        self
    }

    pub fn max_seq_len(mut self, len: usize) -> Self {
        self.config.max_seq_len = len;
        self
    }

    pub fn temperature(mut self, temp: f64) -> Self {
        self.config.generation_config.temperature = temp;
        self
    }

    pub fn max_tokens(mut self, tokens: usize) -> Self {
        self.config.generation_config.max_tokens = tokens;
        self
    }

    /// Set the Llama3 model variant
    pub fn model_variant(mut self, variant: Llama3Model) -> Self {
        self.config.model_variant = variant;
        self
    }

    /// Convenience method to set Llama-3-8B model
    pub fn llama3_8b(mut self) -> Self {
        self.config.model_variant = Llama3Model::Llama3_8B;
        self
    }

    /// Convenience method to set Llama-3.1-8B model
    pub fn llama3_1_8b(mut self) -> Self {
        self.config.model_variant = Llama3Model::Llama3_1_8B;
        self
    }

    /// Convenience method to set Llama-3.2-3B model
    pub fn llama3_2_3b(mut self) -> Self {
        self.config.model_variant = Llama3Model::Llama3_2_3B;
        self
    }

    /// Convenience method to set Llama-3.2-1B model
    pub fn llama3_2_1b(mut self) -> Self {
        self.config.model_variant = Llama3Model::Llama3_2_1B;
        self
    }

    /// Convenience method to set Llama-3.2-1B quantized model
    pub fn llama3_2_1b_q4(mut self) -> Self {
        self.config.model_variant = Llama3Model::Llama3_2_1bQ4;
        self
    }

    pub fn with_model_bytes(mut self, bytes: Vec<u8>) -> Self {
        self.config.model_bytes = Some(bytes);
        self.config.import = true;
        self
    }

    pub fn with_tokenizer_bytes(mut self, bytes: Vec<u8>) -> Self {
        self.config.tokenizer_bytes = Some(bytes);
        self
    }

    #[cfg(feature = "pretrained")]
    pub async fn build_from_pretrained(
        self,
    ) -> Result<Arc<LlamaChat<InferenceBackend, Tiktoken>>, LLMError> {
        use crate::backend::burn_backend_types::INFERENCE_DEVICE;

        let device = INFERENCE_DEVICE;

        info!(
            "Burn using device {}",
            crate::backend::burn_backend_types::NAME
        );
        info!(
            "Loading Llama3 model variant: {:?}",
            self.config.model_variant
        );

        // Use tokio::task::spawn_blocking to run the blocking pretrained loading in a separate thread
        let max_seq_len = self.config.max_seq_len;
        let model_variant = self.config.model_variant;

        let llama = spawn_blocking(move || match model_variant {
            Llama3Model::Llama3_8B => {
                LlamaConfig::llama3_8b_pretrained::<InferenceBackend>(max_seq_len, &device)
            }
            Llama3Model::Llama3_1_8B => {
                LlamaConfig::llama3_1_8b_pretrained::<InferenceBackend>(max_seq_len, &device)
            }
            Llama3Model::Llama3_2_3B => {
                LlamaConfig::llama3_2_3b_pretrained::<InferenceBackend>(max_seq_len, &device)
            }
            Llama3Model::Llama3_2_1B => {
                LlamaConfig::llama3_2_1b_pretrained::<InferenceBackend>(max_seq_len, &device)
            }
            Llama3Model::Llama3_2_1bQ4 => {
                LlamaConfig::llama3_2_1b_pretrained_q4::<InferenceBackend>(max_seq_len, &device)
            }
        })
        .await
        .map_err(LLMError::Generic)?
        .map_err(|e| LLMError::Generic(format!("Failed to load model: {}", e)))?;

        Ok(Arc::new(LlamaChat {
            llama: Arc::new(CustomMutex::new(llama)),
            config: self.config.generation_config,
            marker: PhantomData,
            model: LLamaModel::Llama3,
        }))
    }

    #[cfg(not(target_arch = "wasm32"))]
    pub async fn build_from_bytes(
        self,
    ) -> Result<Arc<LlamaChat<InferenceBackend, Tiktoken>>, LLMError> {
        use crate::backend::burn_backend_types::INFERENCE_DEVICE;

        let device = INFERENCE_DEVICE;

        info!(
            "Burn using device {}",
            crate::backend::burn_backend_types::NAME
        );

        // Convert paths into owned Strings
        let model_path = self
            .config
            .model_path
            .to_str()
            .ok_or_else(|| LLMError::Generic("Invalid model path".to_string()))?
            .to_string();
        let tokenizer_path = self
            .config
            .tokenizer_path
            .to_str()
            .ok_or_else(|| LLMError::Generic("Invalid tokenizer path".to_string()))?
            .to_string();

        let max_seq_len = self.config.max_seq_len;

        // Move owned values into spawn_blocking
        let llama = spawn_blocking(move || {
            LlamaConfig::load_llama3_2_1b_from_bytes::<InferenceBackend>(
                &model_path,
                &tokenizer_path,
                max_seq_len,
                &device,
            )
        })
        .await
        .map_err(LLMError::Generic)?
        .map_err(|e| LLMError::Generic(format!("Failed to load model: {}", e)))?;

        Ok(Arc::new(LlamaChat {
            llama: Arc::new(CustomMutex::new(llama)),
            config: self.config.generation_config,
            marker: PhantomData,
            model: LLamaModel::TinyLLama,
        }))
    }

    pub fn build(self) -> Result<Arc<LlamaChat<InferenceBackend, Tiktoken>>, LLMError> {
        use crate::backend::burn_backend_types::INFERENCE_DEVICE;

        let device = INFERENCE_DEVICE;

        info!("Burn Llama using device: {device:?}");
        info!(
            "Building Llama3 model variant: {:?}",
            self.config.model_variant
        );

        let model_path = self
            .config
            .model_path
            .to_str()
            .ok_or_else(|| LLMError::Generic("Invalid model path".to_string()))?;
        let tokenizer_path = self
            .config
            .tokenizer_path
            .to_str()
            .ok_or_else(|| LLMError::Generic("Invalid tokenizer path".to_string()))?;

        // Select the appropriate load function based on model variant
        let llama: Llama<InferenceBackend, _> = match self.config.model_variant {
            Llama3Model::Llama3_8B => LlamaConfig::load_llama3_8b::<InferenceBackend>(
                model_path,
                tokenizer_path,
                self.config.max_seq_len,
                &device,
            )
            .map_err(|e| LLMError::Generic(format!("Failed to load Llama3-8B model: {}", e)))?,
            Llama3Model::Llama3_1_8B => LlamaConfig::load_llama3_1_8b::<InferenceBackend>(
                model_path,
                tokenizer_path,
                self.config.max_seq_len,
                &device,
            )
            .map_err(|e| LLMError::Generic(format!("Failed to load Llama3.1-8B model: {}", e)))?,
            Llama3Model::Llama3_2_3B => LlamaConfig::load_llama3_2_3b::<InferenceBackend>(
                model_path,
                tokenizer_path,
                self.config.max_seq_len,
                &device,
            )
            .map_err(|e| LLMError::Generic(format!("Failed to load Llama3.2-3B model: {}", e)))?,
            Llama3Model::Llama3_2_1B => LlamaConfig::load_llama3_2_1b::<InferenceBackend>(
                model_path,
                tokenizer_path,
                self.config.max_seq_len,
                &device,
            )
            .map_err(|e| LLMError::Generic(format!("Failed to load Llama3.2-1B model: {}", e)))?,
            Llama3Model::Llama3_2_1bQ4 => {
                // For quantized model, use the same loader as the 1B model
                // The quantization is handled in the model files themselves
                LlamaConfig::load_llama3_2_1b::<InferenceBackend>(
                    model_path,
                    tokenizer_path,
                    self.config.max_seq_len,
                    &device,
                )
                .map_err(|e| {
                    LLMError::Generic(format!("Failed to load Llama3.2-1B-Q4 model: {}", e))
                })?
            }
        };

        Ok(Arc::new(LlamaChat {
            llama: Arc::new(CustomMutex::new(llama)),
            config: self.config.generation_config,
            marker: PhantomData,
            model: LLamaModel::Llama3,
        }))
    }

    pub async fn build_from_bytes_wasm(
        self,
    ) -> Result<Arc<LlamaChat<InferenceBackend, Tiktoken>>, LLMError> {
        #[cfg(all(target_arch = "wasm32", feature = "webgpu"))]
        {
            use crate::backend::burn_backend_types::init_setup;
            use crate::backend::burn_backend_types::INFERENCE_DEVICE;

            let device = INFERENCE_DEVICE;

            let _ = init_setup().await;

            info!("Building Llama3 model from bytes for WASM");
            info!("Model variant: {:?}", self.config.model_variant);

            let model_bytes = self
                .config
                .model_bytes
                .ok_or_else(|| LLMError::Generic("Model bytes not provided".to_string()))?;

            let tokenizer_bytes = self
                .config
                .tokenizer_bytes
                .ok_or_else(|| LLMError::Generic("Tokenizer bytes not provided".to_string()))?;

            // For WASM, we use a generic import function that works with bytes
            // The model variant determines which configuration to use
            let llama = match self.config.model_variant {
                Llama3Model::Llama3_8B => {
                    LlamaConfig::load_llama3_8b_from_bytes::<InferenceBackend>(
                        &model_bytes,
                        &tokenizer_bytes,
                        self.config.max_seq_len,
                        &device,
                    )
                    .map_err(|e| {
                        LLMError::Generic(format!("Failed to load Llama3-8B from bytes: {}", e))
                    })?
                }
                Llama3Model::Llama3_1_8B => {
                    LlamaConfig::load_llama3_1_8b_from_bytes::<InferenceBackend>(
                        &model_bytes,
                        &tokenizer_bytes,
                        self.config.max_seq_len,
                        &device,
                    )
                    .map_err(|e| {
                        LLMError::Generic(format!("Failed to load Llama3.1-8B from bytes: {}", e))
                    })?
                }
                Llama3Model::Llama3_2_3B => {
                    LlamaConfig::load_llama3_2_3b_from_bytes::<InferenceBackend>(
                        &model_bytes,
                        &tokenizer_bytes,
                        self.config.max_seq_len,
                        &device,
                    )
                    .map_err(|e| {
                        LLMError::Generic(format!("Failed to load Llama3.2-3B from bytes: {}", e))
                    })?
                }
                Llama3Model::Llama3_2_1B => {
                    LlamaConfig::load_llama3_2_1b_from_bytes::<InferenceBackend>(
                        &model_bytes,
                        &tokenizer_bytes,
                        self.config.max_seq_len,
                        &device,
                    )
                    .map_err(|e| {
                        LLMError::Generic(format!("Failed to load Llama3.2-1B from bytes: {}", e))
                    })?
                }
                Llama3Model::Llama3_2_1bQ4 => LlamaConfig::load_llama3_2_1b_from_bytes::<
                    InferenceBackend,
                >(
                    &model_bytes,
                    &tokenizer_bytes,
                    self.config.max_seq_len,
                    &device,
                )
                .map_err(|e| {
                    LLMError::Generic(format!("Failed to load Llama3.2-1B-Q4 from bytes: {}", e))
                })?,
            };

            return Ok(Arc::new(LlamaChat {
                llama: Arc::new(CustomMutex::new(llama)),
                config: self.config.generation_config,
                marker: PhantomData,
                model: LLamaModel::Llama3,
            }));
        }
        #[cfg(not(target_arch = "wasm32"))]
        {
            Err(LLMError::Generic("Does not support non WASM".to_string()))
        }
    }
}
