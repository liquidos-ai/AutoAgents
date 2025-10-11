use crate::backend::burn_backend_types::InferenceBackend;
use crate::model::llama::chat::{GenerationConfig, LLamaModel, LlamaChat};
use crate::model::llama::tokenizer::SentencePieceTokenizer;
use crate::model::llama::{Llama, LlamaConfig};
use crate::utils::CustomMutex;
use autoagents_llm::error::LLMError;
use log::info;
use std::marker::PhantomData;
use std::path::PathBuf;
use std::sync::Arc;

pub struct TinyLLamaModelConfig {
    pub model_path: PathBuf,
    pub tokenizer_path: PathBuf,
    pub max_seq_len: usize,
    pub generation_config: GenerationConfig,
    #[allow(dead_code)]
    pub import: bool,
    #[allow(dead_code)]
    pub model_bytes: Option<Vec<u8>>,
    #[allow(dead_code)]
    pub tokenizer_bytes: Option<Vec<u8>>,
}

impl Default for TinyLLamaModelConfig {
    fn default() -> Self {
        Self {
            model_path: PathBuf::from("models/tinyllama.mpk"),
            tokenizer_path: PathBuf::from("models/tokenizer.model"),
            max_seq_len: 512,
            generation_config: GenerationConfig::default(),
            import: false,
            model_bytes: None,
            tokenizer_bytes: None,
        }
    }
}

pub struct TinyLlamaBuilder {
    config: TinyLLamaModelConfig,
}

impl TinyLlamaBuilder {
    pub fn new() -> Self {
        Self {
            config: TinyLLamaModelConfig::default(),
        }
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

    pub fn with_model_bytes(self, _bytes: Vec<u8>) -> Self {
        #[cfg(all(feature = "import", target_arch = "wasm32"))]
        {
            self.config.model_bytes = Some(bytes);
            self.config.import = true;
            self
        }
        #[cfg(not(target_arch = "wasm32"))]
        {
            panic!("Model weights is supported only in wasm32");
        }
    }

    pub fn with_tokenizer_bytes(self, _bytes: Vec<u8>) -> Self {
        #[cfg(all(feature = "import", target_arch = "wasm32"))]
        {
            self.config.tokenizer_bytes = Some(bytes);
            self
        }
        #[cfg(not(target_arch = "wasm32"))]
        {
            panic!("Tokenizer bytes is supported only in wasm32");
        }
    }

    #[cfg(feature = "pretrained")]
    pub async fn build_from_pretrained(
        self,
    ) -> Result<Arc<LlamaChat<InferenceBackend, SentencePieceTokenizer>>, LLMError> {
        use crate::backend::burn_backend_types::INFERENCE_DEVICE;

        let device = INFERENCE_DEVICE;

        info!(
            "Burn using device {}",
            crate::backend::burn_backend_types::NAME
        );

        // Use tokio::task::spawn_blocking to run the blocking pretrained loading in a separate thread
        let max_seq_len = self.config.max_seq_len;
        let llama = tokio::task::spawn_blocking(move || {
            LlamaConfig::tiny_llama_pretrained::<InferenceBackend>(max_seq_len, &device)
        })
        .await
        .map_err(|e| LLMError::Generic(format!("Task join error: {}", e)))?
        .map_err(|e| LLMError::Generic(format!("Failed to load model: {}", e)))?;

        Ok(Arc::new(LlamaChat {
            llama: Arc::new(CustomMutex::new(llama)),
            config: self.config.generation_config,
            marker: PhantomData,
            model: LLamaModel::TinyLLama,
        }))
    }

    pub fn build(
        self,
    ) -> Result<Arc<LlamaChat<InferenceBackend, SentencePieceTokenizer>>, LLMError> {
        use crate::backend::burn_backend_types::INFERENCE_DEVICE;

        let device = INFERENCE_DEVICE;

        info!(
            "Burn using device {}",
            crate::backend::burn_backend_types::NAME
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

        let llama: Llama<InferenceBackend, _> = {
            #[cfg(all(feature = "import", target_arch = "wasm32"))]
            {
                if self.config.import {
                    use crate::model::llama::tokenizer::SentencePieceTokenizer;

                    let model_bytes = self.config.model_bytes.ok_or_else(|| {
                        LLMError::Generic("Model bytes required for WASM import".to_string())
                    })?;
                    let tokenizer_bytes = self.config.tokenizer_bytes.ok_or_else(|| {
                        LLMError::Generic("Tokenizer bytes required for WASM".to_string())
                    })?;

                    let tokenizer =
                        SentencePieceTokenizer::from_bytes(&tokenizer_bytes).map_err(|e| {
                            LLMError::Generic(format!("Failed to load tokenizer: {}", e))
                        })?;

                    let config =
                        LlamaConfig::tiny_llama("").with_max_seq_len(self.config.max_seq_len);

                    config
                        .load_pretrained_with_tokenizer(&model_bytes, tokenizer, &device)
                        .map_err(|e| LLMError::Generic(format!("Failed to load model: {}", e)))?
                } else {
                    let model_bytes = self.config.model_bytes.ok_or_else(|| {
                        LLMError::Generic("Model bytes required for WASM".to_string())
                    })?;
                    let tokenizer_bytes = self.config.tokenizer_bytes.ok_or_else(|| {
                        LLMError::Generic("Tokenizer bytes required for WASM".to_string())
                    })?;

                    LlamaConfig::load_tiny_llama_from_bytes::<InferenceBackend>(
                        &model_bytes,
                        &tokenizer_bytes,
                        self.config.max_seq_len,
                        &device,
                    )
                    .map_err(|e| LLMError::Generic(format!("Failed to load model: {}", e)))?
                }
            }

            #[cfg(all(feature = "import", not(target_arch = "wasm32")))]
            {
                if self.config.import {
                    let config = LlamaConfig::tiny_llama(tokenizer_path)
                        .with_max_seq_len(self.config.max_seq_len);
                    config
                        .load_pretrained::<InferenceBackend, _>(model_path, &device)
                        .map_err(|e| LLMError::Generic(format!("Failed to load model: {}", e)))?
                } else {
                    LlamaConfig::load_tiny_llama::<InferenceBackend>(
                        model_path,
                        tokenizer_path,
                        self.config.max_seq_len,
                        &device,
                    )
                    .map_err(|e| LLMError::Generic(format!("Failed to load model: {}", e)))?
                }
            }

            #[cfg(not(feature = "import"))]
            {
                #[cfg(not(target_arch = "wasm32"))]
                {
                    LlamaConfig::load_tiny_llama::<InferenceBackend>(
                        model_path,
                        tokenizer_path,
                        self.config.max_seq_len,
                        &device,
                    )
                    .map_err(|e| LLMError::Generic(format!("Failed to load model: {}", e)))?
                }

                #[cfg(target_arch = "wasm32")]
                {
                    let model_bytes = self.config.model_bytes.ok_or_else(|| {
                        LLMError::Generic("Model bytes required for WASM".to_string())
                    })?;
                    let tokenizer_bytes = self.config.tokenizer_bytes.ok_or_else(|| {
                        LLMError::Generic("Tokenizer bytes required for WASM".to_string())
                    })?;

                    LlamaConfig::load_tiny_llama_from_bytes::<InferenceBackend>(
                        &model_bytes,
                        &tokenizer_bytes,
                        self.config.max_seq_len,
                        &device,
                    )
                    .map_err(|e| LLMError::Generic(format!("Failed to load model: {}", e)))?
                }
            }
        };

        Ok(Arc::new(LlamaChat {
            llama: Arc::new(CustomMutex::new(llama)),
            config: self.config.generation_config,
            marker: PhantomData,
            model: LLamaModel::TinyLLama,
        }))
    }
}
