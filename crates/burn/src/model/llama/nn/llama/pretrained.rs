use burn::prelude::*;

use super::{Llama, LlamaConfig, LlamaVersion, TinyLlamaVersion};

#[cfg(feature = "llama3")]
use crate::model::llama::tokenizer::Tiktoken;

#[cfg(feature = "tiny")]
use crate::model::llama::tokenizer::SentencePieceTokenizer;

/// Pre-trained model metadata.
pub struct Pretrained {
    pub(super) name: &'static str,
    pub(super) model: &'static str,
    pub(super) tokenizer: &'static str,
}

mod downloader {
    use super::*;
    use burn::data::network::downloader;
    use std::fs::{create_dir_all, File};
    use std::io::Write;
    use std::path::PathBuf;

    impl Pretrained {
        fn model_dir(&self) -> PathBuf {
            dirs::home_dir()
                .expect("Should be able to get home directory")
                .join(".cache")
                .join("llama")
                .join(self.name)
        }

        fn model_file_name(&self, url: &str) -> String {
            url.rsplit_once('/')
                .unwrap()
                .1
                .replace("?download=true", "")
        }

        pub fn is_downloaded(&self) -> bool {
            let model_name = self.model_dir().join(self.model_file_name(self.model));
            let tokenizer_name = self.model_dir().join(self.model_file_name(self.tokenizer));
            model_name.exists() && tokenizer_name.exists()
        }

        /// Download the file to the local cache directory.
        fn download(&self, url: &str) -> Result<PathBuf, std::io::Error> {
            // Model cache directory
            let model_dir = self.model_dir();

            if !model_dir.exists() {
                create_dir_all(&model_dir)?;
            }

            let file_base_name = self.model_file_name(url);
            let file_name = model_dir.join(&file_base_name);
            if !file_name.exists() {
                // Download file content
                let bytes = downloader::download_file_as_bytes(url, &file_base_name);

                // Write content to file
                let mut output_file = File::create(&file_name)?;
                output_file.write_all(&bytes)?; // write_all is not OS limited (files over 2GB)
            }

            Ok(file_name)
        }

        /// Delete the file to the local cache directory.
        fn delete(&self, url: &str) -> Result<(), std::io::Error> {
            let model_dir = self.model_dir();
            if !model_dir.exists() {
                return Ok(());
            }
            let file_base_name = self.model_file_name(url);
            let file_name = model_dir.join(&file_base_name);
            if file_name.exists() {
                std::fs::remove_file(file_name)
                    .unwrap_or_else(|_| panic!("should delete model file '{file_base_name}'"));
            }
            Ok(())
        }

        /// Download the pre-trained model weights to the local cache directory.
        pub fn download_weights(&self) -> Result<PathBuf, std::io::Error> {
            self.download(self.model)
        }

        /// Delete the tokenizer to the local cache directory.
        pub fn download_tokenizer(&self) -> Result<PathBuf, std::io::Error> {
            self.download(self.tokenizer)
        }

        /// Delete the pre-trained model weights from the local cache directory.
        pub fn delete_weights(&self) -> Result<(), std::io::Error> {
            self.delete(self.model)
        }

        /// Delete the tokenizer from the local cache directory.
        pub fn delete_tokenizer(&self) -> Result<(), std::io::Error> {
            self.delete(self.tokenizer)
        }
    }
}

pub trait ModelMeta {
    fn pretrained(&self) -> Pretrained;
}

impl ModelMeta for LlamaVersion {
    fn pretrained(&self) -> Pretrained {
        match self {
            Self::Llama3Instruct => Pretrained {
                name: "Llama-3-8B-Instruct",
                model: "https://huggingface.co/tracel-ai/llama-3-8b-instruct-burn/resolve/main/model.mpk?download=true",
                tokenizer: "https://huggingface.co/tracel-ai/llama-3-8b-instruct-burn/resolve/main/tokenizer.model?download=true",
            },
            Self::Llama31Instruct => Pretrained {
                name: "Llama-3.1-8B-Instruct",
                model: "https://huggingface.co/tracel-ai/llama-3.1-8b-instruct-burn/resolve/main/model.mpk?download=true",
                tokenizer: "https://huggingface.co/tracel-ai/llama-3.1-8b-instruct-burn/resolve/main/tokenizer.model?download=true",
            },
            Self::Llama323bInstruct => Pretrained {
                name: "Llama-3.2-3B-Instruct",
                model: "https://huggingface.co/tracel-ai/llama-3.2-3b-instruct-burn/resolve/main/model.mpk?download=true",
                tokenizer: "https://huggingface.co/tracel-ai/llama-3.2-3b-instruct-burn/resolve/main/tokenizer.model?download=true",
            },
            Self::Llama321bInstruct => Pretrained {
                name: "Llama-3.2-1B-Instruct",
                model: "https://huggingface.co/tracel-ai/llama-3.2-1b-instruct-burn/resolve/main/model.mpk?download=true",
                tokenizer: "https://huggingface.co/tracel-ai/llama-3.2-1b-instruct-burn/resolve/main/tokenizer.model?download=true",
            },
        }
    }
}

impl ModelMeta for TinyLlamaVersion {
    fn pretrained(&self) -> Pretrained {
        match self {
            TinyLlamaVersion::V1 => Pretrained {
                name: "TinyLlama-1.1B",
                model: "https://huggingface.co/tracel-ai/tiny-llama-1.1b-burn/resolve/main/model.mpk?download=true",
                tokenizer: "https://huggingface.co/tracel-ai/tiny-llama-1.1b-burn/resolve/main/tokenizer.json?download=true",
            }
        }
    }
}

fn check_context_length(max_seq_len: usize, max_context_len: usize) {
    assert!(
        max_seq_len <= max_context_len,
        "Maximum sequence length must not exceed {max_context_len}"
    );
}

impl LlamaConfig {
    /// Load pre-trained Llama-3.2-3B-Instruct model with [Tiktoken](https://github.com/openai/tiktoken) tokenizer.
    ///
    /// # Arguments
    /// - `max_seq_len` - The maximum sequence length for input text.
    /// - `device` - The device to load the model on.
    #[cfg(feature = "llama3")]
    pub fn llama3_2_3b_pretrained<B: Backend>(
        max_seq_len: usize,
        device: &Device<B>,
    ) -> Result<Llama<B, Tiktoken>, String> {
        // Llama-3.2 models support context length up to 128K tokens.
        check_context_length(max_seq_len, 128 * 1024);

        // Download checkpoint and tokenizer
        let model = LlamaVersion::Llama323bInstruct.pretrained();
        let checkpoint = model
            .download_weights()
            .map_err(|err| format!("Could not download weights.\nError: {err}"))?;
        let tokenizer = model
            .download_tokenizer()
            .map_err(|err| format!("Could not download tokenizer.\nError: {err}"))?;

        Self::load_llama3_2_3b(
            checkpoint.to_str().unwrap(),
            tokenizer.to_str().unwrap(),
            max_seq_len,
            device,
        )
    }

    /// Load pre-trained Llama-3.2-3B-Instruct model with [Tiktoken](https://github.com/openai/tiktoken) tokenizer.
    ///
    /// # Arguments
    /// - `max_seq_len` - The maximum sequence length for input text.
    /// - `device` - The device to load the model on.
    #[cfg(feature = "llama3")]
    pub fn llama3_2_1b_pretrained<B: Backend>(
        max_seq_len: usize,
        device: &Device<B>,
    ) -> Result<Llama<B, Tiktoken>, String> {
        // Llama-3.2 models support context length up to 128K tokens.
        check_context_length(max_seq_len, 128 * 1024);

        // Download checkpoint and tokenizer
        let model = LlamaVersion::Llama321bInstruct.pretrained();
        let checkpoint = model
            .download_weights()
            .map_err(|err| format!("Could not download weights.\nError: {err}"))?;
        let tokenizer = model
            .download_tokenizer()
            .map_err(|err| format!("Could not download tokenizer.\nError: {err}"))?;

        Self::load_llama3_2_1b(
            checkpoint.to_str().unwrap(),
            tokenizer.to_str().unwrap(),
            max_seq_len,
            device,
        )
    }

    /// Load pre-trained Llama-3.1-8B-Instruct model with [Tiktoken](https://github.com/openai/tiktoken) tokenizer.
    ///
    /// # Arguments
    /// - `max_seq_len` - The maximum sequence length for input text.
    /// - `device` - The device to load the model on.
    #[cfg(feature = "llama3")]
    pub fn llama3_1_8b_pretrained<B: Backend>(
        max_seq_len: usize,
        device: &Device<B>,
    ) -> Result<Llama<B, Tiktoken>, String> {
        // Llama-3.1 models support context length up to 128K tokens.
        check_context_length(max_seq_len, 128 * 1024);

        // Download checkpoint and tokenizer
        let model = LlamaVersion::Llama31Instruct.pretrained();
        let checkpoint = model
            .download_weights()
            .map_err(|err| format!("Could not download weights.\nError: {err}"))?;
        let tokenizer = model
            .download_tokenizer()
            .map_err(|err| format!("Could not download tokenizer.\nError: {err}"))?;

        Self::load_llama3_1_8b(
            checkpoint.to_str().unwrap(),
            tokenizer.to_str().unwrap(),
            max_seq_len,
            device,
        )
    }

    /// Load pre-trained Llama-3-8B-Instruct model with [Tiktoken](https://github.com/openai/tiktoken) tokenizer.
    ///
    /// # Arguments
    /// - `max_seq_len` - The maximum sequence length for input text.
    /// - `device` - The device to load the model on.
    #[cfg(feature = "llama3")]
    pub fn llama3_8b_pretrained<B: Backend>(
        max_seq_len: usize,
        device: &Device<B>,
    ) -> Result<Llama<B, Tiktoken>, String> {
        // Llama-3 models support context length up to 8K tokens.
        check_context_length(max_seq_len, 8 * 1024);

        // Download checkpoint and tokenizer
        let model = LlamaVersion::Llama3Instruct.pretrained();
        let checkpoint = model
            .download_weights()
            .map_err(|err| format!("Could not download weights.\nError: {err}"))?;
        let tokenizer = model
            .download_tokenizer()
            .map_err(|err| format!("Could not download tokenizer.\nError: {err}"))?;

        Self::load_llama3_8b(
            checkpoint.to_str().unwrap(),
            tokenizer.to_str().unwrap(),
            max_seq_len,
            device,
        )
    }

    /// Load pre-trained TinyLlama-1.1B Chat v1.0 model with [SentenciePiece](https://github.com/google/sentencepiece) tokenizer.
    #[cfg(feature = "tiny")]
    pub fn tiny_llama_pretrained<B: Backend>(
        max_seq_len: usize,
        device: &Device<B>,
    ) -> Result<Llama<B, SentencePieceTokenizer>, String> {
        // TinyLlama models support context length up to 2K tokens.

        check_context_length(max_seq_len, 2 * 1024);

        // Download checkpoint and tokenizer
        let model = TinyLlamaVersion::V1.pretrained();
        let checkpoint = model
            .download_weights()
            .map_err(|err| format!("Could not download weights.\nError: {err}"))?;
        let tokenizer = model
            .download_tokenizer()
            .map_err(|err| format!("Could not download tokenizer.\nError: {err}"))?;

        Self::load_tiny_llama(
            checkpoint.to_str().unwrap(),
            tokenizer.to_str().unwrap(),
            max_seq_len,
            device,
        )
    }
}
