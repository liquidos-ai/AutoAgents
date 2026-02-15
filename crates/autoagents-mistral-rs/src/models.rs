//! Model definitions and configurations for mistral.rs backend
//!
//! This module provides enums and structs for configuring both HuggingFace and GGUF models.

use serde::{Deserialize, Serialize};

/// Quantization types for GGUF models
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
#[allow(non_camel_case_types)]
pub enum GgufQuant {
    /// 4-bit quantization, medium quality
    Q4_K_M,
    /// 4-bit quantization, small quality
    Q4_K_S,
    /// 5-bit quantization, medium quality
    Q5_K_M,
    /// 5-bit quantization, small quality
    Q5_K_S,
    /// 8-bit quantization
    Q8_0,
    /// 16-bit floating point
    F16,
    /// 32-bit floating point
    F32,
}

impl GgufQuant {
    /// Get the file suffix for this quantization type
    pub fn file_suffix(&self) -> &'static str {
        match self {
            GgufQuant::Q4_K_M => "Q4_K_M",
            GgufQuant::Q4_K_S => "Q4_K_S",
            GgufQuant::Q5_K_M => "Q5_K_M",
            GgufQuant::Q5_K_S => "Q5_K_S",
            GgufQuant::Q8_0 => "Q8_0",
            GgufQuant::F16 => "F16",
            GgufQuant::F32 => "F32",
        }
    }
}

/// Model type for automatic builder selection
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
pub enum ModelType {
    /// Text-only model (uses TextModelBuilder)
    Text,
    /// Vision model (uses VisionModelBuilder)
    Vision,
    /// Auto-detect based on model name
    Auto,
}

/// Source type for loading models
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub enum ModelSource {
    /// HuggingFace model repository
    HuggingFace {
        /// Repository ID (e.g., "microsoft/Phi-3.5-mini-instruct")
        repo_id: String,
        /// Optional revision/branch
        revision: Option<String>,
        /// Model type (text, vision, or auto-detect)
        model_type: ModelType,
    },
    /// Local GGUF file(s)
    Gguf {
        /// Directory containing GGUF files
        model_dir: String,
        /// GGUF file names (can be multiple for sharded models)
        files: Vec<String>,
        /// Optional tokenizer (HF repo or local path)
        tokenizer: Option<String>,
        /// Optional chat template path
        chat_template: Option<String>,
    },
}

impl ModelSource {
    /// Detect if this is a vision model based on the repo ID
    pub fn detect_model_type(&self) -> ModelType {
        match self {
            ModelSource::HuggingFace {
                repo_id,
                model_type,
                ..
            } => {
                if *model_type != ModelType::Auto {
                    return *model_type;
                }

                let repo_lower = repo_id.to_lowercase();
                if repo_lower.contains("vlm")
                    || repo_lower.contains("vision")
                    || repo_lower.contains("gemma-3")
                    || repo_lower.contains("paligemma")
                    || repo_lower.contains("llava")
                    || repo_lower.contains("idefics")
                {
                    ModelType::Vision
                } else {
                    ModelType::Text
                }
            }
            ModelSource::Gguf { .. } => ModelType::Text, // GGUF always uses text builder
        }
    }
}

/// Pre-configured HF models
#[derive(Debug, Clone, PartialEq, Eq)]
#[allow(non_camel_case_types)]
pub enum HFModels {
    // ============================================================
    // HuggingFace Models
    // ============================================================
    /// Microsoft Phi-3.5-mini-instruct (3.8B parameters)
    Phi35MiniInstruct,

    /// Qwen 2.5 0.5B parameters
    Qwen25_05B,

    /// Qwen 2.5 1.5B parameters
    Qwen25_15B,

    /// Qwen 2.5 3B parameters
    Qwen25_3B,

    /// Meta Llama 3.2 1B parameters
    Llama32_1B,

    /// Meta Llama 3.2 3B parameters
    Llama32_3B,

    /// Mistral 7B Instruct v0.3
    Mistral7BInstruct,

    // ============================================================
    // SmolLM Models - Compact language models
    // ============================================================
    /// SmolLM 135M Instruct
    SmolLM_135M,

    /// SmolLM 360M Instruct
    SmolLM_360M,

    /// SmolLM 1.7B Instruct
    SmolLM_1_7B,

    /// SmolLM2 1.7B Instruct (Version 2)
    SmolLM2_1_7B,

    // ============================================================
    // StarCoder2 Models - Code generation models
    // ============================================================
    /// StarCoder2 3B - Code generation model
    StarCoder2_3B,

    /// StarCoder2 7B - Code generation model
    StarCoder2_7B,

    /// StarCoder2 15B - Code generation model
    StarCoder2_15B,

    // ============================================================
    // Qwen2.5-Coder Models - Specialized code models
    // ============================================================
    /// Qwen2.5-Coder 1.5B Instruct
    Qwen25Coder_1_5B,

    /// Qwen2.5-Coder 7B Instruct
    Qwen25Coder_7B,

    /// Qwen2.5-Coder 14B Instruct
    Qwen25Coder_14B,

    /// Qwen2.5-Coder 32B Instruct
    Qwen25Coder_32B,

    // ============================================================
    // Gemma Models - Google's lightweight models
    // ============================================================
    /// Gemma 2B Instruct (Version 1)
    Gemma_2B,

    /// Gemma 7B Instruct (Version 1)
    Gemma_7B,

    /// Gemma2 2B Instruct (Version 2)
    Gemma2_2B,

    /// Gemma2 9B Instruct (Version 2)
    Gemma2_9B,

    /// Gemma2 27B Instruct (Version 2)
    Gemma2_27B,

    // ============================================================
    // DeepSeek Models - Advanced reasoning models
    // ============================================================
    /// DeepSeek-V3 671B MoE model (37B active)
    DeepSeekV3,

    /// DeepSeek-V3.1 Hybrid model with thinking mode
    DeepSeekV3_1,

    // ============================================================
    // Vision Models - Multimodal models for vision + text
    // ============================================================
    /// SmolVLM - Compact vision-language model
    SmolVLM,

    /// Gemma3 12B IT - Multimodal model (vision + text)
    Gemma3_12B,

    /// Gemma3 27B IT - Multimodal model (vision + text)
    Gemma3_27B,

    // ============================================================
    // GGUF Models (Pre-configured with popular quantizations)
    // ============================================================
    /// Phi-3.5-mini-instruct GGUF variant
    Phi35MiniInstructGguf {
        /// Quantization type
        quant: GgufQuant,
        /// Model directory
        model_dir: String,
    },

    /// Mistral 7B Instruct GGUF variant
    Mistral7BInstructGguf {
        /// Quantization type
        quant: GgufQuant,
        /// Model directory
        model_dir: String,
    },

    /// Llama 3.2 1B GGUF variant
    Llama32_1BGguf {
        /// Quantization type
        quant: GgufQuant,
        /// Model directory
        model_dir: String,
    },

    /// Llama 3.2 3B GGUF variant
    Llama32_3BGguf {
        /// Quantization type
        quant: GgufQuant,
        /// Model directory
        model_dir: String,
    },

    /// Custom model source for full flexibility
    Custom(ModelSource),
}

impl HFModels {
    /// Convert to ModelSource
    pub fn to_source(&self) -> ModelSource {
        match self {
            // HuggingFace models
            HFModels::Phi35MiniInstruct => ModelSource::HuggingFace {
                repo_id: "microsoft/Phi-3.5-mini-instruct".to_string(),
                revision: None,
                model_type: ModelType::Text,
            },
            HFModels::Qwen25_05B => ModelSource::HuggingFace {
                repo_id: "Qwen/Qwen2.5-0.5B-Instruct".to_string(),
                revision: None,
                model_type: ModelType::Text,
            },
            HFModels::Qwen25_15B => ModelSource::HuggingFace {
                repo_id: "Qwen/Qwen2.5-1.5B-Instruct".to_string(),
                revision: None,
                model_type: ModelType::Text,
            },
            HFModels::Qwen25_3B => ModelSource::HuggingFace {
                repo_id: "Qwen/Qwen2.5-3B-Instruct".to_string(),
                revision: None,
                model_type: ModelType::Text,
            },
            HFModels::Llama32_1B => ModelSource::HuggingFace {
                repo_id: "meta-llama/Llama-3.2-1B-Instruct".to_string(),
                revision: None,
                model_type: ModelType::Text,
            },
            HFModels::Llama32_3B => ModelSource::HuggingFace {
                repo_id: "meta-llama/Llama-3.2-3B-Instruct".to_string(),
                revision: None,
                model_type: ModelType::Text,
            },
            HFModels::Mistral7BInstruct => ModelSource::HuggingFace {
                repo_id: "mistralai/Mistral-7B-Instruct-v0.3".to_string(),
                revision: None,
                model_type: ModelType::Text,
            },

            // SmolLM models
            HFModels::SmolLM_135M => ModelSource::HuggingFace {
                repo_id: "HuggingFaceTB/SmolLM-135M-Instruct".to_string(),
                revision: None,
                model_type: ModelType::Text,
            },
            HFModels::SmolLM_360M => ModelSource::HuggingFace {
                repo_id: "HuggingFaceTB/SmolLM-360M-Instruct".to_string(),
                revision: None,
                model_type: ModelType::Text,
            },
            HFModels::SmolLM_1_7B => ModelSource::HuggingFace {
                repo_id: "HuggingFaceTB/SmolLM-1.7B-Instruct".to_string(),
                revision: None,
                model_type: ModelType::Text,
            },
            HFModels::SmolLM2_1_7B => ModelSource::HuggingFace {
                repo_id: "HuggingFaceTB/SmolLM2-1.7B-Instruct".to_string(),
                revision: None,
                model_type: ModelType::Text,
            },

            // StarCoder2 models
            HFModels::StarCoder2_3B => ModelSource::HuggingFace {
                repo_id: "bigcode/starcoder2-3b".to_string(),
                revision: None,
                model_type: ModelType::Text,
            },
            HFModels::StarCoder2_7B => ModelSource::HuggingFace {
                repo_id: "bigcode/starcoder2-7b".to_string(),
                revision: None,
                model_type: ModelType::Text,
            },
            HFModels::StarCoder2_15B => ModelSource::HuggingFace {
                repo_id: "bigcode/starcoder2-15b".to_string(),
                revision: None,
                model_type: ModelType::Text,
            },

            // Qwen2.5-Coder models
            HFModels::Qwen25Coder_1_5B => ModelSource::HuggingFace {
                repo_id: "Qwen/Qwen2.5-Coder-1.5B-Instruct".to_string(),
                revision: None,
                model_type: ModelType::Text,
            },
            HFModels::Qwen25Coder_7B => ModelSource::HuggingFace {
                repo_id: "Qwen/Qwen2.5-Coder-7B-Instruct".to_string(),
                revision: None,
                model_type: ModelType::Text,
            },
            HFModels::Qwen25Coder_14B => ModelSource::HuggingFace {
                repo_id: "Qwen/Qwen2.5-Coder-14B-Instruct".to_string(),
                revision: None,
                model_type: ModelType::Text,
            },
            HFModels::Qwen25Coder_32B => ModelSource::HuggingFace {
                repo_id: "Qwen/Qwen2.5-Coder-32B-Instruct".to_string(),
                revision: None,
                model_type: ModelType::Text,
            },

            // Gemma models
            HFModels::Gemma_2B => ModelSource::HuggingFace {
                repo_id: "google/gemma-2b-it".to_string(),
                revision: None,
                model_type: ModelType::Text,
            },
            HFModels::Gemma_7B => ModelSource::HuggingFace {
                repo_id: "google/gemma-7b-it".to_string(),
                revision: None,
                model_type: ModelType::Text,
            },
            HFModels::Gemma2_2B => ModelSource::HuggingFace {
                repo_id: "google/gemma-2-2b-it".to_string(),
                revision: None,
                model_type: ModelType::Text,
            },
            HFModels::Gemma2_9B => ModelSource::HuggingFace {
                repo_id: "google/gemma-2-9b-it".to_string(),
                revision: None,
                model_type: ModelType::Text,
            },
            HFModels::Gemma2_27B => ModelSource::HuggingFace {
                repo_id: "google/gemma-2-27b-it".to_string(),
                revision: None,
                model_type: ModelType::Text,
            },

            // DeepSeek models
            HFModels::DeepSeekV3 => ModelSource::HuggingFace {
                repo_id: "deepseek-ai/DeepSeek-V3".to_string(),
                revision: None,
                model_type: ModelType::Text,
            },
            HFModels::DeepSeekV3_1 => ModelSource::HuggingFace {
                repo_id: "deepseek-ai/DeepSeek-V3.1".to_string(),
                revision: None,
                model_type: ModelType::Text,
            },

            // Vision models
            HFModels::SmolVLM => ModelSource::HuggingFace {
                repo_id: "HuggingFaceTB/SmolVLM-Instruct".to_string(),
                revision: None,
                model_type: ModelType::Vision,
            },
            HFModels::Gemma3_12B => ModelSource::HuggingFace {
                repo_id: "google/gemma-3-12b-it".to_string(),
                revision: None,
                model_type: ModelType::Vision,
            },
            HFModels::Gemma3_27B => ModelSource::HuggingFace {
                repo_id: "google/gemma-3-27b-it".to_string(),
                revision: None,
                model_type: ModelType::Vision,
            },

            // GGUF models
            HFModels::Phi35MiniInstructGguf { quant, model_dir } => {
                let filename = format!("Phi-3.5-mini-instruct-{}.gguf", quant.file_suffix());
                ModelSource::Gguf {
                    model_dir: model_dir.clone(),
                    files: vec![filename],
                    tokenizer: Some("microsoft/Phi-3.5-mini-instruct".to_string()),
                    chat_template: None,
                }
            }
            HFModels::Mistral7BInstructGguf { quant, model_dir } => {
                let filename = format!("mistral-7b-instruct-v0.3.{}.gguf", quant.file_suffix());
                ModelSource::Gguf {
                    model_dir: model_dir.clone(),
                    files: vec![filename],
                    tokenizer: Some("mistralai/Mistral-7B-Instruct-v0.3".to_string()),
                    chat_template: None,
                }
            }
            HFModels::Llama32_1BGguf { quant, model_dir } => {
                let filename = format!("Llama-3.2-1B-Instruct-{}.gguf", quant.file_suffix());
                ModelSource::Gguf {
                    model_dir: model_dir.clone(),
                    files: vec![filename],
                    tokenizer: Some("meta-llama/Llama-3.2-1B-Instruct".to_string()),
                    chat_template: None,
                }
            }
            HFModels::Llama32_3BGguf { quant, model_dir } => {
                let filename = format!("Llama-3.2-3B-Instruct-{}.gguf", quant.file_suffix());
                ModelSource::Gguf {
                    model_dir: model_dir.clone(),
                    files: vec![filename],
                    tokenizer: Some("meta-llama/Llama-3.2-3B-Instruct".to_string()),
                    chat_template: None,
                }
            }

            // Custom
            HFModels::Custom(source) => source.clone(),
        }
    }

    /// Get the model name for display
    pub fn name(&self) -> &str {
        match self {
            HFModels::Phi35MiniInstruct => "Phi-3.5-mini-instruct",
            HFModels::Qwen25_05B => "Qwen2.5-0.5B-Instruct",
            HFModels::Qwen25_15B => "Qwen2.5-1.5B-Instruct",
            HFModels::Qwen25_3B => "Qwen2.5-3B-Instruct",
            HFModels::Llama32_1B => "Llama-3.2-1B-Instruct",
            HFModels::Llama32_3B => "Llama-3.2-3B-Instruct",
            HFModels::Mistral7BInstruct => "Mistral-7B-Instruct-v0.3",

            // SmolLM models
            HFModels::SmolLM_135M => "SmolLM-135M-Instruct",
            HFModels::SmolLM_360M => "SmolLM-360M-Instruct",
            HFModels::SmolLM_1_7B => "SmolLM-1.7B-Instruct",
            HFModels::SmolLM2_1_7B => "SmolLM2-1.7B-Instruct",

            // StarCoder2 models
            HFModels::StarCoder2_3B => "StarCoder2-3B",
            HFModels::StarCoder2_7B => "StarCoder2-7B",
            HFModels::StarCoder2_15B => "StarCoder2-15B",

            // Qwen2.5-Coder models
            HFModels::Qwen25Coder_1_5B => "Qwen2.5-Coder-1.5B-Instruct",
            HFModels::Qwen25Coder_7B => "Qwen2.5-Coder-7B-Instruct",
            HFModels::Qwen25Coder_14B => "Qwen2.5-Coder-14B-Instruct",
            HFModels::Qwen25Coder_32B => "Qwen2.5-Coder-32B-Instruct",

            // Gemma models
            HFModels::Gemma_2B => "Gemma-2B-Instruct",
            HFModels::Gemma_7B => "Gemma-7B-Instruct",
            HFModels::Gemma2_2B => "Gemma2-2B-Instruct",
            HFModels::Gemma2_9B => "Gemma2-9B-Instruct",
            HFModels::Gemma2_27B => "Gemma2-27B-Instruct",

            // DeepSeek models
            HFModels::DeepSeekV3 => "DeepSeek-V3",
            HFModels::DeepSeekV3_1 => "DeepSeek-V3.1",

            // Vision models
            HFModels::SmolVLM => "SmolVLM-Instruct",
            HFModels::Gemma3_12B => "Gemma-3-12B-IT",
            HFModels::Gemma3_27B => "Gemma-3-27B-IT",

            HFModels::Phi35MiniInstructGguf { quant, .. } => match quant {
                GgufQuant::Q4_K_M => "Phi-3.5-mini-instruct (Q4_K_M)",
                GgufQuant::Q4_K_S => "Phi-3.5-mini-instruct (Q4_K_S)",
                GgufQuant::Q5_K_M => "Phi-3.5-mini-instruct (Q5_K_M)",
                GgufQuant::Q5_K_S => "Phi-3.5-mini-instruct (Q5_K_S)",
                GgufQuant::Q8_0 => "Phi-3.5-mini-instruct (Q8_0)",
                GgufQuant::F16 => "Phi-3.5-mini-instruct (F16)",
                GgufQuant::F32 => "Phi-3.5-mini-instruct (F32)",
            },
            HFModels::Mistral7BInstructGguf { quant, .. } => match quant {
                GgufQuant::Q4_K_M => "Mistral-7B-Instruct (Q4_K_M)",
                GgufQuant::Q4_K_S => "Mistral-7B-Instruct (Q4_K_S)",
                GgufQuant::Q5_K_M => "Mistral-7B-Instruct (Q5_K_M)",
                GgufQuant::Q5_K_S => "Mistral-7B-Instruct (Q5_K_S)",
                GgufQuant::Q8_0 => "Mistral-7B-Instruct (Q8_0)",
                GgufQuant::F16 => "Mistral-7B-Instruct (F16)",
                GgufQuant::F32 => "Mistral-7B-Instruct (F32)",
            },
            HFModels::Llama32_1BGguf { quant, .. } => match quant {
                GgufQuant::Q4_K_M => "Llama-3.2-1B (Q4_K_M)",
                GgufQuant::Q4_K_S => "Llama-3.2-1B (Q4_K_S)",
                GgufQuant::Q5_K_M => "Llama-3.2-1B (Q5_K_M)",
                GgufQuant::Q5_K_S => "Llama-3.2-1B (Q5_K_S)",
                GgufQuant::Q8_0 => "Llama-3.2-1B (Q8_0)",
                GgufQuant::F16 => "Llama-3.2-1B (F16)",
                GgufQuant::F32 => "Llama-3.2-1B (F32)",
            },
            HFModels::Llama32_3BGguf { quant, .. } => match quant {
                GgufQuant::Q4_K_M => "Llama-3.2-3B (Q4_K_M)",
                GgufQuant::Q4_K_S => "Llama-3.2-3B (Q4_K_S)",
                GgufQuant::Q5_K_M => "Llama-3.2-3B (Q5_K_M)",
                GgufQuant::Q5_K_S => "Llama-3.2-3B (Q5_K_S)",
                GgufQuant::Q8_0 => "Llama-3.2-3B (Q8_0)",
                GgufQuant::F16 => "Llama-3.2-3B (F16)",
                GgufQuant::F32 => "Llama-3.2-3B (F32)",
            },
            HFModels::Custom(_) => "Custom Model",
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn gguf_quant_suffixes() {
        let cases = [
            (GgufQuant::Q4_K_M, "Q4_K_M"),
            (GgufQuant::Q4_K_S, "Q4_K_S"),
            (GgufQuant::Q5_K_M, "Q5_K_M"),
            (GgufQuant::Q5_K_S, "Q5_K_S"),
            (GgufQuant::Q8_0, "Q8_0"),
            (GgufQuant::F16, "F16"),
            (GgufQuant::F32, "F32"),
        ];
        for (quant, expected) in cases {
            assert_eq!(quant.file_suffix(), expected);
        }
    }

    #[test]
    fn detect_model_type_respects_override() {
        let source = ModelSource::HuggingFace {
            repo_id: "org/vision-model".to_string(),
            revision: None,
            model_type: ModelType::Text,
        };
        assert_eq!(source.detect_model_type(), ModelType::Text);
    }

    #[test]
    fn detect_model_type_auto_vision() {
        let source = ModelSource::HuggingFace {
            repo_id: "acme/gemma-3-vision".to_string(),
            revision: None,
            model_type: ModelType::Auto,
        };
        assert_eq!(source.detect_model_type(), ModelType::Vision);
    }

    #[test]
    fn hf_models_to_source_and_name() {
        let hf = HFModels::Phi35MiniInstruct;
        let source = hf.to_source();
        assert!(matches!(source, ModelSource::HuggingFace { .. }));
        assert_eq!(hf.name(), "Phi-3.5-mini-instruct");

        let gguf = HFModels::Phi35MiniInstructGguf {
            quant: GgufQuant::Q4_K_M,
            model_dir: "/models".to_string(),
        };
        let source = gguf.to_source();
        match source {
            ModelSource::Gguf {
                model_dir, files, ..
            } => {
                assert_eq!(model_dir, "/models");
                assert_eq!(files.len(), 1);
                assert!(files[0].contains("Q4_K_M"));
            }
            _ => panic!("expected GGUF source"),
        }
        assert_eq!(gguf.name(), "Phi-3.5-mini-instruct (Q4_K_M)");

        let custom = HFModels::Custom(ModelSource::HuggingFace {
            repo_id: "org/custom".to_string(),
            revision: Some("rev".to_string()),
            model_type: ModelType::Text,
        });
        let source = custom.to_source();
        assert!(matches!(source, ModelSource::HuggingFace { .. }));
    }

    #[test]
    fn hf_models_cover_all_variants() {
        let hf_variants = vec![
            HFModels::Phi35MiniInstruct,
            HFModels::Qwen25_05B,
            HFModels::Qwen25_15B,
            HFModels::Qwen25_3B,
            HFModels::Llama32_1B,
            HFModels::Llama32_3B,
            HFModels::Mistral7BInstruct,
            HFModels::SmolLM_135M,
            HFModels::SmolLM_360M,
            HFModels::SmolLM_1_7B,
            HFModels::SmolLM2_1_7B,
            HFModels::StarCoder2_3B,
            HFModels::StarCoder2_7B,
            HFModels::StarCoder2_15B,
            HFModels::Qwen25Coder_1_5B,
            HFModels::Qwen25Coder_7B,
            HFModels::Qwen25Coder_14B,
            HFModels::Qwen25Coder_32B,
            HFModels::Gemma_2B,
            HFModels::Gemma_7B,
            HFModels::Gemma2_2B,
            HFModels::Gemma2_9B,
            HFModels::Gemma2_27B,
            HFModels::DeepSeekV3,
            HFModels::DeepSeekV3_1,
            HFModels::SmolVLM,
            HFModels::Gemma3_12B,
            HFModels::Gemma3_27B,
        ];

        for model in hf_variants {
            let name = model.name();
            assert!(!name.is_empty());
            let source = model.to_source();
            match source {
                ModelSource::HuggingFace { repo_id, .. } => {
                    assert!(!repo_id.is_empty());
                }
                ModelSource::Gguf { .. } => {
                    panic!("expected HuggingFace source for {name}");
                }
            }
        }

        let quantizations = [
            GgufQuant::Q4_K_M,
            GgufQuant::Q4_K_S,
            GgufQuant::Q5_K_M,
            GgufQuant::Q5_K_S,
            GgufQuant::Q8_0,
            GgufQuant::F16,
            GgufQuant::F32,
        ];

        for quant in quantizations {
            let gguf_models = [
                HFModels::Phi35MiniInstructGguf {
                    quant,
                    model_dir: "/models".to_string(),
                },
                HFModels::Mistral7BInstructGguf {
                    quant,
                    model_dir: "/models".to_string(),
                },
                HFModels::Llama32_1BGguf {
                    quant,
                    model_dir: "/models".to_string(),
                },
                HFModels::Llama32_3BGguf {
                    quant,
                    model_dir: "/models".to_string(),
                },
            ];

            for model in gguf_models {
                let name = model.name();
                assert!(name.contains(quant.file_suffix()));
                let source = model.to_source();
                match source {
                    ModelSource::Gguf {
                        model_dir, files, ..
                    } => {
                        assert_eq!(model_dir, "/models");
                        assert_eq!(files.len(), 1);
                        assert!(files[0].contains(quant.file_suffix()));
                    }
                    ModelSource::HuggingFace { .. } => {
                        panic!("expected GGUF source for {name}");
                    }
                }
            }
        }
    }

    #[test]
    fn detect_model_type_auto_text_and_vision_keywords() {
        let text_source = ModelSource::HuggingFace {
            repo_id: "org/regular-model".to_string(),
            revision: None,
            model_type: ModelType::Auto,
        };
        assert_eq!(text_source.detect_model_type(), ModelType::Text);

        let vision_source = ModelSource::HuggingFace {
            repo_id: "acme/llava-1.5".to_string(),
            revision: None,
            model_type: ModelType::Auto,
        };
        assert_eq!(vision_source.detect_model_type(), ModelType::Vision);
    }
}
