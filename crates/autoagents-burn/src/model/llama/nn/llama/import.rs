use super::{Llama, LlamaConfig};
use crate::model::llama::nn::transformer::TransformerRecord;
use crate::model::llama::tokenizer::Tokenizer;
use burn::record::Recorder;
use burn::{prelude::*, record::HalfPrecisionSettings};
#[cfg(not(target_arch = "wasm32"))]
use burn_import::pytorch::{LoadArgs, PyTorchFileRecorder};

impl LlamaConfig {
    /// Load pre-trained Llama checkpoint.
    #[cfg(not(target_arch = "wasm32"))]
    pub fn load_pretrained<B: Backend, T: Tokenizer>(
        &self,
        checkpoint: &str,
        device: &Device<B>,
    ) -> Result<Llama<B, T>, String> {
        let mut llama = self.init(device)?;

        // Load weights from torch state_dict
        let mut load_args = LoadArgs::new(checkpoint.into());

        if !cfg!(feature = "tiny") {
            load_args = load_args
                // Map layers.[i].feed_forward.w1.* -> layers.[i].feed_forward.swiglu.linear_inner.*
                .with_key_remap(
                    "(layers\\.[0-9]+\\.feed_forward)\\.w1\\.(.+)",
                    "$1.swiglu.linear_inner.$2",
                )
                // Map layers.[i].feed_forward.w3.* -> layers.[i].feed_forward.swiglu.linear_outer.*
                .with_key_remap(
                    "(layers\\.[0-9]+\\.feed_forward)\\.w3\\.(.+)",
                    "$1.swiglu.linear_outer.$2",
                )
                // Map norm.weight -> norm.gamma for all layers
                .with_key_remap("(.*)norm\\.weight", "${1}norm.gamma");
        } else {
            load_args = load_args
                // Map lm_head.* -> output.*
                .with_key_remap("lm_head\\.(.+)", "output.$1")
                // Remove model. prefix
                .with_key_remap("model\\.(.+)", "$1")
                // Map embed_tokens.* -> tok_embeddings.*
                .with_key_remap("embed_tokens\\.(.+)", "tok_embeddings.$1")
                // Map layers.[i].input_layernorm.* -> layers.[i].attention_norm.*
                .with_key_remap(
                    "(layers\\.[0-9]+)\\.input_layernorm\\.(.+)",
                    "$1.attention_norm.$2",
                )
                // Map layers.[i].post_attention_layernorm.* -> layers.[i].ffn_norm.*
                .with_key_remap(
                    "(layers\\.[0-9]+)\\.post_attention_layernorm\\.(.+)",
                    "$1.ffn_norm.$2",
                )
                // Map layers.[i].mlp.down_proj.* -> layers.[i].feed_forward.w2.*
                .with_key_remap(
                    "(layers\\.[0-9]+)\\.mlp\\.down_proj\\.(.+)",
                    "$1.feed_forward.w2.$2",
                )
                // Map layers.[i].mlp.gate_proj.* -> layers.[i].feed_forward.swiglu.linear_inner.*
                .with_key_remap(
                    "(layers\\.[0-9]+)\\.mlp\\.gate_proj\\.(.+)",
                    "$1.feed_forward.swiglu.linear_inner.$2",
                )
                // Map layers.[i].mlp.up_proj.* -> layers.[i].feed_forward.swiglu.linear_outer.*
                .with_key_remap(
                    "(layers\\.[0-9]+)\\.mlp\\.up_proj\\.(.+)",
                    "$1.feed_forward.swiglu.linear_outer.$2",
                )
                // Map layers.[i].self_attn.k_proj.* -> layers.[i].attention.wk.*
                .with_key_remap(
                    "(layers\\.[0-9]+)\\.self_attn\\.k_proj\\.(.+)",
                    "$1.attention.wk.$2",
                )
                // Map layers.[i].self_attn.o_proj.* -> layers.[i].attention.wo.*
                .with_key_remap(
                    "(layers\\.[0-9]+)\\.self_attn\\.o_proj\\.(.+)",
                    "$1.attention.wo.$2",
                )
                // Map layers.[i].self_attn.q_proj.* -> layers.[i].attention.wq.*
                .with_key_remap(
                    "(layers\\.[0-9]+)\\.self_attn\\.q_proj\\.(.+)",
                    "$1.attention.wq.$2",
                )
                // Map layers.[i].self_attn.v_proj.* -> layers.[i].attention.wv.*
                .with_key_remap(
                    "(layers\\.[0-9]+)\\.self_attn\\.v_proj\\.(.+)",
                    "$1.attention.wv.$2",
                )
                // Map norm.weight -> norm.gamma for all layers
                .with_key_remap("(.*)norm\\.weight", "${1}norm.gamma");
        }
        let mut record: TransformerRecord<B> = PyTorchFileRecorder::<HalfPrecisionSettings>::new()
            .load(load_args, device)
            .map_err(|e| e.to_string())?;

        if cfg!(feature = "tiny") {
            // TinyLlama weights from HuggingFace use a different rotary positional encoding
            // which requires weight permutation:
            // https://github.com/huggingface/transformers/issues/25199#issuecomment-1687720247
            // https://github.com/jzhang38/TinyLlama/issues/24
            let n_heads = self.num_attention_heads;
            let n_kv_heads = self.num_key_value_heads.unwrap_or(n_heads);
            let wk_dim = self.d_model * n_kv_heads / n_heads;
            let permute = |w: Tensor<B, 2>, n_heads: usize, dim1: usize, dim2: usize| {
                w // [2048, 256]
                    .reshape([dim1, n_heads, 2, dim2 / n_heads / 2]) // [2048, 4, 2, 32]
                    .swap_dims(2, 3) // [2048, 4, 32, 2]
                    .reshape([dim1, dim2])
            };

            record.layers = record
                .layers
                .into_iter()
                .map(|mut layer| {
                    layer.attention.wq.weight = layer
                        .attention
                        .wq
                        .weight
                        .map(|w| permute(w, n_heads, self.d_model, self.d_model));
                    layer.attention.wk.weight = layer
                        .attention
                        .wk
                        .weight
                        .map(|w| permute(w, n_kv_heads, self.d_model, wk_dim));
                    layer
                })
                .collect::<Vec<_>>();
        }

        llama.model = llama.model.load_record(record);

        Ok(llama)
    }

    /// Load pre-trained Llama checkpoint for WASM targets.
    #[cfg(target_arch = "wasm32")]
    pub fn load_pretrained_with_tokenizer<B: Backend, T: Tokenizer>(
        &self,
        checkpoint_bytes: &[u8],
        tokenizer: T,
        device: &Device<B>,
    ) -> Result<Llama<B, T>, String> {
        use burn::record::NamedMpkBytesRecorder;

        let mut llama = self.init_with_tokenizer(tokenizer, device)?;

        // Load the model weights from bytes
        let recorder = NamedMpkBytesRecorder::<HalfPrecisionSettings>::new();
        let record: TransformerRecord<B> = recorder
            .load(checkpoint_bytes.to_vec(), device)
            .map_err(|e| format!("Failed to load model from bytes: {}", e))?;

        // Apply TinyLlama-specific weight permutations if needed
        let record = if cfg!(feature = "tiny") {
            let n_heads = self.num_attention_heads;
            let n_kv_heads = self.num_key_value_heads.unwrap_or(n_heads);
            let wk_dim = self.d_model * n_kv_heads / n_heads;
            let permute = |w: Tensor<B, 2>, n_heads: usize, dim1: usize, dim2: usize| {
                let w = w
                    .reshape([dim1, n_heads, 2, dim2 / n_heads / 2])
                    .swap_dims(2, 3)
                    .reshape([dim1, dim2]);
                w
            };

            TransformerRecord {
                layers: record
                    .layers
                    .into_iter()
                    .map(|mut layer| {
                        layer.attention.wq.weight = layer
                            .attention
                            .wq
                            .weight
                            .map(|w| permute(w, n_heads, self.d_model, self.d_model));
                        layer.attention.wk.weight = layer
                            .attention
                            .wk
                            .weight
                            .map(|w| permute(w, n_kv_heads, self.d_model, wk_dim));
                        layer
                    })
                    .collect::<Vec<_>>(),
                ..record
            }
        } else {
            record
        };

        llama.model = llama.model.load_record(record);

        Ok(llama)
    }
}
