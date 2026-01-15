use crate::config::ModelConfig;

/// Generate a unique key for a model configuration
#[allow(dead_code)]
pub(crate) fn generate_model_key(config: &ModelConfig) -> String {
    use std::collections::hash_map::DefaultHasher;
    use std::hash::{Hash, Hasher};

    // Create a deterministic hash based on model configuration
    let mut hasher = DefaultHasher::new();

    config.provider.hash(&mut hasher);
    config.backend.kind.hash(&mut hasher);

    if let Some(model_name) = &config.model_name {
        model_name.hash(&mut hasher);
    }

    if let Some(source) = &config.source {
        source.hash(&mut hasher);
    }

    if let Some(params) = &config.parameters {
        if let Some(model_dir) = &params.model_dir {
            model_dir.hash(&mut hasher);
        }
        if let Some(quant) = &params.quant {
            quant.hash(&mut hasher);
        }
    }

    let hash = hasher.finish();

    // Create a readable key
    let provider = &config.provider;
    let model_id = config
        .model_name
        .as_ref()
        .or(config.source.as_ref())
        .map(|s| {
            // Take last part of path/repo
            s.split('/').next_back().unwrap_or(s).to_string()
        })
        .unwrap_or_else(|| "unknown".to_string());

    format!("{}_{}_{}", provider, model_id, &format!("{:x}", hash)[..8])
}
