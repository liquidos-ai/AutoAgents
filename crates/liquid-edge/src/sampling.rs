#![allow(dead_code)]
//! Sampling strategies for text generation
//!
//! This module provides various sampling strategies for controlling text generation
//! behavior. Each strategy implements different approaches to selecting the next
//! token from model output logits.
//!
//! ## Available Strategies
//!
//! - **Greedy**: Always select the token with highest probability (deterministic)
//! - **Temperature**: Apply temperature scaling for controlled randomness
//! - **Top-K**: Sample from the K most likely tokens
//! - **Top-P (Nucleus)**: Sample from tokens that make up the top P probability mass
//! - **Combined**: Combine multiple strategies (e.g., temperature + top-p)
//!
//! ## Example
//!
//! ```rust
//! use liquid_edge::sampling::{GreedySampler, TemperatureSampler, TopPSampler};
//! use liquid_edge::traits::{SamplingStrategy, GenerationOptions};
//!
//! let options = GenerationOptions::default();
//! let logits = vec![0.1, 0.2, 0.7]; // Example logits
//!
//! // Greedy sampling - always picks highest probability
//! let greedy = GreedySampler::new();
//! let token = greedy.sample(&logits, &options).unwrap();
//!
//! // Temperature sampling for controlled randomness
//! let temp_sampler = TemperatureSampler::new(0.8);
//! let token = temp_sampler.sample(&logits, &options).unwrap();
//!
//! // Top-p (nucleus) sampling
//! let top_p = TopPSampler::new(0.9);
//! let token = top_p.sample(&logits, &options).unwrap();
//! ```

use crate::{
    error::{EdgeError, EdgeResult},
    traits::{GenerationOptions, SamplingStrategy},
};
use rand::{thread_rng, Rng};
use std::cmp::Ordering;

/// Greedy sampling strategy - always selects the token with highest probability
#[derive(Debug, Clone)]
pub struct GreedySampler;

impl GreedySampler {
    /// Create a new greedy sampler
    pub fn new() -> Self {
        Self
    }
}

impl Default for GreedySampler {
    fn default() -> Self {
        Self::new()
    }
}

impl SamplingStrategy for GreedySampler {
    fn sample(&self, logits: &[f32], _options: &GenerationOptions) -> EdgeResult<u32> {
        if logits.is_empty() {
            return Err(EdgeError::invalid_input("Empty logits array"));
        }

        let max_idx = logits
            .iter()
            .enumerate()
            .max_by(|(_, a), (_, b)| a.partial_cmp(b).unwrap_or(Ordering::Equal))
            .map(|(idx, _)| idx)
            .ok_or_else(|| EdgeError::inference("Failed to find maximum logit"))?;

        Ok(max_idx as u32)
    }

    fn name(&self) -> &'static str {
        "greedy"
    }
}

/// Temperature sampling strategy - applies temperature scaling for controlled randomness
#[derive(Debug, Clone)]
pub struct TemperatureSampler {
    temperature: f32,
}

impl TemperatureSampler {
    /// Create a new temperature sampler
    ///
    /// # Arguments
    /// * `temperature` - Temperature value (0.0 = deterministic, higher = more random)
    pub fn new(temperature: f32) -> EdgeResult<Self> {
        if temperature < 0.0 {
            return Err(EdgeError::invalid_input("Temperature must be non-negative"));
        }
        Ok(Self { temperature })
    }

    /// Apply temperature scaling to logits
    fn apply_temperature(&self, logits: &[f32]) -> Vec<f32> {
        if self.temperature == 0.0 {
            // For temperature 0, return extremely peaked distribution
            let max_val = logits.iter().fold(f32::NEG_INFINITY, |a, &b| a.max(b));
            logits
                .iter()
                .map(|&x| {
                    if (x - max_val).abs() < f32::EPSILON {
                        1000.0
                    } else {
                        -1000.0
                    }
                })
                .collect()
        } else {
            logits.iter().map(|&x| x / self.temperature).collect()
        }
    }
}

impl SamplingStrategy for TemperatureSampler {
    fn sample(&self, logits: &[f32], options: &GenerationOptions) -> EdgeResult<u32> {
        if logits.is_empty() {
            return Err(EdgeError::invalid_input("Empty logits array"));
        }

        // Always use the sampler's own temperature setting
        let temperature = self.temperature;

        if temperature == 0.0 {
            // Fall back to greedy sampling
            return GreedySampler::new().sample(logits, options);
        }

        // Apply temperature scaling
        let scaled_logits: Vec<f32> = logits.iter().map(|&x| x / temperature).collect();

        // Convert to probabilities using softmax
        let probs = softmax(&scaled_logits)?;

        // Sample from the distribution
        sample_from_distribution(&probs)
    }

    fn name(&self) -> &'static str {
        "temperature"
    }
}

/// Top-K sampling strategy - samples from the K most likely tokens
#[derive(Debug, Clone)]
pub struct TopKSampler {
    k: usize,
}

impl TopKSampler {
    /// Create a new top-k sampler
    ///
    /// # Arguments
    /// * `k` - Number of top tokens to consider (0 = disabled)
    pub fn new(k: usize) -> Self {
        Self { k }
    }
}

impl SamplingStrategy for TopKSampler {
    fn sample(&self, logits: &[f32], options: &GenerationOptions) -> EdgeResult<u32> {
        if logits.is_empty() {
            return Err(EdgeError::invalid_input("Empty logits array"));
        }

        // Use k from options if available, otherwise use sampler's k
        let k = if options.top_k > 0 {
            options.top_k
        } else {
            self.k
        };

        if k == 0 || k >= logits.len() {
            // No top-k filtering, use temperature sampling
            return TemperatureSampler::new(options.temperature)?.sample(logits, options);
        }

        // Get top-k indices
        let mut indexed_logits: Vec<(usize, f32)> = logits
            .iter()
            .enumerate()
            .map(|(i, &logit)| (i, logit))
            .collect();

        indexed_logits.sort_by(|a, b| b.1.partial_cmp(&a.1).unwrap_or(Ordering::Equal));
        indexed_logits.truncate(k);

        // Create filtered logits array
        let mut filtered_logits = vec![f32::NEG_INFINITY; logits.len()];
        for (idx, logit) in indexed_logits {
            filtered_logits[idx] = logit;
        }

        // Apply temperature and sample
        TemperatureSampler::new(options.temperature)?.sample(&filtered_logits, options)
    }

    fn name(&self) -> &'static str {
        "top_k"
    }
}

/// Top-P (Nucleus) sampling strategy - samples from tokens that make up top P probability mass
#[derive(Debug, Clone)]
pub struct TopPSampler {
    p: f32,
}

impl TopPSampler {
    /// Create a new top-p sampler
    ///
    /// # Arguments
    /// * `p` - Probability mass threshold (0.0-1.0)
    pub fn new(p: f32) -> EdgeResult<Self> {
        if !(0.0..=1.0).contains(&p) {
            return Err(EdgeError::invalid_input(
                "Top-p value must be between 0.0 and 1.0",
            ));
        }
        Ok(Self { p })
    }
}

impl SamplingStrategy for TopPSampler {
    fn sample(&self, logits: &[f32], options: &GenerationOptions) -> EdgeResult<u32> {
        if logits.is_empty() {
            return Err(EdgeError::invalid_input("Empty logits array"));
        }

        // Use p from options if available, otherwise use sampler's p
        let p = if options.top_p > 0.0 && options.top_p <= 1.0 {
            options.top_p
        } else {
            self.p
        };

        if p >= 1.0 {
            // No top-p filtering, use temperature sampling
            return TemperatureSampler::new(options.temperature)?.sample(logits, options);
        }

        // Convert logits to probabilities
        let probs = softmax(logits)?;

        // Sort by probability (descending)
        let mut indexed_probs: Vec<(usize, f32)> = probs
            .iter()
            .enumerate()
            .map(|(i, &prob)| (i, prob))
            .collect();

        indexed_probs.sort_by(|a, b| b.1.partial_cmp(&a.1).unwrap_or(Ordering::Equal));

        // Find nucleus (tokens that make up top-p probability mass)
        let mut cumulative_prob = 0.0;
        let mut nucleus_size = 0;

        for (_, prob) in &indexed_probs {
            cumulative_prob += prob;
            nucleus_size += 1;
            if cumulative_prob >= p {
                break;
            }
        }

        // Ensure at least one token is included
        nucleus_size = nucleus_size.max(1);

        // Create filtered probability distribution
        let nucleus = &indexed_probs[..nucleus_size];
        let nucleus_sum: f32 = nucleus.iter().map(|(_, prob)| prob).sum();

        if nucleus_sum == 0.0 {
            return Err(EdgeError::inference("Zero probability mass in nucleus"));
        }

        // Sample from the nucleus
        let mut rng = thread_rng();
        let random_value = rng.gen::<f32>() * nucleus_sum;

        let mut cumulative = 0.0;
        for (token_id, prob) in nucleus {
            cumulative += prob;
            if cumulative >= random_value {
                return Ok(*token_id as u32);
            }
        }

        // Fallback to first token in nucleus
        Ok(nucleus[0].0 as u32)
    }

    fn name(&self) -> &'static str {
        "top_p"
    }
}

/// Combined sampling strategy that applies multiple strategies in sequence
#[derive(Debug)]
pub struct CombinedSampler {
    temperature: f32,
    top_p: f32,
    top_k: usize,
}

impl CombinedSampler {
    /// Create a new combined sampler
    pub fn new() -> Self {
        Self {
            temperature: 0.7,
            top_p: 0.9,
            top_k: 0,
        }
    }

    /// Create a common temperature + top-p combination
    pub fn temperature_top_p(temperature: f32, top_p: f32) -> EdgeResult<Self> {
        Ok(Self {
            temperature,
            top_p,
            top_k: 0,
        })
    }

    /// Create a temperature + top-k combination
    pub fn temperature_top_k(temperature: f32, top_k: usize) -> EdgeResult<Self> {
        Ok(Self {
            temperature,
            top_p: 1.0,
            top_k,
        })
    }
}

impl Default for CombinedSampler {
    fn default() -> Self {
        Self::new()
    }
}

impl SamplingStrategy for CombinedSampler {
    fn sample(&self, logits: &[f32], options: &GenerationOptions) -> EdgeResult<u32> {
        // Apply top-k filtering first if enabled
        let filtered_logits = if self.top_k > 0 && self.top_k < logits.len() {
            let top_k_sampler = TopKSampler::new(self.top_k);
            return top_k_sampler.sample(logits, options);
        } else {
            logits.to_vec()
        };

        // Apply top-p filtering if enabled
        if self.top_p < 1.0 {
            let top_p_sampler = TopPSampler::new(self.top_p)?;
            return top_p_sampler.sample(&filtered_logits, options);
        }

        // Fall back to temperature sampling
        let temperature_sampler = TemperatureSampler::new(self.temperature)?;
        temperature_sampler.sample(&filtered_logits, options)
    }

    fn name(&self) -> &'static str {
        "combined"
    }
}

/// Repetition penalty sampler that reduces probability of recently generated tokens
#[derive(Debug, Clone)]
pub struct RepetitionPenaltySampler {
    penalty: f32,
    recent_tokens: Vec<u32>,
    max_history: usize,
}

impl RepetitionPenaltySampler {
    /// Create a new repetition penalty sampler
    pub fn new(penalty: f32, max_history: usize) -> EdgeResult<Self> {
        if penalty <= 0.0 {
            return Err(EdgeError::invalid_input(
                "Repetition penalty must be positive",
            ));
        }

        Ok(Self {
            penalty,
            recent_tokens: Vec::new(),
            max_history,
        })
    }

    /// Add a token to the recent tokens history
    pub fn add_token(&mut self, token: u32) {
        self.recent_tokens.push(token);
        if self.recent_tokens.len() > self.max_history {
            self.recent_tokens.remove(0);
        }
    }

    /// Apply repetition penalty to logits
    fn apply_repetition_penalty(&self, logits: &[f32], penalty: f32) -> Vec<f32> {
        let mut penalized_logits = logits.to_vec();

        for &token in &self.recent_tokens {
            if let Some(logit) = penalized_logits.get_mut(token as usize) {
                if *logit > 0.0 {
                    *logit /= penalty;
                } else {
                    *logit *= penalty;
                }
            }
        }

        penalized_logits
    }
}

impl SamplingStrategy for RepetitionPenaltySampler {
    fn sample(&self, logits: &[f32], options: &GenerationOptions) -> EdgeResult<u32> {
        let penalty = if options.repetition_penalty > 0.0 {
            options.repetition_penalty
        } else {
            self.penalty
        };

        let penalized_logits = self.apply_repetition_penalty(logits, penalty);

        // Use temperature sampling as base sampler
        let temperature_sampler = TemperatureSampler::new(options.temperature)?;
        temperature_sampler.sample(&penalized_logits, options)
    }

    fn name(&self) -> &'static str {
        "repetition_penalty"
    }
}

/// Utility function to apply softmax to logits
fn softmax(logits: &[f32]) -> EdgeResult<Vec<f32>> {
    if logits.is_empty() {
        return Err(EdgeError::invalid_input("Empty logits for softmax"));
    }

    // Find maximum for numerical stability
    let max_logit = logits.iter().fold(f32::NEG_INFINITY, |a, &b| a.max(b));

    // Compute exponentials
    let exp_logits: Vec<f32> = logits.iter().map(|&x| (x - max_logit).exp()).collect();

    // Compute sum
    let sum: f32 = exp_logits.iter().sum();

    if sum == 0.0 || !sum.is_finite() {
        return Err(EdgeError::inference("Invalid softmax computation"));
    }

    // Normalize
    Ok(exp_logits.iter().map(|&x| x / sum).collect())
}

/// Sample from a probability distribution
fn sample_from_distribution(probs: &[f32]) -> EdgeResult<u32> {
    if probs.is_empty() {
        return Err(EdgeError::invalid_input("Empty probability distribution"));
    }

    let mut rng = thread_rng();
    let random_value = rng.gen::<f32>();

    let mut cumulative = 0.0;
    for (i, &prob) in probs.iter().enumerate() {
        cumulative += prob;
        if cumulative >= random_value {
            return Ok(i as u32);
        }
    }

    // Fallback to last token
    Ok((probs.len() - 1) as u32)
}

/// Factory for creating commonly used sampling strategies
pub struct SamplerFactory;

impl SamplerFactory {
    /// Create a greedy sampler
    pub fn greedy() -> GreedySampler {
        GreedySampler::new()
    }

    /// Create a temperature sampler
    pub fn temperature(temp: f32) -> EdgeResult<TemperatureSampler> {
        TemperatureSampler::new(temp)
    }

    /// Create a top-k sampler
    pub fn top_k(k: usize) -> TopKSampler {
        TopKSampler::new(k)
    }

    /// Create a top-p sampler
    pub fn top_p(p: f32) -> EdgeResult<TopPSampler> {
        TopPSampler::new(p)
    }

    /// Create a balanced sampler (temperature + top-p)
    pub fn balanced() -> EdgeResult<CombinedSampler> {
        CombinedSampler::temperature_top_p(0.7, 0.9)
    }

    /// Create a creative sampler (higher temperature + top-p)
    pub fn creative() -> EdgeResult<CombinedSampler> {
        CombinedSampler::temperature_top_p(0.9, 0.95)
    }

    /// Create a focused sampler (lower temperature + top-p)
    pub fn focused() -> EdgeResult<CombinedSampler> {
        CombinedSampler::temperature_top_p(0.2, 0.8)
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_greedy_sampler() {
        let sampler = GreedySampler::new();
        let logits = vec![0.1, 0.5, 0.3, 0.7, 0.2];
        let options = GenerationOptions::default();

        let token = sampler.sample(&logits, &options).unwrap();
        assert_eq!(token, 3); // Index of highest logit (0.7)
    }

    #[test]
    fn test_temperature_sampler() {
        let sampler = TemperatureSampler::new(0.0).unwrap(); // Should be deterministic
        let logits = vec![0.1, 0.5, 0.3, 0.7, 0.2];
        let options = GenerationOptions::default();

        let token = sampler.sample(&logits, &options).unwrap();
        assert_eq!(token, 3); // Should behave like greedy when temperature = 0
    }

    #[test]
    fn test_top_k_sampler() {
        let sampler = TopKSampler::new(2);
        let logits = vec![0.1, 0.5, 0.3, 0.7, 0.2];
        let options = GenerationOptions::deterministic();

        let token = sampler.sample(&logits, &options).unwrap();
        // Should only consider top-2 tokens (indices 3 and 1)
        assert!(token == 1 || token == 3);
    }

    #[test]
    fn test_top_p_sampler() {
        let sampler = TopPSampler::new(0.8).unwrap();
        let logits = vec![0.1, 0.5, 0.3, 0.7, 0.2];
        let options = GenerationOptions::deterministic();

        let token = sampler.sample(&logits, &options).unwrap();
        assert!(token < 5); // Should be a valid token index
    }

    #[test]
    fn test_softmax() {
        let logits = vec![1.0, 2.0, 3.0];
        let probs = softmax(&logits).unwrap();

        // Probabilities should sum to 1
        let sum: f32 = probs.iter().sum();
        assert!((sum - 1.0).abs() < 1e-6);

        // Higher logits should have higher probabilities
        assert!(probs[2] > probs[1]);
        assert!(probs[1] > probs[0]);
    }

    #[test]
    fn test_sampler_factory() {
        let greedy = SamplerFactory::greedy();
        assert_eq!(greedy.name(), "greedy");

        let temp = SamplerFactory::temperature(0.8).unwrap();
        assert_eq!(temp.name(), "temperature");

        let balanced = SamplerFactory::balanced().unwrap();
        assert_eq!(balanced.name(), "combined");
    }

    #[test]
    fn test_invalid_inputs() {
        // Test empty logits
        let sampler = GreedySampler::new();
        let result = sampler.sample(&[], &GenerationOptions::default());
        assert!(result.is_err());

        // Test invalid temperature
        let result = TemperatureSampler::new(-1.0);
        assert!(result.is_err());

        // Test invalid top-p
        let result = TopPSampler::new(1.5);
        assert!(result.is_err());
    }
}
