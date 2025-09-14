use crate::model::llama::tokenizer::Tokenizer;

/// A streaming decoder wrapper for tokenizers that require
/// buffering to correctly decode tokens incrementally.
///
/// This is useful for tokenizers like SentencePiece that rely on
/// decoding context (e.g., byte-fallback fusion, leading spaces)
/// and cannot decode tokens one-by-one correctly.
#[derive(Debug, Clone)]
pub struct StreamingDecoder<T: Tokenizer> {
    tokenizer: T,
    buffer: Vec<u32>,
    context_size: usize,
    context_overlap: usize,
    emitted_tokens: usize,
}

impl<T: Tokenizer> StreamingDecoder<T> {
    pub fn new(tokenizer: T) -> Self {
        let context_size = tokenizer.streaming_context_size();
        Self {
            tokenizer,
            buffer: Vec::with_capacity(context_size),
            context_size,
            context_overlap: 2,
            emitted_tokens: 0,
        }
    }

    /// Feed new tokens incrementally, decode buffered tokens, return only the newly decoded text slice.
    pub fn push_tokens(&mut self, new_tokens: &[u32]) -> Option<String> {
        if self.context_size == 0 {
            return Some(self.tokenizer.decode(new_tokens));
        }

        self.buffer.extend_from_slice(new_tokens);

        if self.buffer.len() - self.emitted_tokens < self.context_size {
            return None;
        }

        let overlap_start = self.emitted_tokens.saturating_sub(self.context_overlap);
        let decoded = self.tokenizer.decode(&self.buffer[overlap_start..]);

        // No guaranteed token-to-text alignment, but decoding is cheap to figure out overlap
        let overlap_decoded = self
            .tokenizer
            .decode(&self.buffer[overlap_start..self.emitted_tokens]);

        match decoded.get(overlap_decoded.len()..) {
            Some(new_text) if !new_text.is_empty() => {
                self.emitted_tokens = self.buffer.len();
                Some(new_text.to_string())
            }
            _ => None,
        }
    }
}
