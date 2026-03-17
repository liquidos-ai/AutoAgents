//! Sentence chunker for TTS input.
//!
//! Splits streaming text into natural sentence boundaries so each chunk can be
//! synthesized independently. Handles decimal numbers and enforces min/max
//! chunk sizes to avoid TTS quality issues.
//!
//! # Design
//!
//! Inspired by production chunkers in LiveKit Agents and Pipecat:
//! - Split at `[.!?]` followed by whitespace + uppercase (or end of input)
//! - Split at `\n\n` (paragraph break) unconditionally
//! - Do NOT split on decimal numbers (`$4.50`, `v2.0`)
//! - Force flush when buffer exceeds `max_chunk_chars`
//! - Hold result if it would be shorter than `min_chunk_chars`

/// Configuration for the sentence chunker.
#[derive(Debug, Clone)]
pub struct ChunkerConfig {
    /// Minimum characters before emitting a chunk (default: 20).
    /// Avoids sending very short fragments like "Yes." to TTS.
    pub min_chunk_chars: usize,
    /// Maximum characters before forcing a flush (default: 250).
    /// Prevents TTS timeouts on very long runs without punctuation.
    pub max_chunk_chars: usize,
}

impl Default for ChunkerConfig {
    fn default() -> Self {
        Self {
            min_chunk_chars: 20,
            max_chunk_chars: 250,
        }
    }
}

/// A sentence chunker that accumulates tokens and emits complete sentences.
///
/// Feed tokens via [`push_token`](SentenceChunker::push_token) and collect
/// emitted sentences. When the input stream ends, call
/// [`force_flush`](SentenceChunker::force_flush) to emit any remainder.
#[derive(Debug, Default)]
pub struct SentenceChunker {
    buffer: String,
    config: ChunkerConfig,
}

impl SentenceChunker {
    /// Create a new chunker with the default configuration.
    pub fn new() -> Self {
        Self {
            buffer: String::default(),
            config: ChunkerConfig::default(),
        }
    }

    /// Create a new chunker with a custom configuration.
    pub fn with_config(config: ChunkerConfig) -> Self {
        Self {
            buffer: String::default(),
            config,
        }
    }

    /// Push a token into the chunker.
    ///
    /// Returns a (possibly empty) list of complete sentences that were detected
    /// after appending `token`. A single token may produce multiple sentences
    /// if it spans several sentence boundaries.
    ///
    /// Returns an empty `Vec` if still accumulating.
    pub fn push_token(&mut self, token: &str) -> Vec<String> {
        self.buffer.push_str(token);
        self.emit_all()
    }

    /// Drain all complete sentences currently available in the buffer.
    fn emit_all(&mut self) -> Vec<String> {
        let mut results = Vec::new();
        while let Some(sentence) = self.try_emit() {
            results.push(sentence);
        }
        results
    }

    /// Force-flush the internal buffer, returning any remaining text.
    ///
    /// Call this when the input stream has ended (e.g., LLM finished generating).
    pub fn force_flush(&mut self) -> Option<String> {
        if self.buffer.trim().is_empty() {
            self.buffer.clear();
            return None;
        }
        let text = std::mem::take(&mut self.buffer);
        Some(text)
    }

    /// Try to emit a sentence from the buffer.
    fn try_emit(&mut self) -> Option<String> {
        // Force flush if buffer exceeds max length
        if self.buffer.len() > self.config.max_chunk_chars {
            return self.force_flush_at_best_point();
        }

        // Check for paragraph break (\n\n)
        if let Some(pos) = self.buffer.find("\n\n") {
            let split_pos = pos + 2; // include the \n\n in the first chunk
            let candidate = self.buffer[..split_pos].trim().to_string();
            if candidate.is_empty() {
                // Just whitespace before the break — discard it
                self.buffer = self.buffer[split_pos..].to_string();
                return None;
            }
            if candidate.len() >= self.config.min_chunk_chars {
                self.buffer = self.buffer[split_pos..].to_string();
                return Some(candidate);
            }
            // Too short — keep accumulating
            return None;
        }

        // Look for sentence-ending punctuation.
        // Skip boundaries that would produce chunks shorter than min_chunk_chars
        // by continuing the search from after the rejected boundary.
        let mut search_from: usize = 0;
        loop {
            match self.find_sentence_boundary_from(search_from) {
                Some((split_pos, _)) => {
                    let candidate = self.buffer[..split_pos].trim().to_string();
                    if candidate.len() >= self.config.min_chunk_chars {
                        self.buffer = self.buffer[split_pos..].to_string();
                        return Some(candidate);
                    }
                    // Candidate too short — continue searching past this boundary
                    search_from = split_pos;
                }
                None => return None,
            }
        }
    }

    /// Find the first sentence boundary in the buffer starting from `from_byte`.
    ///
    /// Returns `Some((byte_position_after_punctuation, char))` for the FIRST
    /// valid boundary found at or after `from_byte`, so we emit the earliest
    /// complete sentence and keep the remainder for subsequent calls.
    fn find_sentence_boundary_from(&self, from_byte: usize) -> Option<(usize, char)> {
        let bytes = self.buffer.as_bytes();
        let chars: Vec<(usize, char)> = self.buffer.char_indices().collect();

        for (idx, &(byte_pos, ch)) in chars.iter().enumerate() {
            if byte_pos < from_byte {
                continue;
            }
            if !matches!(ch, '.' | '!' | '?') {
                continue;
            }

            // The byte position right after this punctuation character
            let after_punct = byte_pos + ch.len_utf8();

            // Skip if this is a decimal number: digit.digit
            if ch == '.' && self.is_decimal_at(byte_pos, &chars, idx) {
                continue;
            }

            // Valid boundary if followed by whitespace + uppercase letter,
            // OR if punctuation is at the very end of the buffer.
            if after_punct >= bytes.len() {
                // Punctuation at end of buffer — valid boundary.
                return Some((after_punct, ch));
            }

            // Check what follows the punctuation
            let remainder = &self.buffer[after_punct..];
            if self.starts_with_whitespace_then_upper(remainder) {
                return Some((after_punct, ch));
            }
        }

        None
    }

    /// Check if the period at `byte_pos` is a decimal: digit.digit
    fn is_decimal_at(&self, _byte_pos: usize, chars: &[(usize, char)], char_idx: usize) -> bool {
        // Need a digit before and after the period
        if char_idx == 0 {
            return false;
        }
        let prev_char = chars[char_idx - 1].1;
        if !prev_char.is_ascii_digit() {
            return false;
        }
        // Check char after
        if char_idx + 1 < chars.len() {
            let next_char = chars[char_idx + 1].1;
            return next_char.is_ascii_digit();
        }
        // Period at end after a digit — might be a decimal waiting for more
        // tokens. Treat as decimal to be safe.
        false
    }

    /// Check if a string starts with whitespace followed by an uppercase letter.
    fn starts_with_whitespace_then_upper(&self, s: &str) -> bool {
        let mut chars = s.chars();
        match chars.next() {
            Some(c) if c.is_whitespace() => {}
            _ => return false,
        }
        // Skip additional whitespace
        for c in chars {
            if c.is_whitespace() {
                continue;
            }
            return c.is_uppercase();
        }
        false
    }

    /// Force flush at the best available point when buffer exceeds max length.
    /// Tries to split at the last sentence boundary; falls back to the full buffer.
    fn force_flush_at_best_point(&mut self) -> Option<String> {
        // Try to find a sentence boundary to split at
        if let Some((split_pos, _)) = self.find_sentence_boundary_from(0) {
            let candidate = self.buffer[..split_pos].trim().to_string();
            if !candidate.is_empty() {
                self.buffer = self.buffer[split_pos..].to_string();
                return Some(candidate);
            }
        }

        // No good boundary — flush the whole thing
        self.force_flush()
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    /// Helper: push all tokens and collect emitted sentences, then force flush.
    fn chunk_text(tokens: &[&str], config: ChunkerConfig) -> Vec<String> {
        let mut chunker = SentenceChunker::with_config(config);
        let mut results = Vec::new();
        for token in tokens {
            results.extend(chunker.push_token(token));
        }
        if let Some(remainder) = chunker.force_flush() {
            results.push(remainder);
        }
        results
    }

    /// Helper with default config but low min so tests are simpler.
    fn chunk_text_default(tokens: &[&str]) -> Vec<String> {
        chunk_text(
            tokens,
            ChunkerConfig {
                min_chunk_chars: 1,
                max_chunk_chars: 250,
            },
        )
    }

    #[test]
    fn test_decimal_no_split() {
        let tokens = vec!["Price is $4.50. Buy now!"];
        let result = chunk_text_default(&tokens);
        assert_eq!(result, vec!["Price is $4.50.", "Buy now!"]);
    }

    #[test]
    fn test_multiple_sentences() {
        let tokens = vec!["Hello! How are you? Fine."];
        let result = chunk_text_default(&tokens);
        assert_eq!(result, vec!["Hello!", "How are you?", "Fine."]);
    }

    #[test]
    fn test_force_flush_long_text() {
        let config = ChunkerConfig {
            min_chunk_chars: 1,
            max_chunk_chars: 250,
        };
        // 300 chars, no punctuation
        let long_text = "a".repeat(300);
        let tokens = vec![long_text.as_str()];
        let result = chunk_text(&tokens, config);
        assert_eq!(result.len(), 1);
        assert_eq!(result[0], long_text);
    }

    #[test]
    fn test_force_flush_remainder() {
        let mut chunker = SentenceChunker::new();
        chunker.push_token("Hello there");
        let flushed = chunker.force_flush();
        assert_eq!(flushed, Some("Hello there".to_string()));
    }

    #[test]
    fn test_force_flush_empty() {
        let mut chunker = SentenceChunker::new();
        assert_eq!(chunker.force_flush(), None);
    }

    #[test]
    fn test_force_flush_whitespace_only() {
        let mut chunker = SentenceChunker::new();
        chunker.push_token("   ");
        assert_eq!(chunker.force_flush(), None);
    }

    #[test]
    fn test_streaming_tokens() {
        // Simulate LLM streaming token by token
        let tokens = vec![
            "Hello", " ", "world", ".", " ", "How", " ", "are", " ", "you", "?",
        ];
        let result = chunk_text_default(&tokens);
        assert_eq!(result, vec!["Hello world.", "How are you?"]);
    }

    #[test]
    fn test_paragraph_break() {
        let tokens = vec!["First paragraph.\n\nSecond paragraph."];
        let result = chunk_text_default(&tokens);
        assert_eq!(result, vec!["First paragraph.", "Second paragraph."]);
    }

    #[test]
    fn test_min_chunk_chars_holds() {
        let config = ChunkerConfig {
            min_chunk_chars: 20,
            max_chunk_chars: 250,
        };
        // "Hi." is only 3 chars — should be held until more text arrives
        let mut chunker = SentenceChunker::with_config(config);
        assert!(chunker.push_token("Hi. ").is_empty());
        // Now push more text to complete a longer chunk
        let result = chunker.push_token("What is the meaning of life? I wonder.");
        // "Hi. What is the meaning of life?" should now be emitted (>= 20 chars)
        assert!(!result.is_empty());
        assert!(result[0].len() >= 20);
    }

    #[test]
    fn test_version_number_no_split() {
        let tokens = vec!["Use v2.0 for this. It is better."];
        let result = chunk_text_default(&tokens);
        assert_eq!(result, vec!["Use v2.0 for this.", "It is better."]);
    }

    #[test]
    fn test_exclamation_and_question() {
        let tokens = vec!["Wow! Really? Yes."];
        let result = chunk_text_default(&tokens);
        assert_eq!(result, vec!["Wow!", "Really?", "Yes."]);
    }

    #[test]
    fn test_max_chunk_with_boundary() {
        let config = ChunkerConfig {
            min_chunk_chars: 1,
            max_chunk_chars: 50,
        };
        // First sentence < 50, second pushes over
        let tokens = vec![
            "Short sentence here. And then a much longer sentence that pushes over the limit.",
        ];
        let result = chunk_text(&tokens, config);
        assert_eq!(result[0], "Short sentence here.");
        assert!(result.len() >= 2);
    }
}
