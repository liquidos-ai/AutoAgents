pub trait Tokenizer: Send + Sync + Clone {
    /// Load the tokenizer from the provided path.
    fn new(tokenizer_path: &str) -> Result<Self, String>
    where
        Self: Sized;

    /// Encode a string into a list of token identifiers.
    fn encode(&self, text: &str, bos: bool, eos: bool) -> Vec<u32>;

    /// Decode a list of token identifiers into a string.
    fn decode(&self, tokens: &[u32]) -> String;

    /// Beginning of sentence token.
    fn bos(&self) -> String {
        self.decode(&[self.bos_id()])
    }

    /// Beginning of sentence token identifier.
    fn bos_id(&self) -> u32;

    /// End of sentence token.
    fn eos(&self) -> String {
        self.decode(&[self.eos_id()])
    }

    /// End of sentence token identifier.
    fn eos_id(&self) -> u32;

    /// Stop token identifiers.
    fn stop_ids(&self) -> Vec<u32>;

    /// Number of tokens needed as context for incremental streaming decoding.
    /// Default is 0 (no context/buffering needed).
    fn streaming_context_size(&self) -> usize {
        0
    }
}
