use std::marker::PhantomData;

/// Generic builder for TTS providers
///
/// This builder provides a consistent interface for configuring TTS providers.
pub struct TTSBuilder<T> {
    _phantom: PhantomData<T>,
}

impl<T> TTSBuilder<T> {
    /// Create a new TTS builder
    pub fn new() -> Self {
        Self {
            _phantom: PhantomData,
        }
    }
}

impl<T> Default for TTSBuilder<T> {
    fn default() -> Self {
        Self::new()
    }
}
