use async_trait::async_trait;

use crate::error::LLMError;

#[async_trait]
pub trait EmbeddingProvider {
    async fn embed(&self, input: Vec<String>) -> Result<Vec<Vec<f32>>, LLMError>;
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::error::LLMError;

    // Mock embedding provider for testing
    struct MockEmbeddingProvider {
        should_fail: bool,
        dimension: usize,
    }

    impl MockEmbeddingProvider {
        fn new(dimension: usize) -> Self {
            Self {
                should_fail: false,
                dimension,
            }
        }

        fn new_failing() -> Self {
            Self {
                should_fail: true,
                dimension: 0,
            }
        }
    }

    #[async_trait::async_trait]
    impl EmbeddingProvider for MockEmbeddingProvider {
        async fn embed(&self, input: Vec<String>) -> Result<Vec<Vec<f32>>, LLMError> {
            if self.should_fail {
                return Err(LLMError::ProviderError(
                    "Mock embedding failure".to_string(),
                ));
            }

            let mut embeddings = Vec::new();
            for (i, _text) in input.iter().enumerate() {
                let mut embedding = Vec::new();
                for j in 0..self.dimension {
                    embedding.push((i as f32 + j as f32) / 10.0);
                }
                embeddings.push(embedding);
            }
            Ok(embeddings)
        }
    }

    #[tokio::test]
    async fn test_embedding_provider_single_text() {
        let provider = MockEmbeddingProvider::new(3);
        let input = vec!["Hello world".to_string()];

        let result = provider.embed(input).await;
        assert!(result.is_ok());

        let embeddings = result.unwrap();
        assert_eq!(embeddings.len(), 1);
        assert_eq!(embeddings[0].len(), 3);
        assert_eq!(embeddings[0][0], 0.0);
        assert_eq!(embeddings[0][1], 0.1);
        assert_eq!(embeddings[0][2], 0.2);
    }

    #[tokio::test]
    async fn test_embedding_provider_multiple_texts() {
        let provider = MockEmbeddingProvider::new(2);
        let input = vec![
            "First text".to_string(),
            "Second text".to_string(),
            "Third text".to_string(),
        ];

        let result = provider.embed(input).await;
        assert!(result.is_ok());

        let embeddings = result.unwrap();
        assert_eq!(embeddings.len(), 3);

        // Check dimensions
        for embedding in &embeddings {
            assert_eq!(embedding.len(), 2);
        }

        // Check that each embedding is different
        assert_ne!(embeddings[0], embeddings[1]);
        assert_ne!(embeddings[1], embeddings[2]);

        // Check specific values
        assert_eq!(embeddings[0][0], 0.0);
        assert_eq!(embeddings[0][1], 0.1);
        assert_eq!(embeddings[1][0], 0.1);
        assert_eq!(embeddings[1][1], 0.2);
    }

    #[tokio::test]
    async fn test_embedding_provider_empty_input() {
        let provider = MockEmbeddingProvider::new(5);
        let input: Vec<String> = vec![];

        let result = provider.embed(input).await;
        assert!(result.is_ok());

        let embeddings = result.unwrap();
        assert!(embeddings.is_empty());
    }

    #[tokio::test]
    async fn test_embedding_provider_failure() {
        let provider = MockEmbeddingProvider::new_failing();
        let input = vec!["Test text".to_string()];

        let result = provider.embed(input).await;
        assert!(result.is_err());

        let error = result.unwrap_err();
        assert!(error.to_string().contains("Mock embedding failure"));
    }

    #[tokio::test]
    async fn test_embedding_provider_large_input() {
        let provider = MockEmbeddingProvider::new(10);
        let large_text = "x".repeat(10000);
        let input = vec![large_text];

        let result = provider.embed(input).await;
        assert!(result.is_ok());

        let embeddings = result.unwrap();
        assert_eq!(embeddings.len(), 1);
        assert_eq!(embeddings[0].len(), 10);
    }

    #[tokio::test]
    async fn test_embedding_provider_unicode_text() {
        let provider = MockEmbeddingProvider::new(3);
        let input = vec![
            "Hello 世界".to_string(),
            "🌍 Earth".to_string(),
            "测试 test".to_string(),
        ];

        let result = provider.embed(input).await;
        assert!(result.is_ok());

        let embeddings = result.unwrap();
        assert_eq!(embeddings.len(), 3);

        for embedding in embeddings {
            assert_eq!(embedding.len(), 3);
        }
    }

    #[tokio::test]
    async fn test_embedding_provider_special_characters() {
        let provider = MockEmbeddingProvider::new(2);
        let input = vec![
            "Special chars: !@#$%^&*()".to_string(),
            "Newlines\nand\ttabs".to_string(),
            "\"Quotes\" and 'apostrophes'".to_string(),
        ];

        let result = provider.embed(input).await;
        assert!(result.is_ok());

        let embeddings = result.unwrap();
        assert_eq!(embeddings.len(), 3);
    }

    #[tokio::test]
    async fn test_embedding_provider_very_large_dimension() {
        let provider = MockEmbeddingProvider::new(1000);
        let input = vec!["Test".to_string()];

        let result = provider.embed(input).await;
        assert!(result.is_ok());

        let embeddings = result.unwrap();
        assert_eq!(embeddings.len(), 1);
        assert_eq!(embeddings[0].len(), 1000);

        // Check that values are within expected range
        for (i, value) in embeddings[0].iter().enumerate() {
            assert_eq!(*value, i as f32 / 10.0);
        }
    }

    #[tokio::test]
    async fn test_embedding_provider_zero_dimension() {
        let provider = MockEmbeddingProvider::new(0);
        let input = vec!["Test".to_string()];

        let result = provider.embed(input).await;
        assert!(result.is_ok());

        let embeddings = result.unwrap();
        assert_eq!(embeddings.len(), 1);
        assert_eq!(embeddings[0].len(), 0);
    }

    #[tokio::test]
    async fn test_embedding_provider_mixed_content() {
        let provider = MockEmbeddingProvider::new(4);
        let input = vec![
            "".to_string(),                                           // Empty string
            "Single word".to_string(),                                // Short text
            "This is a longer sentence with more words.".to_string(), // Long text
            "123 456 789".to_string(),                                // Numbers
        ];

        let result = provider.embed(input).await;
        assert!(result.is_ok());

        let embeddings = result.unwrap();
        assert_eq!(embeddings.len(), 4);

        for embedding in embeddings {
            assert_eq!(embedding.len(), 4);
        }
    }

    #[tokio::test]
    async fn test_embedding_provider_consistency() {
        let provider = MockEmbeddingProvider::new(3);
        let input = vec!["Consistent test".to_string()];

        // Run embedding multiple times
        let result1 = provider.embed(input.clone()).await.unwrap();
        let result2 = provider.embed(input.clone()).await.unwrap();
        let result3 = provider.embed(input).await.unwrap();

        // Results should be identical
        assert_eq!(result1, result2);
        assert_eq!(result2, result3);
    }

    #[tokio::test]
    async fn test_embedding_provider_batch_processing() {
        let provider = MockEmbeddingProvider::new(2);
        let batch_size = 100;
        let input: Vec<String> = (0..batch_size)
            .map(|i| format!("Text number {i}"))
            .collect();

        let result = provider.embed(input).await;
        assert!(result.is_ok());

        let embeddings = result.unwrap();
        assert_eq!(embeddings.len(), batch_size);

        // Check that each embedding is different
        for i in 0..batch_size - 1 {
            assert_ne!(embeddings[i], embeddings[i + 1]);
        }
    }
}
