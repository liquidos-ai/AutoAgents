//! Builder module for configuring and instantiating LLM providers.
//!
//! This module provides a flexible builder pattern for creating and configuring
//! LLM (Large Language Model) provider instances with various settings and options.

use crate::{
    LLMProvider,
    chat::{FunctionTool, ParameterProperty, ParametersSchema, ReasoningEffort, Tool, ToolChoice},
    error::LLMError,
};
use std::{collections::HashMap, marker::PhantomData};

/// A function type for validating LLM provider outputs.
/// Takes a response string and returns Ok(()) if valid, or Err with an error message if invalid.
pub type ValidatorFn = dyn Fn(&str) -> Result<(), String> + Send + Sync + 'static;

/// Supported LLM backend providers.
#[derive(Debug, Clone)]
pub enum LLMBackend {
    /// OpenAI API provider (GPT-3, GPT-4, etc.)
    OpenAI,
    /// Anthropic API provider (Claude models)
    Anthropic,
    /// Ollama local LLM provider for self-hosted models
    Ollama,
    /// DeepSeek API provider for their LLM models
    DeepSeek,
    /// X.AI (formerly Twitter) API provider
    XAI,
    /// Phind API provider for code-specialized models
    Phind,
    /// Google Gemini API provider
    Google,
    /// Groq API provider
    Groq,
    /// Azure OpenAI API provider
    AzureOpenAI,
    /// OpenRouter API provider for various models
    OpenRouter,
    /// MiniMax API provider
    MiniMax,
}

/// Implements string parsing for LLMBackend enum.
///
/// Converts a string representation of a backend provider name into the corresponding
/// LLMBackend variant. The parsing is case-insensitive.
///
/// # Arguments
///
/// * `s` - The string to parse
///
/// # Returns
///
/// * `Ok(LLMBackend)` - The corresponding backend variant if valid
/// * `Err(LLMError)` - An error if the string doesn't match any known backend
///
/// # Examples
///
/// ```
/// use std::str::FromStr;
/// use autoagents_llm::builder::LLMBackend;
///
/// let backend = LLMBackend::from_str("openai").unwrap();
/// assert!(matches!(backend, LLMBackend::OpenAI));
///
/// let err = LLMBackend::from_str("invalid").unwrap_err();
/// assert!(err.to_string().contains("Unknown LLM backend"));
/// ```
impl std::str::FromStr for LLMBackend {
    type Err = LLMError;

    fn from_str(s: &str) -> Result<Self, Self::Err> {
        match s.to_lowercase().as_str() {
            "openai" => Ok(LLMBackend::OpenAI),
            "anthropic" => Ok(LLMBackend::Anthropic),
            "ollama" => Ok(LLMBackend::Ollama),
            "deepseek" => Ok(LLMBackend::DeepSeek),
            "xai" => Ok(LLMBackend::XAI),
            "phind" => Ok(LLMBackend::Phind),
            "google" => Ok(LLMBackend::Google),
            "groq" => Ok(LLMBackend::Groq),
            "azure-openai" => Ok(LLMBackend::AzureOpenAI),
            "openrouter" => Ok(LLMBackend::OpenRouter),
            "minimax" => Ok(LLMBackend::MiniMax),
            _ => Err(LLMError::InvalidRequest(format!(
                "Unknown LLM backend: {s}"
            ))),
        }
    }
}

/// Builder for configuring and instantiating LLM providers.
///
/// Provides a fluent interface for setting various configuration options
/// like model selection, API keys, generation parameters, etc.
pub struct LLMBuilder<L: LLMProvider> {
    /// Selected backend provider
    pub(crate) backend: PhantomData<L>,
    /// API key for authentication with the provider
    pub(crate) api_key: Option<String>,
    /// Base URL for API requests (primarily for self-hosted instances)
    pub(crate) base_url: Option<String>,
    /// Model identifier/name to use
    pub model: Option<String>,
    /// Maximum tokens to generate in responses
    pub max_tokens: Option<u32>,
    /// Temperature parameter for controlling response randomness (0.0-1.0)
    pub temperature: Option<f32>,
    /// Request timeout duration in seconds
    pub(crate) timeout_seconds: Option<u64>,
    /// Top-p (nucleus) sampling parameter
    pub top_p: Option<f32>,
    /// Top-k sampling parameter
    pub(crate) top_k: Option<u32>,
    /// Format specification for embedding outputs
    pub(crate) embedding_encoding_format: Option<String>,
    /// Vector dimensions for embedding outputs
    pub(crate) embedding_dimensions: Option<u32>,
    /// Optional validation function for response content
    pub(crate) validator: Option<Box<ValidatorFn>>,
    /// Number of retry attempts when validation fails
    pub(crate) validator_attempts: usize,
    /// Tool choice
    pub(crate) tool_choice: Option<ToolChoice>,
    /// Enable parallel tool use
    pub(crate) enable_parallel_tool_use: Option<bool>,
    /// Enable reasoning
    pub(crate) reasoning: Option<bool>,
    /// Enable reasoning effort
    pub(crate) reasoning_effort: Option<String>,
    /// reasoning_budget_tokens
    pub(crate) reasoning_budget_tokens: Option<u32>,
    /// API Version
    pub(crate) api_version: Option<String>,
    /// Deployment Id
    pub(crate) deployment_id: Option<String>,
    /// Voice
    #[allow(dead_code)]
    pub(crate) voice: Option<String>,
    /// Whether to normalize response format
    #[allow(dead_code)]
    pub(crate) normalize_response: Option<bool>,
    /// ExtraBody
    pub(crate) extra_body: Option<serde_json::Value>,
    /// Keep Alive
    #[allow(dead_code)]
    pub(crate) keep_alive: Option<String>,
}

impl<L: LLMProvider> Default for LLMBuilder<L> {
    fn default() -> Self {
        Self {
            backend: PhantomData,
            api_key: None,
            base_url: None,
            model: None,
            max_tokens: None,
            temperature: None,
            timeout_seconds: None,
            top_p: None,
            top_k: None,
            embedding_encoding_format: None,
            embedding_dimensions: None,
            validator: None,
            validator_attempts: 0,
            tool_choice: None,
            enable_parallel_tool_use: None,
            reasoning: None,
            reasoning_effort: None,
            reasoning_budget_tokens: None,
            api_version: None,
            deployment_id: None,
            voice: None,
            normalize_response: Some(true), //Defaulting so it accumilates tool calls in streams, easy for agent handling
            extra_body: None,
            keep_alive: None,
        }
    }
}

impl<L: LLMProvider> LLMBuilder<L> {
    /// Creates a new empty builder instance with default values.
    pub fn new() -> Self {
        Self::default()
    }

    /// Sets the API key for authentication.
    pub fn api_key(mut self, key: impl Into<String>) -> Self {
        self.api_key = Some(key.into());
        self
    }

    /// Sets the base URL for API requests.
    pub fn base_url(mut self, url: impl Into<String>) -> Self {
        self.base_url = Some(url.into());
        self
    }

    /// Sets the model identifier to use.
    pub fn model(mut self, model: impl Into<String>) -> Self {
        self.model = Some(model.into());
        self
    }

    /// Sets the maximum number of tokens to generate.
    pub fn max_tokens(mut self, max_tokens: u32) -> Self {
        self.max_tokens = Some(max_tokens);
        self
    }

    /// Sets the request timeout in seconds.
    pub fn normalize_response(mut self, normalize_response: bool) -> Self {
        self.normalize_response = Some(normalize_response);
        self
    }

    /// Sets the temperature for controlling response randomness (0.0-1.0).
    pub fn temperature(mut self, temperature: f32) -> Self {
        self.temperature = Some(temperature);
        self
    }

    /// Sets the reasoning flag.
    pub fn reasoning_effort(mut self, reasoning_effort: ReasoningEffort) -> Self {
        self.reasoning_effort = Some(reasoning_effort.to_string());
        self
    }

    /// Sets the reasoning flag.
    pub fn reasoning(mut self, reasoning: bool) -> Self {
        self.reasoning = Some(reasoning);
        self
    }

    /// Sets the reasoning budget tokens.
    pub fn reasoning_budget_tokens(mut self, reasoning_budget_tokens: u32) -> Self {
        self.reasoning_budget_tokens = Some(reasoning_budget_tokens);
        self
    }

    /// Sets the request timeout in seconds.
    pub fn timeout_seconds(mut self, timeout_seconds: u64) -> Self {
        self.timeout_seconds = Some(timeout_seconds);
        self
    }

    /// Sets the top-p (nucleus) sampling parameter.
    pub fn top_p(mut self, top_p: f32) -> Self {
        self.top_p = Some(top_p);
        self
    }

    /// Sets the top-k sampling parameter.
    pub fn top_k(mut self, top_k: u32) -> Self {
        self.top_k = Some(top_k);
        self
    }

    /// Sets the encoding format for embeddings.
    pub fn embedding_encoding_format(
        mut self,
        embedding_encoding_format: impl Into<String>,
    ) -> Self {
        self.embedding_encoding_format = Some(embedding_encoding_format.into());
        self
    }

    /// Sets the dimensions for embeddings.
    pub fn embedding_dimensions(mut self, embedding_dimensions: u32) -> Self {
        self.embedding_dimensions = Some(embedding_dimensions);
        self
    }

    /// Sets a validation function to verify LLM responses.
    ///
    /// # Arguments
    ///
    /// * `f` - Function that takes a response string and returns Ok(()) if valid, or Err with error message if invalid
    pub fn validator<F>(mut self, f: F) -> Self
    where
        F: Fn(&str) -> Result<(), String> + Send + Sync + 'static,
    {
        self.validator = Some(Box::new(f));
        self
    }

    /// Sets the number of retry attempts for validation failures.
    ///
    /// # Arguments
    ///
    /// * `attempts` - Maximum number of times to retry generating a valid response
    pub fn validator_attempts(mut self, attempts: usize) -> Self {
        self.validator_attempts = attempts;
        self
    }

    /// Enable parallel tool use
    pub fn enable_parallel_tool_use(mut self, enable: bool) -> Self {
        self.enable_parallel_tool_use = Some(enable);
        self
    }

    /// Set tool choice.  Note that if the choice is given as Tool(name), and that
    /// tool isn't available, the builder will fail.
    pub fn tool_choice(mut self, choice: ToolChoice) -> Self {
        self.tool_choice = Some(choice);
        self
    }

    /// Explicitly disable the use of tools, even if they are provided.
    pub fn disable_tools(mut self) -> Self {
        self.tool_choice = Some(ToolChoice::None);
        self
    }

    /// Set the API version.
    pub fn api_version(mut self, api_version: impl Into<String>) -> Self {
        self.api_version = Some(api_version.into());
        self
    }

    /// Set the deployment id. Used in Azure OpenAI.
    pub fn deployment_id(mut self, deployment_id: impl Into<String>) -> Self {
        self.deployment_id = Some(deployment_id.into());
        self
    }

    pub fn extra_body(mut self, extra_body: impl serde::Serialize) -> Self {
        let value = serde_json::to_value(extra_body).ok();
        self.extra_body = value;
        self
    }
}

/// Builder for function parameters
#[allow(dead_code)]
pub struct ParamBuilder {
    name: String,
    property_type: String,
    description: String,
    items: Option<Box<ParameterProperty>>,
    enum_list: Option<Vec<String>>,
}

impl ParamBuilder {
    /// Creates a new parameter builder
    pub fn new(name: impl Into<String>) -> Self {
        Self {
            name: name.into(),
            property_type: "string".to_string(),
            description: String::default(),
            items: None,
            enum_list: None,
        }
    }

    /// Sets the parameter type
    pub fn type_of(mut self, type_str: impl Into<String>) -> Self {
        self.property_type = type_str.into();
        self
    }

    /// Sets the parameter description
    pub fn description(mut self, desc: impl Into<String>) -> Self {
        self.description = desc.into();
        self
    }

    /// Sets the array item type for array parameters
    pub fn items(mut self, item_property: ParameterProperty) -> Self {
        self.items = Some(Box::new(item_property));
        self
    }

    /// Sets the enum values for enum parameters
    pub fn enum_values(mut self, values: Vec<String>) -> Self {
        self.enum_list = Some(values);
        self
    }

    /// Builds the parameter property
    #[allow(dead_code)]
    fn build(self) -> (String, ParameterProperty) {
        (
            self.name,
            ParameterProperty {
                property_type: self.property_type,
                description: self.description,
                items: self.items,
                enum_list: self.enum_list,
            },
        )
    }
}

/// Builder for function tools
#[allow(dead_code)]
pub struct FunctionBuilder {
    name: String,
    description: String,
    parameters: Vec<ParamBuilder>,
    required: Vec<String>,
    raw_schema: Option<serde_json::Value>,
}

impl FunctionBuilder {
    /// Creates a new function builder
    pub fn new(name: impl Into<String>) -> Self {
        Self {
            name: name.into(),
            description: String::default(),
            parameters: Vec::default(),
            required: Vec::default(),
            raw_schema: None,
        }
    }

    /// Sets the function description
    pub fn description(mut self, desc: impl Into<String>) -> Self {
        self.description = desc.into();
        self
    }

    /// Adds a parameter to the function
    pub fn param(mut self, param: ParamBuilder) -> Self {
        self.parameters.push(param);
        self
    }

    /// Marks parameters as required
    pub fn required(mut self, param_names: Vec<String>) -> Self {
        self.required = param_names;
        self
    }

    /// Provides a full JSON Schema for the parameters.  Using this method
    /// bypasses the DSL and allows arbitrary complex schemas (nested arrays,
    /// objects, oneOf, etc.).
    pub fn json_schema(mut self, schema: serde_json::Value) -> Self {
        self.raw_schema = Some(schema);
        self
    }

    /// Builds the function tool
    #[allow(dead_code)]
    pub fn build(self) -> Tool {
        let parameters_value = if let Some(schema) = self.raw_schema {
            schema
        } else {
            let mut properties = HashMap::new();
            for param in self.parameters {
                let (name, prop) = param.build();
                properties.insert(name, prop);
            }

            serde_json::to_value(ParametersSchema {
                schema_type: "object".to_string(),
                properties,
                required: self.required,
            })
            .unwrap_or_else(|_| serde_json::Value::Object(serde_json::Map::new()))
        };

        Tool {
            tool_type: "function".to_string(),
            function: FunctionTool {
                name: self.name,
                description: self.description,
                parameters: parameters_value,
            },
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::chat::{ChatMessage, ChatResponse, StructuredOutputFormat};
    use crate::error::LLMError;
    use serde_json::json;
    use std::str::FromStr;

    #[test]
    fn test_llm_backend_from_str() {
        assert!(matches!(
            LLMBackend::from_str("openai").unwrap(),
            LLMBackend::OpenAI
        ));
        assert!(matches!(
            LLMBackend::from_str("OpenAI").unwrap(),
            LLMBackend::OpenAI
        ));
        assert!(matches!(
            LLMBackend::from_str("OPENAI").unwrap(),
            LLMBackend::OpenAI
        ));
        assert!(matches!(
            LLMBackend::from_str("anthropic").unwrap(),
            LLMBackend::Anthropic
        ));
        assert!(matches!(
            LLMBackend::from_str("ollama").unwrap(),
            LLMBackend::Ollama
        ));
        assert!(matches!(
            LLMBackend::from_str("deepseek").unwrap(),
            LLMBackend::DeepSeek
        ));
        assert!(matches!(
            LLMBackend::from_str("xai").unwrap(),
            LLMBackend::XAI
        ));
        assert!(matches!(
            LLMBackend::from_str("phind").unwrap(),
            LLMBackend::Phind
        ));
        assert!(matches!(
            LLMBackend::from_str("google").unwrap(),
            LLMBackend::Google
        ));
        assert!(matches!(
            LLMBackend::from_str("groq").unwrap(),
            LLMBackend::Groq
        ));
        assert!(matches!(
            LLMBackend::from_str("azure-openai").unwrap(),
            LLMBackend::AzureOpenAI
        ));

        let result = LLMBackend::from_str("invalid");
        assert!(result.is_err());
        assert!(
            result
                .unwrap_err()
                .to_string()
                .contains("Unknown LLM backend")
        );
    }

    #[test]
    fn test_param_builder_new() {
        let builder = ParamBuilder::new("test_param");
        assert_eq!(builder.name, "test_param");
        assert_eq!(builder.property_type, "string");
        assert_eq!(builder.description, "");
        assert!(builder.items.is_none());
        assert!(builder.enum_list.is_none());
    }

    #[test]
    fn test_param_builder_fluent_interface() {
        let builder = ParamBuilder::new("test_param")
            .type_of("integer")
            .description("A test parameter")
            .enum_values(vec!["option1".to_string(), "option2".to_string()]);

        assert_eq!(builder.name, "test_param");
        assert_eq!(builder.property_type, "integer");
        assert_eq!(builder.description, "A test parameter");
        assert_eq!(
            builder.enum_list,
            Some(vec!["option1".to_string(), "option2".to_string()])
        );
    }

    #[test]
    fn test_param_builder_with_items() {
        let item_property = ParameterProperty {
            property_type: "string".to_string(),
            description: "Array item".to_string(),
            items: None,
            enum_list: None,
        };

        let builder = ParamBuilder::new("array_param")
            .type_of("array")
            .description("An array parameter")
            .items(item_property);

        assert_eq!(builder.name, "array_param");
        assert_eq!(builder.property_type, "array");
        assert_eq!(builder.description, "An array parameter");
        assert!(builder.items.is_some());
    }

    #[test]
    fn test_param_builder_build() {
        let builder = ParamBuilder::new("test_param")
            .type_of("string")
            .description("A test parameter");

        let (name, property) = builder.build();
        assert_eq!(name, "test_param");
        assert_eq!(property.property_type, "string");
        assert_eq!(property.description, "A test parameter");
    }

    #[test]
    fn test_function_builder_new() {
        let builder = FunctionBuilder::new("test_function");
        assert_eq!(builder.name, "test_function");
        assert_eq!(builder.description, "");
        assert!(builder.parameters.is_empty());
        assert!(builder.required.is_empty());
        assert!(builder.raw_schema.is_none());
    }

    #[test]
    fn test_function_builder_fluent_interface() {
        let param = ParamBuilder::new("name")
            .type_of("string")
            .description("Name");
        let builder = FunctionBuilder::new("test_function")
            .description("A test function")
            .param(param)
            .required(vec!["name".to_string()]);

        assert_eq!(builder.name, "test_function");
        assert_eq!(builder.description, "A test function");
        assert_eq!(builder.parameters.len(), 1);
        assert_eq!(builder.required, vec!["name".to_string()]);
    }

    #[test]
    fn test_function_builder_with_json_schema() {
        let schema = json!({
            "type": "object",
            "properties": {
                "name": {"type": "string"},
                "age": {"type": "integer"}
            },
            "required": ["name", "age"]
        });

        let builder = FunctionBuilder::new("test_function").json_schema(schema.clone());
        assert_eq!(builder.raw_schema, Some(schema));
    }

    #[test]
    fn test_function_builder_build_with_parameters() {
        let param = ParamBuilder::new("name").type_of("string");
        let tool = FunctionBuilder::new("test_function")
            .description("A test function")
            .param(param)
            .required(vec!["name".to_string()])
            .build();

        assert_eq!(tool.tool_type, "function");
        assert_eq!(tool.function.name, "test_function");
        assert_eq!(tool.function.description, "A test function");
        assert!(tool.function.parameters.is_object());
    }

    #[test]
    fn test_function_builder_build_with_raw_schema() {
        let schema = json!({
            "type": "object",
            "properties": {
                "name": {"type": "string"}
            }
        });

        let tool = FunctionBuilder::new("test_function")
            .json_schema(schema.clone())
            .build();

        assert_eq!(tool.function.parameters, schema);
    }

    // Mock LLM provider for testing
    struct MockLLMProvider;

    #[async_trait::async_trait]
    impl crate::chat::ChatProvider for MockLLMProvider {
        async fn chat(
            &self,
            _messages: &[ChatMessage],
            _json_schema: Option<StructuredOutputFormat>,
        ) -> Result<Box<dyn ChatResponse>, LLMError> {
            unimplemented!()
        }

        async fn chat_with_tools(
            &self,
            _messages: &[ChatMessage],
            _tools: Option<&[Tool]>,
            _json_schema: Option<StructuredOutputFormat>,
        ) -> Result<Box<dyn ChatResponse>, LLMError> {
            unimplemented!()
        }
    }

    #[async_trait::async_trait]
    impl crate::completion::CompletionProvider for MockLLMProvider {
        async fn complete(
            &self,
            _req: &crate::completion::CompletionRequest,
            _json_schema: Option<crate::chat::StructuredOutputFormat>,
        ) -> Result<crate::completion::CompletionResponse, LLMError> {
            unimplemented!()
        }
    }

    #[async_trait::async_trait]
    impl crate::embedding::EmbeddingProvider for MockLLMProvider {
        async fn embed(&self, _text: Vec<String>) -> Result<Vec<Vec<f32>>, LLMError> {
            unimplemented!()
        }
    }

    #[async_trait::async_trait]
    impl crate::models::ModelsProvider for MockLLMProvider {}

    impl crate::LLMProvider for MockLLMProvider {}

    #[test]
    fn test_llm_builder_new() {
        let builder = LLMBuilder::<MockLLMProvider>::new();
        assert!(builder.api_key.is_none());
        assert!(builder.base_url.is_none());
        assert!(builder.model.is_none());
        assert!(builder.max_tokens.is_none());
        assert!(builder.temperature.is_none());
        assert!(builder.timeout_seconds.is_none());
        assert!(builder.tool_choice.is_none());
    }

    #[test]
    fn test_llm_builder_default() {
        let builder = LLMBuilder::<MockLLMProvider>::default();
        assert!(builder.api_key.is_none());
        assert!(builder.base_url.is_none());
        assert!(builder.model.is_none());
        assert_eq!(builder.validator_attempts, 0);
    }

    #[test]
    fn test_llm_builder_api_key() {
        let builder = LLMBuilder::<MockLLMProvider>::new().api_key("test_key");
        assert_eq!(builder.api_key, Some("test_key".to_string()));
    }

    #[test]
    fn test_llm_builder_base_url() {
        let builder = LLMBuilder::<MockLLMProvider>::new().base_url("https://api.example.com");
        assert_eq!(
            builder.base_url,
            Some("https://api.example.com".to_string())
        );
    }

    #[test]
    fn test_llm_builder_model() {
        let builder = LLMBuilder::<MockLLMProvider>::new().model("gpt-4");
        assert_eq!(builder.model, Some("gpt-4".to_string()));
    }

    #[test]
    fn test_llm_builder_max_tokens() {
        let builder = LLMBuilder::<MockLLMProvider>::new().max_tokens(1000);
        assert_eq!(builder.max_tokens, Some(1000));
    }

    #[test]
    fn test_llm_builder_temperature() {
        let builder = LLMBuilder::<MockLLMProvider>::new().temperature(0.7);
        assert_eq!(builder.temperature, Some(0.7));
    }

    #[test]
    fn test_llm_builder_reasoning_effort() {
        let builder = LLMBuilder::<MockLLMProvider>::new()
            .reasoning_effort(crate::chat::ReasoningEffort::High);
        assert_eq!(builder.reasoning_effort, Some("high".to_string()));
    }

    #[test]
    fn test_llm_builder_reasoning() {
        let builder = LLMBuilder::<MockLLMProvider>::new().reasoning(true);
        assert_eq!(builder.reasoning, Some(true));
    }

    #[test]
    fn test_llm_builder_reasoning_budget_tokens() {
        let builder = LLMBuilder::<MockLLMProvider>::new().reasoning_budget_tokens(5000);
        assert_eq!(builder.reasoning_budget_tokens, Some(5000));
    }

    #[test]
    fn test_llm_builder_timeout_seconds() {
        let builder = LLMBuilder::<MockLLMProvider>::new().timeout_seconds(30);
        assert_eq!(builder.timeout_seconds, Some(30));
    }

    #[test]
    fn test_llm_builder_top_p() {
        let builder = LLMBuilder::<MockLLMProvider>::new().top_p(0.9);
        assert_eq!(builder.top_p, Some(0.9));
    }

    #[test]
    fn test_llm_builder_top_k() {
        let builder = LLMBuilder::<MockLLMProvider>::new().top_k(50);
        assert_eq!(builder.top_k, Some(50));
    }

    #[test]
    fn test_llm_builder_embedding_encoding_format() {
        let builder = LLMBuilder::<MockLLMProvider>::new().embedding_encoding_format("float");
        assert_eq!(builder.embedding_encoding_format, Some("float".to_string()));
    }

    #[test]
    fn test_llm_builder_embedding_dimensions() {
        let builder = LLMBuilder::<MockLLMProvider>::new().embedding_dimensions(1536);
        assert_eq!(builder.embedding_dimensions, Some(1536));
    }

    #[test]
    fn test_llm_builder_validator() {
        let builder = LLMBuilder::<MockLLMProvider>::new().validator(|response| {
            if response.contains("error") {
                Err("Response contains error".to_string())
            } else {
                Ok(())
            }
        });
        assert!(builder.validator.is_some());
    }

    #[test]
    fn test_llm_builder_validator_attempts() {
        let builder = LLMBuilder::<MockLLMProvider>::new().validator_attempts(3);
        assert_eq!(builder.validator_attempts, 3);
    }

    #[test]
    fn test_llm_builder_enable_parallel_tool_use() {
        let builder = LLMBuilder::<MockLLMProvider>::new().enable_parallel_tool_use(true);
        assert_eq!(builder.enable_parallel_tool_use, Some(true));
    }

    #[test]
    fn test_llm_builder_tool_choice() {
        let builder = LLMBuilder::<MockLLMProvider>::new().tool_choice(ToolChoice::Auto);
        assert!(matches!(builder.tool_choice, Some(ToolChoice::Auto)));
    }

    #[test]
    fn test_llm_builder_disable_tools() {
        let builder = LLMBuilder::<MockLLMProvider>::new().disable_tools();
        assert!(matches!(builder.tool_choice, Some(ToolChoice::None)));
    }

    #[test]
    fn test_llm_builder_api_version() {
        let builder = LLMBuilder::<MockLLMProvider>::new().api_version("2023-05-15");
        assert_eq!(builder.api_version, Some("2023-05-15".to_string()));
    }

    #[test]
    fn test_llm_builder_deployment_id() {
        let builder = LLMBuilder::<MockLLMProvider>::new().deployment_id("my-deployment");
        assert_eq!(builder.deployment_id, Some("my-deployment".to_string()));
    }

    #[test]
    fn test_llm_builder_chaining() {
        let builder = LLMBuilder::<MockLLMProvider>::new()
            .api_key("test_key")
            .model("gpt-4")
            .max_tokens(2000)
            .temperature(0.8)
            .timeout_seconds(60)
            .top_p(0.95)
            .top_k(40)
            .embedding_encoding_format("float")
            .embedding_dimensions(1536)
            .validator_attempts(5)
            .reasoning(true)
            .reasoning_budget_tokens(10000)
            .api_version("2023-05-15")
            .deployment_id("test-deployment");

        assert_eq!(builder.api_key, Some("test_key".to_string()));
        assert_eq!(builder.model, Some("gpt-4".to_string()));
        assert_eq!(builder.max_tokens, Some(2000));
        assert_eq!(builder.temperature, Some(0.8));
        assert_eq!(builder.timeout_seconds, Some(60));
        assert_eq!(builder.top_p, Some(0.95));
        assert_eq!(builder.top_k, Some(40));
        assert_eq!(builder.embedding_encoding_format, Some("float".to_string()));
        assert_eq!(builder.embedding_dimensions, Some(1536));
        assert_eq!(builder.validator_attempts, 5);
        assert_eq!(builder.reasoning, Some(true));
        assert_eq!(builder.reasoning_budget_tokens, Some(10000));
        assert_eq!(builder.api_version, Some("2023-05-15".to_string()));
        assert_eq!(builder.deployment_id, Some("test-deployment".to_string()));
    }
}
