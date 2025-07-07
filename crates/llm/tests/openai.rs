use autoagents_llm::{
    backends::openai::OpenAI,
    builder::LLMBuilder,
    chat::{ChatMessage},
    error::LLMError,
};
use mockito;
use serde_json::json;

#[tokio::test]
async fn test_openai_chat_succeeds() {
    let mut server = mockito::Server::new_async().await;
    let url = server.url();

    let mock = server
        .mock("POST", "/chat/completions")
        .with_status(200)
        .with_header("content-type", "application/json")
        .with_body(
            json!({
                "choices": [{
                    "message": {
                        "role": "assistant",
                        "content": "Hello! How can I help you today?"
                    }
                }]
            })
            .to_string(),
        )
        .create_async()
        .await;

    let client = LLMBuilder::<OpenAI>::new()
        .api_key("test_key")
        .base_url(&url)
        .build()
        .unwrap();

    let messages = vec![ChatMessage::user().content("Hello").build()];
    let response = client.chat(&messages).await.unwrap();

    assert_eq!(
        response.text().unwrap(),
        "Hello! How can I help you today?"
    );
    mock.assert_async().await;
}

#[tokio::test]
async fn test_openai_chat_fails_on_api_error() {
    let mut server = mockito::Server::new_async().await;
    let url = server.url();

    let mock = server
        .mock("POST", "/chat/completions")
        .with_status(500)
        .with_header("content-type", "application/json")
        .with_body(
            json!({
                "error": {
                    "message": "Internal server error",
                    "type": "server_error",
                    "param": null,
                    "code": null
                }
            })
            .to_string(),
        )
        .create_async()
        .await;

    let client = LLMBuilder::<OpenAI>::new()
        .api_key("test_key")
        .base_url(&url)
        .build()
        .unwrap();

    let messages = vec![ChatMessage::user().content("Hello").build()];
    let result = client.chat(&messages).await;

    assert!(result.is_err());
    if let Err(LLMError::ResponseFormatError { raw_response, .. }) = result {
        assert!(raw_response.contains("Internal server error"));
    } else {
        panic!("Expected LLMError::ResponseFormatError, got {:?}", result);
    }

    mock.assert_async().await;
}
