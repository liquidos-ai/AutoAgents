use autoagents_llm::{
    backends::google::Google,
    builder::LLMBuilder,
    chat::{ChatMessage, ChatProvider},
    error::LLMError,
};
use mockito;
use serde_json::json;

#[tokio::test]
async fn test_google_chat_succeeds() {
    let mut server = mockito::Server::new_async().await;
    let url = server.url();

    let mock = server
        .mock("POST", "/v1beta/models/gemini-1.5-flash:generateContent?key=test_key")
        .with_status(200)
        .with_header("content-type", "application/json")
        .with_body(
            json!({
                "candidates": [{
                    "content": {
                        "parts": [{
                            "text": "Hello! How can I help you today?"
                        }],
                        "role": "model"
                    }
                }]
            })
            .to_string(),
        )
        .create_async()
        .await;

    let client = LLMBuilder::<Google>::new()
        .api_key("test_key")
        .base_url(&url) // This will be ignored, but it's good practice to set it
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
async fn test_google_chat_fails_on_api_error() {
    let mut server = mockito::Server::new_async().await;
    let url = server.url();

    let mock = server
        .mock("POST", "/v1beta/models/gemini-1.5-flash:generateContent?key=test_key")
        .with_status(500)
        .with_header("content-type", "application/json")
        .with_body(
            json!({
                "error": {
                    "code": 500,
                    "message": "Internal server error",
                    "status": "INTERNAL"
                }
            })
            .to_string(),
        )
        .create_async()
        .await;

    let client = LLMBuilder::<Google>::new()
        .api_key("test_key")
        .base_url(&url) // This will be ignored, but it's good practice to set it
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
