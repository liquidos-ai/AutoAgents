use autoagents_llm::error::LLMError;
use mistralrs::{
    IsqType, PagedAttentionMetaBuilder, TextMessageRole, TextMessages, TextModelBuilder,
};

pub async fn run_model() -> Result<(), LLMError> {
    let model = TextModelBuilder::new("microsoft/Phi-3.5-mini-instruct".to_string())
        .with_isq(IsqType::Q8_0)
        .with_paged_attn(|| PagedAttentionMetaBuilder::default().build())
        .unwrap()
        .with_logging()
        .build()
        .await
        .unwrap();

    let messages = TextMessages::new()
        .add_message(
            TextMessageRole::System,
            "You are an AI agent with a specialty in programming.",
        )
        .add_message(
            TextMessageRole::User,
            "Hello! How are you? Please write generic binary search function in Rust.",
        );

    let response = model.send_chat_request(messages).await.unwrap();

    println!("{}", response.choices[0].message.content.as_ref().unwrap());
    Ok(())
}
