use crate::chat::{CompletionTokensDetails, ImageMime, PromptTokensDetails, StreamChunk, Usage};
use crate::{FunctionCall, ToolCall};
use autoagents_protocol as protocol;

impl From<protocol::ImageMime> for ImageMime {
    fn from(value: protocol::ImageMime) -> Self {
        match value {
            protocol::ImageMime::JPEG => ImageMime::JPEG,
            protocol::ImageMime::PNG => ImageMime::PNG,
            protocol::ImageMime::GIF => ImageMime::GIF,
            protocol::ImageMime::WEBP => ImageMime::WEBP,
            _ => ImageMime::PNG,
        }
    }
}

impl From<ImageMime> for protocol::ImageMime {
    fn from(value: ImageMime) -> Self {
        match value {
            ImageMime::JPEG => protocol::ImageMime::JPEG,
            ImageMime::PNG => protocol::ImageMime::PNG,
            ImageMime::GIF => protocol::ImageMime::GIF,
            ImageMime::WEBP => protocol::ImageMime::WEBP,
        }
    }
}

impl From<protocol::FunctionCall> for FunctionCall {
    fn from(value: protocol::FunctionCall) -> Self {
        Self {
            name: value.name,
            arguments: value.arguments,
        }
    }
}

impl From<FunctionCall> for protocol::FunctionCall {
    fn from(value: FunctionCall) -> Self {
        Self {
            name: value.name,
            arguments: value.arguments,
        }
    }
}

impl From<protocol::ToolCall> for ToolCall {
    fn from(value: protocol::ToolCall) -> Self {
        Self {
            id: value.id,
            call_type: value.call_type,
            function: value.function.into(),
        }
    }
}

impl From<ToolCall> for protocol::ToolCall {
    fn from(value: ToolCall) -> Self {
        Self {
            id: value.id,
            call_type: value.call_type,
            function: value.function.into(),
        }
    }
}

impl From<protocol::CompletionTokensDetails> for CompletionTokensDetails {
    fn from(value: protocol::CompletionTokensDetails) -> Self {
        Self {
            reasoning_tokens: value.reasoning_tokens,
            audio_tokens: value.audio_tokens,
        }
    }
}

impl From<CompletionTokensDetails> for protocol::CompletionTokensDetails {
    fn from(value: CompletionTokensDetails) -> Self {
        Self {
            reasoning_tokens: value.reasoning_tokens,
            audio_tokens: value.audio_tokens,
        }
    }
}

impl From<protocol::PromptTokensDetails> for PromptTokensDetails {
    fn from(value: protocol::PromptTokensDetails) -> Self {
        Self {
            cached_tokens: value.cached_tokens,
            audio_tokens: value.audio_tokens,
        }
    }
}

impl From<PromptTokensDetails> for protocol::PromptTokensDetails {
    fn from(value: PromptTokensDetails) -> Self {
        Self {
            cached_tokens: value.cached_tokens,
            audio_tokens: value.audio_tokens,
        }
    }
}

impl From<protocol::Usage> for Usage {
    fn from(value: protocol::Usage) -> Self {
        Self {
            prompt_tokens: value.prompt_tokens,
            completion_tokens: value.completion_tokens,
            total_tokens: value.total_tokens,
            completion_tokens_details: value
                .completion_tokens_details
                .map(CompletionTokensDetails::from),
            prompt_tokens_details: value.prompt_tokens_details.map(PromptTokensDetails::from),
        }
    }
}

impl From<Usage> for protocol::Usage {
    fn from(value: Usage) -> Self {
        Self {
            prompt_tokens: value.prompt_tokens,
            completion_tokens: value.completion_tokens,
            total_tokens: value.total_tokens,
            completion_tokens_details: value
                .completion_tokens_details
                .map(protocol::CompletionTokensDetails::from),
            prompt_tokens_details: value
                .prompt_tokens_details
                .map(protocol::PromptTokensDetails::from),
        }
    }
}

impl From<protocol::StreamChunk> for StreamChunk {
    fn from(value: protocol::StreamChunk) -> Self {
        match value {
            protocol::StreamChunk::Text(text) => StreamChunk::Text(text),
            protocol::StreamChunk::ToolUseStart { index, id, name } => {
                StreamChunk::ToolUseStart { index, id, name }
            }
            protocol::StreamChunk::ToolUseInputDelta {
                index,
                partial_json,
            } => StreamChunk::ToolUseInputDelta {
                index,
                partial_json,
            },
            protocol::StreamChunk::ToolUseComplete { index, tool_call } => {
                StreamChunk::ToolUseComplete {
                    index,
                    tool_call: tool_call.into(),
                }
            }
            protocol::StreamChunk::Done { stop_reason } => StreamChunk::Done { stop_reason },
            protocol::StreamChunk::Usage(usage) => StreamChunk::Usage(usage.into()),
        }
    }
}

impl From<StreamChunk> for protocol::StreamChunk {
    fn from(value: StreamChunk) -> Self {
        match value {
            StreamChunk::Text(text) => protocol::StreamChunk::Text(text),
            StreamChunk::ToolUseStart { index, id, name } => {
                protocol::StreamChunk::ToolUseStart { index, id, name }
            }
            StreamChunk::ToolUseInputDelta {
                index,
                partial_json,
            } => protocol::StreamChunk::ToolUseInputDelta {
                index,
                partial_json,
            },
            StreamChunk::ToolUseComplete { index, tool_call } => {
                protocol::StreamChunk::ToolUseComplete {
                    index,
                    tool_call: tool_call.into(),
                }
            }
            StreamChunk::Done { stop_reason } => protocol::StreamChunk::Done { stop_reason },
            StreamChunk::Usage(usage) => protocol::StreamChunk::Usage(usage.into()),
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn converts_stream_chunk_roundtrip() {
        let chunk = StreamChunk::ToolUseStart {
            index: 1,
            id: "tool_1".to_string(),
            name: "search".to_string(),
        };
        let protocol_chunk: protocol::StreamChunk = chunk.clone().into();
        let roundtrip: StreamChunk = protocol_chunk.into();
        assert_eq!(format!("{chunk:?}"), format!("{roundtrip:?}"));
    }

    #[test]
    fn converts_usage_roundtrip() {
        let usage = Usage {
            prompt_tokens: 1,
            completion_tokens: 2,
            total_tokens: 3,
            completion_tokens_details: Some(CompletionTokensDetails {
                reasoning_tokens: Some(4),
                audio_tokens: None,
            }),
            prompt_tokens_details: Some(PromptTokensDetails {
                cached_tokens: Some(5),
                audio_tokens: None,
            }),
        };
        let protocol_usage: protocol::Usage = usage.clone().into();
        let roundtrip: Usage = protocol_usage.into();
        assert_eq!(usage, roundtrip);
    }

    #[test]
    fn converts_image_mime_roundtrip() {
        for mime in [
            ImageMime::JPEG,
            ImageMime::PNG,
            ImageMime::GIF,
            ImageMime::WEBP,
        ] {
            let proto: protocol::ImageMime = mime.into();
            let back: ImageMime = proto.into();
            assert_eq!(mime, back);
        }
    }

    #[test]
    fn converts_function_call_roundtrip() {
        let fc = FunctionCall {
            name: "search".to_string(),
            arguments: r#"{"q":"test"}"#.to_string(),
        };
        let proto: protocol::FunctionCall = fc.clone().into();
        let back: FunctionCall = proto.into();
        assert_eq!(back.name, fc.name);
        assert_eq!(back.arguments, fc.arguments);
    }

    #[test]
    fn converts_tool_call_roundtrip() {
        let tc = ToolCall {
            id: "tc1".to_string(),
            call_type: "function".to_string(),
            function: FunctionCall {
                name: "tool".to_string(),
                arguments: "{}".to_string(),
            },
        };
        let proto: protocol::ToolCall = tc.clone().into();
        let back: ToolCall = proto.into();
        assert_eq!(back.id, tc.id);
        assert_eq!(back.call_type, tc.call_type);
        assert_eq!(back.function.name, tc.function.name);
    }

    #[test]
    fn converts_stream_chunk_text_roundtrip() {
        let chunk = StreamChunk::Text("hello".to_string());
        let proto: protocol::StreamChunk = chunk.into();
        let back: StreamChunk = proto.into();
        assert!(matches!(back, StreamChunk::Text(ref s) if s == "hello"));
    }

    #[test]
    fn converts_stream_chunk_tool_use_input_delta() {
        let chunk = StreamChunk::ToolUseInputDelta {
            index: 0,
            partial_json: r#"{"ke"#.to_string(),
        };
        let proto: protocol::StreamChunk = chunk.into();
        let back: StreamChunk = proto.into();
        assert!(matches!(
            back,
            StreamChunk::ToolUseInputDelta { index: 0, .. }
        ));
    }

    #[test]
    fn converts_stream_chunk_tool_use_complete() {
        let tc = ToolCall {
            id: "tc1".to_string(),
            call_type: "function".to_string(),
            function: FunctionCall {
                name: "tool".to_string(),
                arguments: "{}".to_string(),
            },
        };
        let chunk = StreamChunk::ToolUseComplete {
            index: 0,
            tool_call: tc,
        };
        let proto: protocol::StreamChunk = chunk.into();
        let back: StreamChunk = proto.into();
        assert!(matches!(
            back,
            StreamChunk::ToolUseComplete { index: 0, .. }
        ));
    }

    #[test]
    fn converts_stream_chunk_done() {
        let chunk = StreamChunk::Done {
            stop_reason: "end_turn".to_string(),
        };
        let proto: protocol::StreamChunk = chunk.into();
        let back: StreamChunk = proto.into();
        assert!(matches!(back, StreamChunk::Done { ref stop_reason } if stop_reason == "end_turn"));
    }

    #[test]
    fn converts_stream_chunk_usage() {
        let usage = Usage {
            prompt_tokens: 10,
            completion_tokens: 20,
            total_tokens: 30,
            completion_tokens_details: None,
            prompt_tokens_details: None,
        };
        let chunk = StreamChunk::Usage(usage);
        let proto: protocol::StreamChunk = chunk.into();
        let back: StreamChunk = proto.into();
        assert!(matches!(back, StreamChunk::Usage(_)));
    }

    #[test]
    fn converts_completion_tokens_details_roundtrip() {
        let details = CompletionTokensDetails {
            reasoning_tokens: Some(10),
            audio_tokens: Some(5),
        };
        let proto: protocol::CompletionTokensDetails = details.clone().into();
        let back: CompletionTokensDetails = proto.into();
        assert_eq!(details, back);
    }

    #[test]
    fn converts_prompt_tokens_details_roundtrip() {
        let details = PromptTokensDetails {
            cached_tokens: Some(100),
            audio_tokens: None,
        };
        let proto: protocol::PromptTokensDetails = details.clone().into();
        let back: PromptTokensDetails = proto.into();
        assert_eq!(details, back);
    }
}
