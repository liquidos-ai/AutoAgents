use async_trait::async_trait;
use once_cell::sync::Lazy;
use regex::Regex;

use crate::guard::{GuardContext, GuardDecision, GuardError, GuardedInput, InputGuard};

static EMAIL_RE: Lazy<Regex> =
    Lazy::new(|| Regex::new(r"(?i)\b[a-z0-9._%+\-]+@[a-z0-9.\-]+\.[a-z]{2,}\b").unwrap());
static PHONE_RE: Lazy<Regex> = Lazy::new(|| {
    Regex::new(r"\b(?:\+?1[-.\s]?)?(?:\(?\d{3}\)?[-.\s]?)\d{3}[-.\s]?\d{4}\b").unwrap()
});
static SSN_RE: Lazy<Regex> = Lazy::new(|| Regex::new(r"\b\d{3}-\d{2}-\d{4}\b").unwrap());
static CARD_RE: Lazy<Regex> = Lazy::new(|| Regex::new(r"\b(?:\d[ -]*?){13,19}\b").unwrap());

/// Input guard that redacts common PII patterns using regular expressions.
#[derive(Debug, Clone)]
pub struct RegexPiiRedactionGuard {
    pub email_replacement: String,
    pub phone_replacement: String,
    pub ssn_replacement: String,
    pub card_replacement: String,
}

impl Default for RegexPiiRedactionGuard {
    fn default() -> Self {
        Self {
            email_replacement: "[redacted:email]".to_string(),
            phone_replacement: "[redacted:phone]".to_string(),
            ssn_replacement: "[redacted:ssn]".to_string(),
            card_replacement: "[redacted:card]".to_string(),
        }
    }
}

impl RegexPiiRedactionGuard {
    fn redact_text(&self, text: &str) -> (String, bool) {
        let mut out = text.to_string();
        let mut changed = false;

        let replaced = EMAIL_RE.replace_all(&out, self.email_replacement.as_str());
        if replaced != out {
            changed = true;
            out = replaced.into_owned();
        }

        let replaced = PHONE_RE.replace_all(&out, self.phone_replacement.as_str());
        if replaced != out {
            changed = true;
            out = replaced.into_owned();
        }

        let replaced = SSN_RE.replace_all(&out, self.ssn_replacement.as_str());
        if replaced != out {
            changed = true;
            out = replaced.into_owned();
        }

        let replaced = CARD_RE.replace_all(&out, self.card_replacement.as_str());
        if replaced != out {
            changed = true;
            out = replaced.into_owned();
        }

        (out, changed)
    }
}

#[async_trait]
impl InputGuard for RegexPiiRedactionGuard {
    fn name(&self) -> &'static str {
        "regex-pii-redaction"
    }

    async fn inspect(
        &self,
        input: &mut GuardedInput,
        _context: &GuardContext,
    ) -> Result<GuardDecision, GuardError> {
        let mut changed = false;

        match input {
            GuardedInput::Chat(chat) => {
                for message in &mut chat.messages {
                    let (redacted, message_changed) = self.redact_text(&message.content);
                    if message_changed {
                        message.content = redacted;
                        changed = true;
                    }
                }
            }
            GuardedInput::Completion(completion) => {
                let (redacted, input_changed) = self.redact_text(&completion.request.prompt);
                if input_changed {
                    completion.request.prompt = redacted;
                    changed = true;
                }
            }
            GuardedInput::WebSearch(web) => {
                let (redacted, input_changed) = self.redact_text(&web.input);
                if input_changed {
                    web.input = redacted;
                    changed = true;
                }
            }
        }

        if changed {
            Ok(GuardDecision::Modify { violation: None })
        } else {
            Ok(GuardDecision::Pass)
        }
    }
}

#[cfg(test)]
mod tests {
    use autoagents_llm::chat::ChatMessage;

    use crate::guard::{GuardOperation, GuardedInput};

    use super::*;

    #[tokio::test]
    async fn redacts_common_pii_in_chat_messages() {
        let guard = RegexPiiRedactionGuard::default();
        let mut input = GuardedInput::Chat(crate::guard::ChatGuardInput {
            messages: vec![
                ChatMessage::user()
                    .content(
                        "Email me at test@example.com or call +1 (555) 123-4567. SSN 123-45-6789",
                    )
                    .build(),
            ],
            tools: None,
            json_schema: None,
        });

        let decision = guard
            .inspect(
                &mut input,
                &crate::guard::GuardContext::new(GuardOperation::Chat),
            )
            .await
            .unwrap();

        assert!(matches!(
            decision,
            GuardDecision::Modify { violation: None }
        ));

        let GuardedInput::Chat(chat) = input else {
            panic!("expected chat input");
        };
        let text = &chat.messages[0].content;
        assert!(text.contains("[redacted:email]"));
        assert!(text.contains("[redacted:phone]"));
        assert!(text.contains("[redacted:ssn]"));
    }
}
