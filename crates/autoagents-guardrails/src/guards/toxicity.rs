use async_trait::async_trait;

use crate::{
    guard::{GuardContext, GuardDecision, GuardError, GuardViolation, GuardedOutput, OutputGuard},
    policy::{GuardCategory, GuardSeverity},
};

/// Heuristic toxicity detector for output content.
#[derive(Debug, Clone)]
pub struct ToxicityGuard {
    terms: Vec<&'static str>,
}

impl Default for ToxicityGuard {
    fn default() -> Self {
        Self {
            terms: vec![
                "kill yourself",
                "i will kill you",
                "racial slur",
                "you are worthless",
                "hate you",
            ],
        }
    }
}

impl ToxicityGuard {
    pub fn new(terms: Vec<&'static str>) -> Self {
        Self { terms }
    }
}

#[async_trait]
impl OutputGuard for ToxicityGuard {
    fn name(&self) -> &'static str {
        "toxicity"
    }

    async fn inspect(
        &self,
        output: &mut GuardedOutput,
        _context: &GuardContext,
    ) -> Result<GuardDecision, GuardError> {
        let text = match output {
            GuardedOutput::Chat(chat) => chat.text.clone().unwrap_or_default(),
            GuardedOutput::Completion(completion) => completion.text.clone(),
        };

        let normalized = text.to_lowercase();
        for term in &self.terms {
            if normalized.contains(term) {
                return Ok(GuardDecision::Reject(
                    GuardViolation::new(
                        "toxicity_detected",
                        GuardCategory::Toxicity,
                        GuardSeverity::High,
                        "toxic content detected",
                    )
                    .with_metadata(serde_json::json!({ "term": term })),
                ));
            }
        }

        Ok(GuardDecision::Pass)
    }
}

#[cfg(test)]
mod tests {
    use crate::guard::{GuardOperation, GuardedOutput};

    use super::*;

    #[tokio::test]
    async fn rejects_toxic_output() {
        let guard = ToxicityGuard::default();
        let mut output = GuardedOutput::Completion(crate::guard::CompletionGuardOutput {
            text: "You are worthless and I hate you".to_string(),
        });

        let decision = guard
            .inspect(
                &mut output,
                &crate::guard::GuardContext::new(GuardOperation::Complete),
            )
            .await
            .unwrap();

        assert!(matches!(decision, GuardDecision::Reject(_)));
    }

    #[tokio::test]
    async fn allows_benign_output() {
        let guard = ToxicityGuard::default();
        let mut output = GuardedOutput::Completion(crate::guard::CompletionGuardOutput {
            text: "Here is a calm answer".to_string(),
        });

        let decision = guard
            .inspect(
                &mut output,
                &crate::guard::GuardContext::new(GuardOperation::Complete),
            )
            .await
            .unwrap();

        assert!(matches!(decision, GuardDecision::Pass));
    }
}
