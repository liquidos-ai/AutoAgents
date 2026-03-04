use std::sync::Arc;

use crate::guard::{GuardContext, GuardViolation, GuardedInput, GuardedOutput};

/// Function signature used to sanitize guarded input payloads.
pub type InputSanitizer =
    dyn Fn(&mut GuardedInput, &GuardViolation, &GuardContext) + Send + Sync + 'static;

/// Function signature used to sanitize guarded output payloads.
pub type OutputSanitizer =
    dyn Fn(&mut GuardedOutput, &GuardViolation, &GuardContext) + Send + Sync + 'static;

/// Shared input sanitizer handle.
pub type SharedInputSanitizer = Arc<InputSanitizer>;

/// Shared output sanitizer handle.
pub type SharedOutputSanitizer = Arc<OutputSanitizer>;

/// Default input sanitizer used by guardrails engine.
///
/// Behavior: redact all input text fields.
pub fn default_input_sanitizer() -> SharedInputSanitizer {
    Arc::new(redact_input_payload)
}

/// Default output sanitizer used by guardrails engine.
///
/// Behavior: redact all output text fields and clear optional chat metadata.
pub fn default_output_sanitizer() -> SharedOutputSanitizer {
    Arc::new(redact_output_payload)
}

/// Built-in input sanitizer: redact every input text field.
pub fn redact_input_payload(
    input: &mut GuardedInput,
    _violation: &GuardViolation,
    _context: &GuardContext,
) {
    input.redact_all();
}

/// Built-in output sanitizer: redact every output text field and clear optional
/// chat metadata.
pub fn redact_output_payload(
    output: &mut GuardedOutput,
    _violation: &GuardViolation,
    _context: &GuardContext,
) {
    output.redact_all();
}

/// Built-in output sanitizer: redact only text while preserving chat metadata.
pub fn redact_output_text_only_payload(
    output: &mut GuardedOutput,
    _violation: &GuardViolation,
    _context: &GuardContext,
) {
    output.redact_text_only();
}

/// Built-in no-op input sanitizer.
pub fn noop_input_payload(
    _input: &mut GuardedInput,
    _violation: &GuardViolation,
    _context: &GuardContext,
) {
}

/// Built-in no-op output sanitizer.
pub fn noop_output_payload(
    _output: &mut GuardedOutput,
    _violation: &GuardViolation,
    _context: &GuardContext,
) {
}

/// Convenience constructor for no-op input sanitization.
pub fn noop_input_sanitizer() -> SharedInputSanitizer {
    Arc::new(noop_input_payload)
}

/// Convenience constructor for no-op output sanitization.
pub fn noop_output_sanitizer() -> SharedOutputSanitizer {
    Arc::new(noop_output_payload)
}

/// Convenience constructor for text-only output redaction.
pub fn redact_output_text_only_sanitizer() -> SharedOutputSanitizer {
    Arc::new(redact_output_text_only_payload)
}
