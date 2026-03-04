# Guardrails Example

OpenAI-based guardrails example using `LLMLayer` + `PipelineBuilder`.

## Run

```bash
OPENAI_API_KEY=... cargo run -p guardrails-example
```

## Scenarios

- `Scenario 1`: Regex PII redaction preview (`RegexPiiRedactionGuard`) on local input payload.
- `Scenario 2`: Early input blocking for prompt injection (`PromptInjectionGuard`, `Block`).
- `Scenario 3`: Custom output sanitization via `output_sanitizer(...)` (`Sanitize`).
