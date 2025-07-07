# LLM Crate Integration Tests

This directory contains integration tests for the `autoagents-llm` crate, which provides a unified interface for various Large Language Model (LLM) backends.

## Testing Framework

The tests are built using the following tools:

- **Rust's built-in test framework**: The core framework for defining and running tests.
- **Tokio**: An asynchronous runtime for Rust, necessary for testing the async LLM client.
- **Mockito**: A library for mocking HTTP requests and responses. This allows for testing the API clients without making actual network calls to LLM providers, ensuring tests are fast, deterministic, and don't require API keys.

## Running the Tests

To run all the tests for the `autoagents-llm` crate, including integration tests and doctests, use the following command from the root of the workspace:

```bash
cargo test -p autoagents-llm
```

## Existing Tests

### OpenAI Backend (`openai.rs`)

This test file covers the OpenAI backend client.

- `test_openai_chat_succeeds()`: This test verifies that the client can successfully parse a valid chat completion response from the mocked OpenAI API.
- `test_openai_chat_fails_on_api_error()`: This test ensures that the client correctly handles API errors (like a 500 Internal Server Error) and returns the appropriate `LLMError` variant.

## Adding New Tests

To add new integration tests, for example for a different backend:

1.  Create a new file in the `tests/` directory (e.g., `google.rs`).
2.  Follow the pattern in `openai.rs` to set up a `mockito` server and mock the API endpoints for that backend.
3.  Write test functions using `#[tokio::test]` to cover various scenarios, including successful responses and different error conditions.
4.  Use `assert!` macros to verify that the client behaves as expected.
