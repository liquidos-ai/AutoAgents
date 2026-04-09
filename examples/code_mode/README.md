# Code Mode Example

This example demonstrates a generic `CodeActAgent` with two custom tools defined inside the example crate:

- `AddNumbers`
- `MultiplyNumbers`

The agent itself is not math-specific. It can solve generic tasks in TypeScript with standard JavaScript APIs such as `Date`, `Math`, `JSON`, arrays, and strings, and it can call the custom math tools when arithmetic is useful. The final answer is plain text, and the program also prints the full CodeAct execution trace so you can inspect the generated script.

Because the CodeAct sandbox is isolated, the example uses built-in JavaScript APIs for generic tasks instead of external imports.

## Run

```sh
export OPENAI_API_KEY=your_openai_api_key_here
cargo run --package code-mode-example
```

The default prompt asks for the current UTC time, so the model should generate a small TypeScript snippet that uses `Date`.

You can also ask a math question to see the custom tools being called:

```sh
cargo run --package code-mode-example -- "Compute (18 + 24) * 3 using AddNumbers and MultiplyNumbers, log the intermediate sum, and answer in one sentence."
```

Or combine both styles in one prompt:

```sh
cargo run --package code-mode-example -- "What is the current UTC time, and what is (7 + 5) * 9 using the math tools? Answer in one short sentence."
```
