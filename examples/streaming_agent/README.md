# Streaming Agent Example

This example demonstrates the real-time streaming capabilities of the AutoAgents library. It shows how to receive a continuous stream of tokens from an agent as it processes a task. The agent can also use tools mid-stream without interrupting the user experience.

## Running the Example

To run this example, you need to have your OpenAI API key set as an environment variable.

1.  **Set the API Key:**
    You can set the environment variable directly in your shell or create a `.env` file in the root of the project with the following content:
    ```sh
    export OPENAI_API_KEY="your_openai_api_key_here"
    ```
    If you use a `.env` file, load it into your shell environment by running `source .env` from the project root.

2.  **Run the Agent:**
    Execute the following command from the root of the project:
    ```sh
    cargo run -p streaming_agent
    ```
