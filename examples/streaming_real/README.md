# Real Streaming Agent Example

This example demonstrates **real streaming functionality** with actual LLM providers in the AutoAgents framework. It follows the same format as `examples/basic/src/simple.rs` but with full streaming support.

## 🚀 Features

- **Real LLM Integration**: Uses actual OpenAI API with streaming
- **Tool Calling**: Demonstrates real tool execution with streaming events
- **Event Streaming**: Real-time text chunks, tool calls, and thinking events
- **Multiple Tools**: Calculator and weather tools with proper error handling
- **Colored Output**: Beautiful terminal output with event categorization

## 🔑 Prerequisites

### **Required: OpenAI API Key**

You **must** have an OpenAI API key to run this example:

1. **Get API Key**: Sign up at [https://platform.openai.com/api-keys](https://platform.openai.com/api-keys)
2. **Set Environment Variable**:
   ```bash
   export OPENAI_API_KEY='your-api-key-here'
   ```

## 🏃‍♂️ Running the Example

```bash
cd examples/streaming_real
cargo run
```

## 📊 Expected Output

When you run the example, you'll see real-time streaming output like this:

```
🚀 Real Streaming Agent Demo
This example uses real LLM providers with streaming support

✅ Using real OpenAI API with streaming
🔐 API Key: sk-12345...

📝 Sending test messages...

📋 Task Started - Agent: [agent-id], Task: What is 15 + 27? Please use the calculator tool.
🔄 Turn 1/5 started
🧠 Let me think about this calculation...
🔧 Tool Call Started: calculator with args: {"operation": "add", "a": 15, "b": 27}
🔧 Tool Call Requested: calculator with args: {"operation": "add", "a": 15, "b": 27}
✅ Tool Call Completed: calculator - Result: {"result": 42, "operation": "add", "a": 15, "b": 27}
✅ Tool Call Completed: calc_1
The result of 15 + 27 is 42. This is a simple addition operation where we combine two numbers.
✅ Turn 1 completed (final)
🎉 Task Complete - Value: 42, Explanation: The result of 15 + 27 is 42. This is a simple addition operation where we combine two numbers.

📋 Task Started - Agent: [agent-id], Task: What's the weather like in New York?
🔄 Turn 1/5 started
🧠 Let me check the weather for New York...
🔧 Tool Call Started: get_weather with args: {"city": "New York"}
🔧 Tool Call Requested: get_weather with args: {"city": "New York"}
✅ Tool Call Completed: get_weather - Result: {"city": "New York", "temperature": 22, "condition": "Partly Cloudy", "humidity": 65, "wind_speed": 12}
✅ Tool Call Completed: weather_1
The weather in New York is currently 22°C with partly cloudy conditions. The humidity is 65% and wind speed is 12 mph.
✅ Turn 1 completed (final)
🎉 Task Complete - Value: 22, Explanation: The weather in New York is currently 22°C with partly cloudy conditions. The humidity is 65% and wind speed is 12 mph.
```

## 🔧 Code Structure

### **1. Tools**

#### **Calculator Tool**
```rust
#[tool(
    name = "calculator",
    description = "Perform basic mathematical operations (add, subtract, multiply, divide)",
    input = CalculatorArgs,
)]
struct Calculator {}
```

#### **Weather Tool**
```rust
#[tool(
    name = "get_weather",
    description = "Get current weather information for a city",
    input = WeatherArgs,
)]
struct WeatherTool {}
```

### **2. Agent**
```rust
#[agent(
    name = "math_agent",
    description = "You are a Math agent that can perform calculations and explain the results.",
    tools = [Calculator, WeatherTool],
    output = MathAgentOutput
)]
pub struct MathAgent {}
```

### **3. Real LLM Provider**
```rust
let llm: Arc<dyn LLMProvider> = LLMBuilder::<OpenAI>::new()
    .api_key(api_key)
    .model("gpt-4o-mini")
    .stream(true)  // Enable streaming
    .temperature(0.7)
    .max_tokens(1000)
    .system("You are a helpful math assistant...")
    .build()?;
```

### **4. Streaming Event Handler**
```rust
fn handle_streaming_events(mut event_stream: ReceiverStream<Event>) {
    tokio::spawn(async move {
        while let Some(event) = event_stream.next().await {
            match event {
                Event::StreamTextChunk { chunk, is_final, .. } => {
                    print!("{}", chunk);
                    if is_final { println!(); }
                }
                Event::StreamToolCallStart { tool_call, .. } => {
                    println!("🔧 Tool Call Started: {}", tool_call.function.name);
                }
                // ... more event handling
            }
        }
    });
}
```

## 🎯 Streaming Events

The example demonstrates these streaming events:

### **Text Streaming**
- `StreamTextChunk`: Real-time text output from the LLM
- `StreamThinkingChunk`: Reasoning/thinking process

### **Tool Call Streaming**
- `StreamToolCallStart`: Tool call initiation
- `StreamToolCallChunk`: Partial tool call data
- `StreamToolCallEnd`: Tool call completion

### **Regular Events**
- `ToolCallRequested`: Tool execution request
- `ToolCallCompleted`: Tool execution result
- `TaskComplete`: Final task result

## 💰 Cost Considerations

- **OpenAI API Usage**: This example makes real API calls
- **Streaming vs Batch**: Streaming may have different pricing
- **Token Usage**: Based on input/output tokens
- **Tool Calls**: Additional tokens for function calling

## 🔒 Security

- **API Key**: Stored in environment variables (never hardcoded)
- **Error Handling**: Proper error management for API failures
- **Validation**: Input validation for tool arguments

## 🎉 Benefits

- **Real Streaming**: Actual LLM responses in real-time
- **Tool Integration**: Real tool calling with LLM reasoning
- **Production Ready**: Can be deployed with real API keys
- **Event-Driven**: Rich event system for monitoring and debugging
- **Error Handling**: Comprehensive error management

## 🔄 Comparison with Basic Example

| Feature | Basic Example | Streaming Real Example |
|---------|---------------|------------------------|
| LLM Provider | Mock | Real OpenAI API |
| Streaming | ❌ | ✅ |
| Tool Calls | ✅ | ✅ |
| Real-time Events | ❌ | ✅ |
| API Key Required | ❌ | ✅ |
| Production Ready | ❌ | ✅ |

This example provides a complete, production-ready streaming implementation that demonstrates the full power of the AutoAgents framework with real LLM providers! 