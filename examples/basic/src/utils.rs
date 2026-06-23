use autoagents::core::utils::BoxEventStream;
use autoagents::protocol::Event;
use serde_json::Value;
use tokio_stream::StreamExt;

pub fn handle_events(mut event_stream: BoxEventStream<Event>) {
    tokio::spawn(async move {
        while let Some(event) = event_stream.next().await {
            match event {
                Event::TaskStarted {
                    actor_id,
                    task_description,
                    ..
                } => {
                    println!(
                        "🎯 Task Started - Actor: {:?}, Task: {}",
                        actor_id, task_description
                    );
                }
                Event::TaskComplete { result, .. } => {
                    print_task_complete(&result);
                }
                Event::StreamChunk { chunk, .. } => {
                    println!("Chunk: {:?}", chunk);
                }
                _ => {
                    // Ignore other events for this example
                }
            }
        }
    });
}

/// Pretty-print a `TaskComplete` payload for console demos.
///
/// Direct and actor agents emit the **executor** output shape in `TaskComplete`
/// (for example `ReActAgentOutput` with `response`, `tool_calls`, `done`).
/// The typed value returned from `agent.run()` uses the agent's `Output` type
/// (for example `MathAgentOutput`) after `From` conversion — see the README.
fn print_task_complete(result: &str) {
    let Ok(value) = serde_json::from_str::<Value>(result) else {
        println!("✅ Task Completed: {result}");
        return;
    };

    // ReAct-style executor payloads nest the agent JSON inside `response`.
    if let Some(response) = value.get("response").and_then(Value::as_str) {
        println!("✅ Task Completed (executor event):");
        match serde_json::from_str::<Value>(response) {
            Ok(agent_json) => {
                println!(
                    "{}",
                    serde_json::to_string_pretty(&agent_json).unwrap_or_else(|_| response.into())
                );
            }
            Err(_) => println!("  {response}"),
        }
        if let Some(tool_calls) = value.get("tool_calls").and_then(Value::as_array)
            && !tool_calls.is_empty()
        {
            println!("  tool_calls: {} invocation(s)", tool_calls.len());
        }
        return;
    }

    println!("✅ Task Completed:");
    println!(
        "{}",
        serde_json::to_string_pretty(&value).unwrap_or_else(|_| result.to_string())
    );
}
