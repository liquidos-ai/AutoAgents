#[cfg(test)]
mod tests {
    use crate::actor::Topic;
    use crate::agent::{memory::SlidingWindowMemory, task::Task, AgentBuilder};
    use crate::environment::Environment;
    use crate::protocol::Event;
    use crate::runtime::{SingleThreadedRuntime, TypedRuntime};
    use crate::tests::agent::{MockAgentImpl, MockTool, TestAgentOutput};
    use crate::tool::{ToolCallError, ToolRuntime, ToolT};
    use autoagents_test_utils::llm::MockLLMProvider;
    use serde_json::Value;
    use std::sync::Arc;
    use tokio::time::{timeout, Duration};
    use tokio_stream::StreamExt;

    // Implement ToolT trait for MockTool
    impl ToolT for MockTool {
        fn name(&self) -> &'static str {
            "mock_tool"
        }

        fn description(&self) -> &'static str {
            "A mock tool for testing"
        }

        fn args_schema(&self) -> Value {
            serde_json::json!({
                "type": "object",
                "properties": {
                    "input": {"type": "string"}
                },
                "required": ["input"]
            })
        }
    }

    impl ToolRuntime for MockTool {
        fn execute(&self, args: Value) -> Result<Value, ToolCallError> {
            let input_str = args
                .get("input")
                .and_then(|v| v.as_str())
                .unwrap_or("no input");
            Ok(serde_json::json!({"output": format!("processed: {}", input_str)}))
        }
    }

    // The implementations for TestAgentOutput, MockAgentImpl, and AgentExecutor
    // are already provided in the base.rs test module, so we don't need to duplicate them

    #[tokio::test]
    async fn test_agent_creation_and_subscription() {
        let llm = Arc::new(MockLLMProvider);
        let runtime = SingleThreadedRuntime::new(None);
        let topic = Topic::<Task>::new("test_topic");
        let memory = Box::new(SlidingWindowMemory::new(10));

        let agent = MockAgentImpl::new("test_agent", "A test agent");
        let result = AgentBuilder::new(agent)
            .llm(llm)
            .runtime(runtime.clone())
            .subscribe(topic.clone())
            .memory(memory)
            .build()
            .await;

        assert!(result.is_ok());
        let agent_handle = result.unwrap();
        assert_eq!(agent_handle.agent.name(), "test_agent");
        assert_eq!(agent_handle.agent.description(), "A test agent");
    }

    #[tokio::test]
    async fn test_agent_task_execution() {
        let llm = Arc::new(MockLLMProvider);
        let runtime = SingleThreadedRuntime::new(None);
        let topic = Topic::<Task>::new("execution_test");
        let memory = Box::new(SlidingWindowMemory::new(10));

        let agent = MockAgentImpl::new("executor_agent", "An agent that executes tasks");
        let _agent_handle = AgentBuilder::new(agent)
            .llm(llm)
            .runtime(runtime.clone())
            .subscribe(topic.clone())
            .memory(memory)
            .build()
            .await
            .expect("Failed to build agent");

        // Set up environment
        let mut environment = Environment::new(None);
        environment
            .register_runtime(runtime.clone())
            .await
            .expect("Failed to register runtime");

        let receiver = environment
            .take_event_receiver(None)
            .await
            .expect("Failed to get event receiver");
        let mut event_stream = receiver;

        // Start environment in background
        let env_handle = tokio::spawn(async move {
            let _ = environment.run().await;
        });

        // Publish task
        let task = Task::new("Test task execution");
        runtime
            .publish(&topic, task)
            .await
            .expect("Failed to publish task");

        // Wait for task completion with timeout
        let result = timeout(Duration::from_secs(3), async {
            while let Some(event) = event_stream.next().await {
                match event {
                    Event::TaskComplete { result: value, .. } => {
                        let output: Result<TestAgentOutput, _> = serde_json::from_str(&value);
                        if let Ok(agent_output) = output {
                            return Some(agent_output);
                        }
                    }
                    _ => continue,
                }
            }
            None
        })
        .await;

        // Clean up
        env_handle.abort();

        assert!(result.is_ok());
        if let Ok(Some(agent_output)) = result {
            assert!(agent_output.result.contains("Test task execution"));
        }
    }

    #[tokio::test]
    async fn test_agent_with_streaming() {
        let llm = Arc::new(MockLLMProvider);
        let runtime = SingleThreadedRuntime::new(None);
        let topic = Topic::<Task>::new("streaming_test");
        let memory = Box::new(SlidingWindowMemory::new(10));

        let agent = MockAgentImpl::new("streaming_agent", "A streaming agent");
        let result = AgentBuilder::new(agent)
            .llm(llm)
            .runtime(runtime.clone())
            .subscribe(topic.clone())
            .memory(memory)
            .stream(true)
            .build()
            .await;

        assert!(result.is_ok());
        let agent_handle = result.unwrap();
        assert!(agent_handle.agent.stream());
    }

    #[tokio::test]
    async fn test_agent_failure_handling() {
        let llm = Arc::new(MockLLMProvider);
        let runtime = SingleThreadedRuntime::new(None);
        let topic = Topic::<Task>::new("failure_test");

        let agent = MockAgentImpl::new("failing_agent", "An agent that fails").with_failure(true);
        let _agent_handle = AgentBuilder::new(agent)
            .llm(llm)
            .runtime(runtime.clone())
            .subscribe(topic.clone())
            .build()
            .await
            .expect("Failed to build agent");

        // Set up environment
        let mut environment = Environment::new(None);
        environment
            .register_runtime(runtime.clone())
            .await
            .expect("Failed to register runtime");

        let receiver = environment
            .take_event_receiver(None)
            .await
            .expect("Failed to get event receiver");
        let mut event_stream = receiver;

        // Start environment in background
        let env_handle = tokio::spawn(async move {
            let _ = environment.run().await;
        });

        // Publish task
        let task = Task::new("This should fail");
        runtime
            .publish(&topic, task)
            .await
            .expect("Failed to publish task");

        // Wait for task result (should be an error)
        let result = timeout(Duration::from_secs(3), async {
            while let Some(event) = event_stream.next().await {
                match event {
                    Event::TaskError { error, .. } => {
                        return Some(error);
                    }
                    _ => continue,
                }
            }
            None
        })
        .await;

        // Clean up
        env_handle.abort();

        assert!(result.is_ok());
        if let Ok(Some(error_message)) = result {
            assert!(error_message.contains("Mock execution failed"));
        }
    }

    #[tokio::test]
    async fn test_agent_multiple_topics() {
        let llm = Arc::new(MockLLMProvider);
        let runtime = SingleThreadedRuntime::new(None);
        let topic1 = Topic::<Task>::new("topic1");
        let topic2 = Topic::<Task>::new("topic2");
        let memory = Box::new(SlidingWindowMemory::new(10));

        let agent = MockAgentImpl::new("multi_topic_agent", "Multi-topic agent");
        let result = AgentBuilder::new(agent)
            .llm(llm)
            .runtime(runtime.clone())
            .subscribe(topic1.clone())
            .subscribe(topic2.clone())
            .memory(memory)
            .build()
            .await;

        assert!(result.is_ok());
        let _agent_handle = result.unwrap();

        // Set up environment
        let mut environment = Environment::new(None);
        environment
            .register_runtime(runtime.clone())
            .await
            .expect("Failed to register runtime");

        // Publish tasks to both topics - should not panic
        let task1 = Task::new("Task for topic 1");
        let task2 = Task::new("Task for topic 2");

        assert!(runtime.publish(&topic1, task1).await.is_ok());
        assert!(runtime.publish(&topic2, task2).await.is_ok());
    }

    #[tokio::test]
    async fn test_agent_without_llm_fails() {
        let runtime = SingleThreadedRuntime::new(None);
        let topic = Topic::<Task>::new("no_llm_test");

        let agent = MockAgentImpl::new("no_llm_agent", "Agent without LLM");
        let result = AgentBuilder::new(agent)
            .runtime(runtime)
            .subscribe(topic)
            .build()
            .await;

        assert!(result.is_err());
        let error_string = result.unwrap_err().to_string();
        // The error should be a build failure
        assert!(error_string.contains("Build Failure"));
    }

    #[tokio::test]
    async fn test_agent_without_runtime_fails() {
        let llm = Arc::new(MockLLMProvider);
        let topic = Topic::<Task>::new("no_runtime_test");

        let agent = MockAgentImpl::new("no_runtime_agent", "Agent without runtime");
        let result = AgentBuilder::new(agent)
            .llm(llm)
            .subscribe(topic)
            .build()
            .await;

        assert!(result.is_err());
        let error_string = result.unwrap_err().to_string();
        // The error should be a build failure
        assert!(error_string.contains("Build Failure"));
    }

    #[tokio::test]
    async fn test_agent_memory_configuration() {
        let llm = Arc::new(MockLLMProvider);
        let runtime = SingleThreadedRuntime::new(None);
        let topic = Topic::<Task>::new("memory_test");
        let memory = Box::new(SlidingWindowMemory::new(5));

        let agent = MockAgentImpl::new("memory_agent", "Agent with memory");
        let agent_handle = AgentBuilder::new(agent)
            .llm(llm)
            .runtime(runtime)
            .subscribe(topic)
            .memory(memory)
            .build()
            .await
            .expect("Failed to build agent");

        // Verify memory is configured
        assert!(agent_handle.agent.memory().is_some());
    }

    #[tokio::test]
    async fn test_agent_tools_access() {
        let llm = Arc::new(MockLLMProvider);
        let runtime = SingleThreadedRuntime::new(None);
        let topic = Topic::<Task>::new("tools_test");

        let agent = MockAgentImpl::new("tools_agent", "Agent with tools");
        let agent_handle = AgentBuilder::new(agent)
            .llm(llm)
            .runtime(runtime)
            .subscribe(topic)
            .build()
            .await
            .expect("Failed to build agent");

        let tools = agent_handle.agent.tools();
        // MockAgentImpl returns empty tools vector by default
        assert_eq!(tools.len(), 0);
    }
}
