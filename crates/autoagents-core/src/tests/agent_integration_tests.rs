#[cfg(test)]
mod tests {
    use crate::actor::Topic;
    use crate::agent::{AgentBuilder, memory::SlidingWindowMemory, task::Task};
    use crate::environment::Environment;
    use crate::runtime::{SingleThreadedRuntime, TypedRuntime};
    use crate::tests::{DivergentStreamingAgent, MockAgentImpl, MockLLMProvider, TestAgentOutput};
    use autoagents_protocol::Event;
    use std::sync::Arc;
    use tokio::time::{Duration, timeout};
    use tokio_stream::StreamExt;

    async fn wait_for_task_complete(
        event_stream: &mut crate::utils::BoxEventStream<Event>,
    ) -> TestAgentOutput {
        timeout(Duration::from_secs(3), async {
            while let Some(event) = event_stream.next().await {
                if let Event::TaskComplete { result: value, .. } = event {
                    return serde_json::from_str::<TestAgentOutput>(&value)
                        .expect("TaskComplete result should deserialize");
                }
            }
            panic!("event stream ended without TaskComplete");
        })
        .await
        .expect("timed out waiting for TaskComplete")
    }

    async fn wait_for_task_error(event_stream: &mut crate::utils::BoxEventStream<Event>) -> String {
        timeout(Duration::from_secs(3), async {
            while let Some(event) = event_stream.next().await {
                if let Event::TaskError { error, .. } = event {
                    return error;
                }
            }
            panic!("event stream ended without TaskError");
        })
        .await
        .expect("timed out waiting for TaskError")
    }

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

        let mut environment = Environment::new(None);
        environment
            .register_runtime(runtime.clone())
            .await
            .expect("Failed to register runtime");

        let mut event_stream = environment
            .take_event_receiver(None)
            .await
            .expect("Failed to get event receiver");

        environment.run().expect("Failed to start environment");

        runtime
            .publish(&topic, Task::new("Test task execution"))
            .await
            .expect("Failed to publish task");

        let agent_output = wait_for_task_complete(&mut event_stream).await;
        assert!(agent_output.result.contains("Test task execution"));

        environment
            .shutdown()
            .await
            .expect("shutdown should succeed");
    }

    #[tokio::test]
    async fn test_agent_with_streaming() {
        let llm = Arc::new(MockLLMProvider);
        let runtime = SingleThreadedRuntime::new(None);
        let topic = Topic::<Task>::new("streaming_test");
        let memory = Box::new(SlidingWindowMemory::new(10));

        let agent = MockAgentImpl::new("streaming_agent", "A streaming agent");
        let _agent_handle = AgentBuilder::new(agent)
            .llm(llm)
            .runtime(runtime.clone())
            .subscribe(topic.clone())
            .memory(memory)
            .stream(true)
            .build()
            .await
            .expect("Failed to build streaming agent");

        let mut environment = Environment::new(None);
        environment
            .register_runtime(runtime.clone())
            .await
            .expect("Failed to register runtime");

        let mut event_stream = environment
            .take_event_receiver(None)
            .await
            .expect("Failed to get event receiver");

        environment.run().expect("Failed to start environment");

        runtime
            .publish(&topic, Task::new("Streaming task execution"))
            .await
            .expect("Failed to publish task");

        let agent_output = wait_for_task_complete(&mut event_stream).await;
        assert!(agent_output.result.contains("Streaming task execution"));

        environment
            .shutdown()
            .await
            .expect("shutdown should succeed");
    }

    #[tokio::test]
    async fn test_streaming_actor_emits_executor_task_complete_through_runtime() {
        let llm = Arc::new(MockLLMProvider);
        let runtime = SingleThreadedRuntime::new(None);
        let topic = Topic::<Task>::new("streaming_executor_payload");

        let _agent_handle = AgentBuilder::new(DivergentStreamingAgent)
            .llm(llm)
            .runtime(runtime.clone())
            .subscribe(topic.clone())
            .stream(true)
            .build()
            .await
            .expect("Failed to build streaming agent");

        let mut environment = Environment::new(None);
        environment
            .register_runtime(runtime.clone())
            .await
            .expect("Failed to register runtime");

        let mut event_stream = environment
            .take_event_receiver(None)
            .await
            .expect("Failed to get event receiver");

        environment.run().expect("Failed to start environment");

        runtime
            .publish(&topic, Task::new("runtime payload"))
            .await
            .expect("Failed to publish task");

        let result = timeout(Duration::from_secs(3), async {
            while let Some(event) = event_stream.next().await {
                if let Event::TaskComplete { result, .. } = event {
                    let parsed: serde_json::Value =
                        serde_json::from_str(&result).expect("TaskComplete should be JSON");
                    return parsed;
                }
            }
            panic!("event stream ended without TaskComplete");
        })
        .await
        .expect("timed out waiting for TaskComplete");

        assert_eq!(result["executor_only"], 42);
        assert_eq!(result["response"], "runtime payload");

        environment
            .shutdown()
            .await
            .expect("shutdown should succeed");
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

        let mut environment = Environment::new(None);
        environment
            .register_runtime(runtime.clone())
            .await
            .expect("Failed to register runtime");

        let mut event_stream = environment
            .take_event_receiver(None)
            .await
            .expect("Failed to get event receiver");

        environment.run().expect("Failed to start environment");

        runtime
            .publish(&topic, Task::new("This should fail"))
            .await
            .expect("Failed to publish task");

        let error_message = wait_for_task_error(&mut event_stream).await;
        assert!(error_message.contains("Mock execution failed"));

        runtime
            .publish(&topic, Task::new("Second failing task"))
            .await
            .expect("Failed to publish second task");

        let second_error = wait_for_task_error(&mut event_stream).await;
        assert!(second_error.contains("Mock execution failed"));

        environment
            .shutdown()
            .await
            .expect("shutdown should succeed");
    }

    #[tokio::test]
    async fn test_streaming_pub_sub_actor_survives_stream_task_failure() {
        let llm = Arc::new(MockLLMProvider);
        let runtime = SingleThreadedRuntime::new(None);
        let topic = Topic::<Task>::new("streaming_failure_test");

        let agent = MockAgentImpl::new("failing_stream_agent", "Streaming agent that fails")
            .with_failure(true);
        let _agent_handle = AgentBuilder::new(agent)
            .llm(llm)
            .runtime(runtime.clone())
            .subscribe(topic.clone())
            .stream(true)
            .build()
            .await
            .expect("Failed to build streaming agent");

        let mut environment = Environment::new(None);
        environment
            .register_runtime(runtime.clone())
            .await
            .expect("Failed to register runtime");

        let mut event_stream = environment
            .take_event_receiver(None)
            .await
            .expect("Failed to get event receiver");

        environment.run().expect("Failed to start environment");

        runtime
            .publish(&topic, Task::new("First streaming failure"))
            .await
            .expect("Failed to publish first task");

        let first_error = wait_for_task_error(&mut event_stream).await;
        assert!(first_error.contains("Mock execution failed"));

        runtime
            .publish(&topic, Task::new("Second streaming failure"))
            .await
            .expect("Failed to publish second task");

        let second_error = wait_for_task_error(&mut event_stream).await;
        assert!(second_error.contains("Mock execution failed"));

        environment
            .shutdown()
            .await
            .expect("shutdown should succeed");
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

        let mut environment = Environment::new(None);
        environment
            .register_runtime(runtime.clone())
            .await
            .expect("Failed to register runtime");

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
        assert_eq!(tools.len(), 0);
    }
}
