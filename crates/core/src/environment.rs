use crate::error::Error;
use crate::protocol::{Event, RuntimeID};
use crate::runtime::manager::RuntimeManager;
use crate::runtime::{Runtime, RuntimeError};
use std::path::PathBuf;
use std::sync::Arc;
use tokio_stream::wrappers::ReceiverStream;

#[derive(Debug, thiserror::Error)]
pub enum EnvironmentError {
    #[error("Runtime not found: {0}")]
    RuntimeNotFound(RuntimeID),

    #[error("Runtime error: {0}")]
    RuntimeError(#[from] RuntimeError),
}

#[derive(Clone)]
pub struct EnvironmentConfig {
    pub working_dir: PathBuf,
    pub channel_buffer: usize,
}

impl Default for EnvironmentConfig {
    fn default() -> Self {
        Self {
            working_dir: std::env::current_dir().unwrap_or_default(),
            channel_buffer: 100,
        }
    }
}

pub struct Environment {
    config: EnvironmentConfig,
    runtime_manager: Arc<RuntimeManager>,
    default_runtime: Option<RuntimeID>,
    handle: Option<tokio::task::JoinHandle<Result<(), RuntimeError>>>,
}

impl Environment {
    pub fn new(config: Option<EnvironmentConfig>) -> Self {
        let config = config.unwrap_or_default();
        let runtime_manager = Arc::new(RuntimeManager::new());

        Self {
            config,
            runtime_manager,
            default_runtime: None,
            handle: None,
        }
    }

    pub async fn register_runtime(&mut self, runtime: Arc<dyn Runtime>) -> Result<(), Error> {
        self.runtime_manager
            .register_runtime(runtime.clone())
            .await?;
        if self.default_runtime.is_none() {
            self.default_runtime = Some(runtime.id());
        }
        Ok(())
    }

    pub fn config(&self) -> &EnvironmentConfig {
        &self.config
    }

    pub async fn get_runtime(&self, runtime_id: Option<RuntimeID>) -> Option<Arc<dyn Runtime>> {
        let rid = runtime_id.unwrap_or(self.default_runtime?);
        self.runtime_manager.get_runtime(&rid).await
    }

    pub async fn get_runtime_or_default(
        &self,
        runtime_id: Option<RuntimeID>,
    ) -> Result<Arc<dyn Runtime>, Error> {
        let rid = runtime_id.unwrap_or(self.default_runtime.unwrap());
        self.get_runtime(Some(rid))
            .await
            .ok_or_else(|| EnvironmentError::RuntimeNotFound(rid).into())
    }

    pub fn run(&mut self) {
        let manager = self.runtime_manager.clone();
        // Spawn background task to run the runtimes.
        let handle = tokio::spawn(async move { manager.run().await });
        self.handle = Some(handle);
    }

    pub async fn take_event_receiver(
        &mut self,
        runtime_id: Option<RuntimeID>,
    ) -> Option<ReceiverStream<Event>> {
        if let Some(runtime) = self.get_runtime(runtime_id).await {
            runtime.take_event_receiver().await
        } else {
            None
        }
    }

    pub async fn shutdown(&mut self) {
        let _ = self.runtime_manager.stop().await;

        if let Some(handle) = self.handle.take() {
            let _ = handle.await;
        }
    }
}

// #[cfg(test)]
// mod tests {
//     use super::*;
//     use crate::agent::AgentRunResult;
//     use crate::agent::RunnableAgent;
//     use crate::memory::MemoryProvider;
//     use crate::protocol::Event;
//     use async_trait::async_trait;
//     use std::sync::Arc;
//     use tokio::sync::mpsc;
//     use uuid::Uuid;

//     // Mock agent for testing
//     #[derive(Debug)]
//     struct MockAgent {
//         id: Uuid,
//         name: String,
//         should_fail: bool,
//     }

//     impl MockAgent {
//         fn new(name: &str) -> Self {
//             Self {
//                 id: Uuid::new_v4(),
//                 name: name.to_string(),
//                 should_fail: false,
//             }
//         }

//         fn new_failing(name: &str) -> Self {
//             Self {
//                 id: Uuid::new_v4(),
//                 name: name.to_string(),
//                 should_fail: true,
//             }
//         }
//     }

//     #[async_trait]
//     impl RunnableAgent for MockAgent {
//         fn name(&self) -> &'static str {
//             Box::leak(self.name.clone().into_boxed_str())
//         }

//         fn description(&self) -> &'static str {
//             "Mock agent for testing"
//         }

//         fn id(&self) -> Uuid {
//             self.id
//         }

//         async fn run(
//             self: Arc<Self>,
//             task: crate::runtime::Task,
//             _tx_event: mpsc::Sender<Event>,
//         ) -> Result<AgentRunResult, crate::error::Error> {
//             if self.should_fail {
//                 Err(crate::error::Error::SessionError(
//                     crate::runtime::SessionError::EmptyTask,
//                 ))
//             } else {
//                 Ok(AgentRunResult::success(serde_json::json!({
//                     "response": format!("Processed: {}", task.prompt)
//                 })))
//             }
//         }

//         fn memory(&self) -> Option<Arc<tokio::sync::RwLock<Box<dyn MemoryProvider>>>> {
//             None
//         }
//     }

//     #[test]
//     fn test_environment_config_default() {
//         let config = EnvironmentConfig::default();
//         assert_eq!(config.channel_buffer, 100);
//         assert_eq!(
//             config.working_dir,
//             std::env::current_dir().unwrap_or_default()
//         );
//     }

//     #[test]
//     fn test_environment_config_custom() {
//         let config = EnvironmentConfig {
//             working_dir: std::path::PathBuf::from("/tmp"),
//             channel_buffer: 50,
//         };
//         assert_eq!(config.channel_buffer, 50);
//         assert_eq!(config.working_dir, std::path::PathBuf::from("/tmp"));
//     }

//     #[tokio::test]
//     async fn test_environment_new_default() {
//         let env = Environment::new(None).await;
//         assert_eq!(env.config().channel_buffer, 100);
//     }

//     #[tokio::test]
//     async fn test_environment_new_with_config() {
//         let config = EnvironmentConfig {
//             working_dir: std::path::PathBuf::from("/tmp"),
//             channel_buffer: 50,
//         };
//         let env = Environment::new(Some(config)).await;
//         assert_eq!(env.config().channel_buffer, 50);
//     }

//     #[tokio::test]
//     async fn test_environment_get_session() {
//         let env = Environment::new(None).await;

//         // Test getting default session
//         let session = env.get_session(None).await;
//         assert!(session.is_some());

//         // Test getting non-existent session
//         let non_existent_id = Uuid::new_v4();
//         let session = env.get_session(Some(non_existent_id)).await;
//         assert!(session.is_none());
//     }

//     #[tokio::test]
//     async fn test_environment_register_agent() {
//         let env = Environment::new(None).await;
//         let agent = Arc::new(MockAgent::new("test_agent"));

//         let result = env.register_agent(agent, None).await;
//         assert!(result.is_ok());
//     }

//     #[tokio::test]
//     async fn test_environment_register_agent_with_id() {
//         let env = Environment::new(None).await;
//         let agent = Arc::new(MockAgent::new("test_agent"));
//         let agent_id = Uuid::new_v4();

//         // let result = env.register_agent_with_id(agent_id, agent, None).await;
//         assert!(result.is_ok());
//     }

//     #[tokio::test]
//     async fn test_environment_add_task() {
//         let env = Environment::new(None).await;
//         let agent = Arc::new(MockAgent::new("test_agent"));
//         let agent_id = env.register_agent(agent, None).await.unwrap();

//         let result = env.add_task(agent_id, "Test task").await;
//         assert!(result.is_ok());
//     }

//     #[tokio::test]
//     async fn test_environment_run_task() {
//         let env = Environment::new(None).await;
//         let agent = Arc::new(MockAgent::new("test_agent"));
//         let agent_id = env.register_agent(agent, None).await.unwrap();

//         let sub_id = env.add_task(agent_id, "Test task").await.unwrap();
//         let result = env.run_task(agent_id, sub_id, None).await;
//         assert!(result.is_ok());

//         let result = result.unwrap();
//         assert!(result.success);
//         assert!(result.output.is_some());
//     }

//     #[tokio::test]
//     async fn test_environment_run() {
//         let env = Environment::new(None).await;
//         let agent = Arc::new(MockAgent::new("test_agent"));
//         let agent_id = env.register_agent(agent, None).await.unwrap();

//         env.add_task(agent_id, "Test task").await.unwrap();
//         // let result = env.run(agent_id, None).await;
//         assert!(result.is_ok());
//     }

//     #[tokio::test]
//     async fn test_environment_run_all() {
//         let env = Environment::new(None).await;
//         let agent = Arc::new(MockAgent::new("test_agent"));
//         let agent_id = env.register_agent(agent, None).await.unwrap();

//         // Add multiple tasks
//         for i in 1..=3 {
//             env.add_task(agent_id, format!("Task {}", i)).await.unwrap();
//         }

//         let results = env.run_all(agent_id, None).await;
//         assert!(results.is_ok());
//         assert_eq!(results.unwrap().len(), 3);
//     }

//     #[tokio::test]
//     async fn test_environment_event_sender() {
//         let env = Environment::new(None).await;
//         let sender = env.event_sender(None).await;
//         assert!(sender.is_ok());
//     }

//     #[tokio::test]
//     async fn test_environment_take_event_receiver() {
//         let mut env = Environment::new(None).await;
//         let receiver = env.take_event_receiver(None).await;
//         assert!(receiver.is_some());

//         // Second call should return None
//         let receiver2 = env.take_event_receiver(None).await;
//         assert!(receiver2.is_none());
//     }

//     #[tokio::test]
//     async fn test_environment_shutdown() {
//         let mut env = Environment::new(None).await;
//         env.shutdown().await;
//         // Should not panic
//     }

//     #[tokio::test]
//     async fn test_environment_with_failing_agent() {
//         let env = Environment::new(None).await;
//         let agent = Arc::new(MockAgent::new_failing("failing_agent"));
//         let agent_id = env.register_agent(agent, None).await.unwrap();

//         env.add_task(agent_id, "Test task").await.unwrap();
//         // let result = env.run(agent_id, None).await;
//         assert!(result.is_err());
//     }

//     #[tokio::test]
//     async fn test_environment_error_session_not_found() {
//         let env = Environment::new(None).await;
//         let non_existent_id = Uuid::new_v4();

//         let result = env.get_session_or_default(Some(non_existent_id)).await;
//         assert!(result.is_err());

//         assert!(result.is_err());
//         // Just test that it's an error, not the specific variant
//         assert!(result.is_err());
//     }

//     #[test]
//     fn test_environment_error_display() {
//         let session_id = Uuid::new_v4();
//         let error = EnvironmentError::SessionNotFound(session_id);
//         assert!(error.to_string().contains("Session not found"));
//         assert!(error.to_string().contains(&session_id.to_string()));
//     }
// }
