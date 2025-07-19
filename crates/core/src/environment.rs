use crate::agent::AgentRunResult;
use crate::agent::RunnableAgent;
use crate::error::Error;
use crate::protocol::{AgentID, Event, SessionId, SubmissionId};
use crate::session::{Session, SessionError, SessionManager, Task};

use std::path::PathBuf;
use std::sync::Arc;
use tokio::sync::{mpsc, Mutex};
use tokio::task::JoinHandle;
use tokio_stream::wrappers::ReceiverStream;
use uuid::Uuid;

#[derive(Debug, thiserror::Error)]
pub enum EnvironmentError {
    #[error("Session not found: {0}")]
    SessionNotFound(SessionId),

    #[error("Session error: {0}")]
    SesssionError(#[from] SessionError),
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
    session_manager: SessionManager,
    default_session: SessionId,
    event_handler: Option<JoinHandle<()>>,
}

impl Environment {
    pub async fn new(config: Option<EnvironmentConfig>) -> Self {
        let config = config.unwrap_or_default();
        let session_manager = SessionManager::new();
        let (default_session_id, _) = session_manager.create_session(config.channel_buffer).await;

        Self {
            config,
            session_manager,
            default_session: default_session_id,
            event_handler: None,
        }
    }

    pub fn config(&self) -> &EnvironmentConfig {
        &self.config
    }

    pub async fn get_session(&self, session_id: Option<SessionId>) -> Option<Arc<Mutex<Session>>> {
        let sid = session_id.unwrap_or(self.default_session);
        self.session_manager.get_session(&sid).await
    }

    pub async fn get_session_or_default(
        &self,
        session_id: Option<SessionId>,
    ) -> Result<Arc<Mutex<Session>>, Error> {
        let sid = session_id.unwrap_or(self.default_session);
        self.get_session(Some(sid))
            .await
            .ok_or_else(|| EnvironmentError::SessionNotFound(sid).into())
    }

    pub async fn get_session_mut(
        &self,
        session_id: Option<SessionId>,
    ) -> Option<Arc<Mutex<Session>>> {
        let sid = session_id.unwrap_or(self.default_session);
        self.session_manager.get_session_mut(&sid).await
    }

    pub async fn register_agent(
        &self,
        agent: Arc<dyn RunnableAgent>,
        session_id: Option<SessionId>,
    ) -> Result<AgentID, Error> {
        let agent_id = Uuid::new_v4();
        self.register_agent_with_id(agent_id, agent, session_id)
            .await?;
        Ok(agent_id)
    }

    pub async fn register_agent_with_id(
        &self,
        agent_id: AgentID,
        agent: Arc<dyn RunnableAgent>,
        session_id: Option<SessionId>,
    ) -> Result<(), Error> {
        let session_arc = self.get_session_or_default(session_id).await?;
        let session = session_arc.lock().await;
        session.register_agent_with_id(agent_id, agent).await;
        Ok(())
    }

    pub async fn add_task<T: Into<String>>(
        &self,
        agent_id: Uuid,
        task: T,
    ) -> Result<SubmissionId, Error> {
        let task = Task::new(task, Some(agent_id));
        let sub_id = task.submission_id;
        let session_arc = self.get_session_or_default(None).await?;
        let session = session_arc.lock().await;
        session.add_task(task).await?;
        Ok(sub_id)
    }

    pub async fn run_task(
        &self,
        agent_id: AgentID,
        sub_id: SubmissionId,
        session_id: Option<SessionId>,
    ) -> Result<AgentRunResult, Error> {
        let session_arc = self.get_session_or_default(session_id).await?;
        let session = session_arc.lock().await;
        let task = session.get_task(sub_id).await;
        session.run_task(task, agent_id).await
    }

    pub async fn run(
        &self,
        agent_id: AgentID,
        session_id: Option<SessionId>,
    ) -> Result<AgentRunResult, Error> {
        let session_arc = self.get_session_or_default(session_id).await?;
        let session = session_arc.lock().await;
        session.run(agent_id).await
    }

    pub async fn run_all(
        &self,
        agent_id: AgentID,
        session_id: Option<SessionId>,
    ) -> Result<Vec<AgentRunResult>, Error> {
        let session_arc = self.get_session_or_default(session_id).await?;
        let session = session_arc.lock().await;
        session.run_all(agent_id).await
    }

    pub async fn event_sender(
        &self,
        session_id: Option<SessionId>,
    ) -> Result<mpsc::Sender<Event>, Error> {
        let session_arc = self.get_session_or_default(session_id).await?;
        let session = session_arc.lock().await;
        Ok(session.event_sender().clone())
    }

    pub async fn take_event_receiver(
        &mut self,
        session_id: Option<SessionId>,
    ) -> Option<ReceiverStream<Event>> {
        if let Some(session_arc_mutex) = self.get_session_mut(session_id).await {
            let mut session = session_arc_mutex.lock().await;
            session.take_event_receiver()
        } else {
            None
        }
    }

    pub async fn shutdown(&mut self) {
        if let Some(handler) = self.event_handler.take() {
            handler.abort();
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::agent::AgentRunResult;
    use crate::agent::RunnableAgent;
    use crate::memory::MemoryProvider;
    use crate::protocol::Event;
    use async_trait::async_trait;
    use std::sync::Arc;
    use tokio::sync::mpsc;
    use uuid::Uuid;

    // Mock agent for testing
    #[derive(Debug)]
    struct MockAgent {
        id: Uuid,
        name: String,
        should_fail: bool,
    }

    impl MockAgent {
        fn new(name: &str) -> Self {
            Self {
                id: Uuid::new_v4(),
                name: name.to_string(),
                should_fail: false,
            }
        }

        fn new_failing(name: &str) -> Self {
            Self {
                id: Uuid::new_v4(),
                name: name.to_string(),
                should_fail: true,
            }
        }
    }

    #[async_trait]
    impl RunnableAgent for MockAgent {
        fn name(&self) -> &'static str {
            Box::leak(self.name.clone().into_boxed_str())
        }

        fn description(&self) -> &'static str {
            "Mock agent for testing"
        }

        fn id(&self) -> Uuid {
            self.id
        }

        async fn run(
            self: Arc<Self>,
            task: crate::session::Task,
            _tx_event: mpsc::Sender<Event>,
        ) -> Result<AgentRunResult, crate::error::Error> {
            if self.should_fail {
                Err(crate::error::Error::SessionError(
                    crate::session::SessionError::EmptyTask,
                ))
            } else {
                Ok(AgentRunResult::success(serde_json::json!({
                    "response": format!("Processed: {}", task.prompt)
                })))
            }
        }

        fn memory(&self) -> Option<Arc<tokio::sync::RwLock<Box<dyn MemoryProvider>>>> {
            None
        }
    }

    #[test]
    fn test_environment_config_default() {
        let config = EnvironmentConfig::default();
        assert_eq!(config.channel_buffer, 100);
        assert_eq!(
            config.working_dir,
            std::env::current_dir().unwrap_or_default()
        );
    }

    #[test]
    fn test_environment_config_custom() {
        let config = EnvironmentConfig {
            working_dir: std::path::PathBuf::from("/tmp"),
            channel_buffer: 50,
        };
        assert_eq!(config.channel_buffer, 50);
        assert_eq!(config.working_dir, std::path::PathBuf::from("/tmp"));
    }

    #[tokio::test]
    async fn test_environment_new_default() {
        let env = Environment::new(None).await;
        assert_eq!(env.config().channel_buffer, 100);
    }

    #[tokio::test]
    async fn test_environment_new_with_config() {
        let config = EnvironmentConfig {
            working_dir: std::path::PathBuf::from("/tmp"),
            channel_buffer: 50,
        };
        let env = Environment::new(Some(config)).await;
        assert_eq!(env.config().channel_buffer, 50);
    }

    #[tokio::test]
    async fn test_environment_get_session() {
        let env = Environment::new(None).await;

        // Test getting default session
        let session = env.get_session(None).await;
        assert!(session.is_some());

        // Test getting non-existent session
        let non_existent_id = Uuid::new_v4();
        let session = env.get_session(Some(non_existent_id)).await;
        assert!(session.is_none());
    }

    #[tokio::test]
    async fn test_environment_register_agent() {
        let env = Environment::new(None).await;
        let agent = Arc::new(MockAgent::new("test_agent"));

        let result = env.register_agent(agent, None).await;
        assert!(result.is_ok());
    }

    #[tokio::test]
    async fn test_environment_register_agent_with_id() {
        let env = Environment::new(None).await;
        let agent = Arc::new(MockAgent::new("test_agent"));
        let agent_id = Uuid::new_v4();

        let result = env.register_agent_with_id(agent_id, agent, None).await;
        assert!(result.is_ok());
    }

    #[tokio::test]
    async fn test_environment_add_task() {
        let env = Environment::new(None).await;
        let agent = Arc::new(MockAgent::new("test_agent"));
        let agent_id = env.register_agent(agent, None).await.unwrap();

        let result = env.add_task(agent_id, "Test task").await;
        assert!(result.is_ok());
    }

    #[tokio::test]
    async fn test_environment_run_task() {
        let env = Environment::new(None).await;
        let agent = Arc::new(MockAgent::new("test_agent"));
        let agent_id = env.register_agent(agent, None).await.unwrap();

        let sub_id = env.add_task(agent_id, "Test task").await.unwrap();
        let result = env.run_task(agent_id, sub_id, None).await;
        assert!(result.is_ok());

        let result = result.unwrap();
        assert!(result.success);
        assert!(result.output.is_some());
    }

    #[tokio::test]
    async fn test_environment_run() {
        let env = Environment::new(None).await;
        let agent = Arc::new(MockAgent::new("test_agent"));
        let agent_id = env.register_agent(agent, None).await.unwrap();

        env.add_task(agent_id, "Test task").await.unwrap();
        let result = env.run(agent_id, None).await;
        assert!(result.is_ok());
    }

    #[tokio::test]
    async fn test_environment_run_all() {
        let env = Environment::new(None).await;
        let agent = Arc::new(MockAgent::new("test_agent"));
        let agent_id = env.register_agent(agent, None).await.unwrap();

        // Add multiple tasks
        for i in 1..=3 {
            env.add_task(agent_id, format!("Task {}", i)).await.unwrap();
        }

        let results = env.run_all(agent_id, None).await;
        assert!(results.is_ok());
        assert_eq!(results.unwrap().len(), 3);
    }

    #[tokio::test]
    async fn test_environment_event_sender() {
        let env = Environment::new(None).await;
        let sender = env.event_sender(None).await;
        assert!(sender.is_ok());
    }

    #[tokio::test]
    async fn test_environment_take_event_receiver() {
        let mut env = Environment::new(None).await;
        let receiver = env.take_event_receiver(None).await;
        assert!(receiver.is_some());

        // Second call should return None
        let receiver2 = env.take_event_receiver(None).await;
        assert!(receiver2.is_none());
    }

    #[tokio::test]
    async fn test_environment_shutdown() {
        let mut env = Environment::new(None).await;
        env.shutdown().await;
        // Should not panic
    }

    #[tokio::test]
    async fn test_environment_with_failing_agent() {
        let env = Environment::new(None).await;
        let agent = Arc::new(MockAgent::new_failing("failing_agent"));
        let agent_id = env.register_agent(agent, None).await.unwrap();

        env.add_task(agent_id, "Test task").await.unwrap();
        let result = env.run(agent_id, None).await;
        assert!(result.is_err());
    }

    #[tokio::test]
    async fn test_environment_error_session_not_found() {
        let env = Environment::new(None).await;
        let non_existent_id = Uuid::new_v4();

        let result = env.get_session_or_default(Some(non_existent_id)).await;
        assert!(result.is_err());

        assert!(result.is_err());
        // Just test that it's an error, not the specific variant
        assert!(result.is_err());
    }

    #[test]
    fn test_environment_error_display() {
        let session_id = Uuid::new_v4();
        let error = EnvironmentError::SessionNotFound(session_id);
        assert!(error.to_string().contains("Session not found"));
        assert!(error.to_string().contains(&session_id.to_string()));
    }
}
