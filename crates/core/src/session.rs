use crate::agent::{AgentRunResult, RunnableAgent, RunnableAgentError};
use crate::error::Error;
use crate::protocol::{AgentID, Event, SessionId, SubmissionId};
use serde::{Deserialize, Serialize};
use serde_json::Value;
use std::collections::HashMap;
use std::sync::Arc;
use tokio::sync::mpsc::error::SendError;
use tokio::sync::{mpsc, Mutex, RwLock};
use tokio::task::JoinError;
use tokio_stream::wrappers::ReceiverStream;
use uuid::Uuid;

/// Error types for Session operations
#[derive(Debug, thiserror::Error)]
pub enum SessionError {
    #[error("Agent not found: {0}")]
    AgentNotFound(Uuid),

    #[error("No task set for agent: {0}")]
    NoTaskSet(Uuid),

    #[error("Task is None")]
    EmptyTask,

    #[error("Task join error: {0}")]
    TaskJoinError(#[from] JoinError),

    #[error("Event error: {0}")]
    EventError(#[from] SendError<Event>),

    #[error("RunnableAgent error: {0}")]
    RunnableAgentError(#[from] RunnableAgentError),
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct Task {
    pub prompt: String,
    pub submission_id: SubmissionId,
    pub completed: bool,
    pub result: Option<Value>,
    agent_id: Option<AgentID>,
}

impl Task {
    pub fn new<T: Into<String>>(task: T, agent_id: Option<AgentID>) -> Self {
        Self {
            prompt: task.into(),
            submission_id: Uuid::new_v4(),
            completed: false,
            result: None,
            agent_id,
        }
    }
}

#[derive(Debug, Default)]
struct State {
    current_task: Option<Task>,
    task_queue: Vec<Task>,
}

pub struct Session {
    pub id: SessionId,
    tx_event: mpsc::Sender<Event>,
    rx_event: Option<mpsc::Receiver<Event>>,
    state: Arc<Mutex<State>>,
    agents: Arc<RwLock<HashMap<AgentID, Arc<dyn RunnableAgent + Send + Sync + 'static>>>>,
}

impl Session {
    pub fn new(channel_buffer: usize) -> (SessionId, Self) {
        let id = Uuid::new_v4();
        let (tx_event, rx_event) = mpsc::channel(channel_buffer);
        (
            id,
            Self {
                id,
                tx_event,
                rx_event: Some(rx_event),
                state: Arc::new(Mutex::new(State::default())),
                agents: Arc::new(RwLock::new(HashMap::new())),
            },
        )
    }

    pub async fn add_task(&self, task: Task) -> Result<(), Error> {
        let task_clone = task.clone();
        self.tx_event
            .send(Event::NewTask {
                sub_id: task_clone.submission_id,
                agent_id: task_clone.agent_id,
                prompt: task_clone.prompt,
            })
            .await
            .map_err(SessionError::EventError)?;
        let mut state = self.state.lock().await;
        state.task_queue.push(task);
        Ok(())
    }

    pub async fn set_current_task(&self, task: Task) {
        let mut state = self.state.lock().await;
        state.current_task = Some(task);
    }

    pub async fn is_task_queue_empty(&self) -> bool {
        let state = self.state.lock().await;
        state.task_queue.is_empty()
    }

    pub async fn get_top_task(&self) -> Option<Task> {
        let mut state = self.state.lock().await;
        if state.task_queue.is_empty() {
            None
        } else {
            let task = state.task_queue.remove(0);
            state.current_task = Some(task.clone());
            Some(task)
        }
    }

    pub async fn get_current_task(&self) -> Option<Task> {
        let state = self.state.lock().await;
        state.current_task.clone()
    }

    pub async fn get_task(&self, sub_id: SubmissionId) -> Option<Task> {
        let state = self.state.lock().await;
        state
            .task_queue
            .iter()
            .find(|t| t.submission_id == sub_id)
            .cloned()
    }

    pub fn event_sender(&self) -> mpsc::Sender<Event> {
        self.tx_event.clone()
    }

    pub fn take_event_receiver(&mut self) -> Option<ReceiverStream<Event>> {
        self.rx_event.take().map(ReceiverStream::new)
    }

    pub async fn register_agent_with_id(&self, agent_id: Uuid, agent: Arc<dyn RunnableAgent>) {
        self.agents.write().await.insert(agent_id, agent);
    }

    pub async fn run_task(
        &self,
        task: Option<Task>,
        agent_id: AgentID,
    ) -> Result<AgentRunResult, Error> {
        let task = task.ok_or_else(|| SessionError::EmptyTask)?;
        let agent = self
            .agents
            .read()
            .await
            .get(&agent_id)
            .ok_or(SessionError::AgentNotFound(agent_id))?
            .clone();

        let join_handle = agent.spawn_task(task, self.tx_event.clone());

        // Await the task completion:
        let result = join_handle.await.map_err(SessionError::TaskJoinError);
        result?
    }

    pub async fn run(&self, agent_id: Uuid) -> Result<AgentRunResult, Error> {
        let task = self.get_top_task().await;
        self.run_task(task, agent_id).await
    }

    pub async fn run_all(&self, agent_id: Uuid) -> Result<Vec<AgentRunResult>, Error> {
        let mut results = Vec::new();
        while !self.is_task_queue_empty().await {
            let result = self.run(agent_id).await?;
            results.push(result);
        }
        Ok(results)
    }
}

#[derive(Default)]
pub struct SessionManager {
    sessions: RwLock<HashMap<SessionId, Arc<Mutex<Session>>>>,
}

impl SessionManager {
    pub fn new() -> Self {
        Self::default()
    }

    pub async fn create_session(&self, channel_buffer: usize) -> (SessionId, Arc<Mutex<Session>>) {
        let (id, session) = Session::new(channel_buffer);
        let session_arc = Arc::new(Mutex::new(session));
        let mut sessions = self.sessions.write().await;
        sessions.insert(id, session_arc.clone());
        (id, session_arc)
    }

    pub async fn get_session(&self, session_id: &SessionId) -> Option<Arc<Mutex<Session>>> {
        let sessions = self.sessions.read().await;
        sessions.get(session_id).cloned()
    }

    pub async fn get_session_mut(&self, session_id: &SessionId) -> Option<Arc<Mutex<Session>>> {
        // This can be same as get_session, since the interior mutability is in Mutex
        let sessions = self.sessions.read().await;
        sessions.get(session_id).cloned()
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::agent::{AgentRunResult, RunnableAgent};
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
            task: Task,
            _tx_event: mpsc::Sender<Event>,
        ) -> Result<AgentRunResult, crate::error::Error> {
            if self.should_fail {
                Err(crate::error::Error::SessionError(SessionError::EmptyTask))
            } else {
                Ok(AgentRunResult::success(serde_json::json!({
                    "response": format!("Processed: {}", task.prompt)
                })))
            }
        }

        fn memory(
            &self,
        ) -> Option<Arc<tokio::sync::RwLock<Box<dyn crate::memory::MemoryProvider>>>> {
            None
        }
    }

    #[tokio::test]
    async fn test_task_creation() {
        let task = Task::new("Test prompt", Some(Uuid::new_v4()));
        assert_eq!(task.prompt, "Test prompt");
        assert!(!task.completed);
        assert!(task.result.is_none());
        assert!(task.agent_id.is_some());
    }

    #[tokio::test]
    async fn test_task_creation_without_agent() {
        let task = Task::new("Test prompt", None);
        assert_eq!(task.prompt, "Test prompt");
        assert!(task.agent_id.is_none());
    }

    #[tokio::test]
    async fn test_session_creation() {
        let (session_id, session) = Session::new(100);
        assert_eq!(session.id, session_id);

        let session_locked = session;
        assert!(session_locked.is_task_queue_empty().await);
        assert!(session_locked.get_current_task().await.is_none());
    }

    #[tokio::test]
    async fn test_session_add_task() {
        let (_session_id, session) = Session::new(100);
        let session_locked = session;

        let task = Task::new("Test task", None);
        let result = session_locked.add_task(task).await;
        assert!(result.is_ok());
        assert!(!session_locked.is_task_queue_empty().await);
    }

    #[tokio::test]
    async fn test_session_get_top_task() {
        let (_session_id, session) = Session::new(100);
        let session_locked = session;

        let task1 = Task::new("First task", None);
        let task2 = Task::new("Second task", None);

        session_locked.add_task(task1).await.unwrap();
        session_locked.add_task(task2).await.unwrap();

        let top_task = session_locked.get_top_task().await;
        assert!(top_task.is_some());
        assert_eq!(top_task.unwrap().prompt, "First task");

        // Should now have second task as current
        let current_task = session_locked.get_current_task().await;
        assert!(current_task.is_some());
        assert_eq!(current_task.unwrap().prompt, "First task");
    }

    #[tokio::test]
    async fn test_session_get_task_by_id() {
        let (_session_id, session) = Session::new(100);
        let session_locked = session;

        let task = Task::new("Test task", None);
        let task_id = task.submission_id;
        session_locked.add_task(task).await.unwrap();

        let retrieved_task = session_locked.get_task(task_id).await;
        assert!(retrieved_task.is_some());
        assert_eq!(retrieved_task.unwrap().prompt, "Test task");
    }

    #[tokio::test]
    async fn test_session_register_agent() {
        let (_session_id, session) = Session::new(100);
        let session_locked = session;

        let agent = Arc::new(MockAgent::new("test_agent"));
        let agent_id = Uuid::new_v4();

        session_locked.register_agent_with_id(agent_id, agent).await;

        // Verify agent was registered by trying to run a task
        let task = Task::new("Test task", Some(agent_id));
        let result = session_locked.run_task(Some(task), agent_id).await;
        assert!(result.is_ok());
    }

    #[tokio::test]
    async fn test_session_run_task_with_missing_agent() {
        let (_session_id, session) = Session::new(100);
        let session_locked = session;

        let non_existent_agent_id = Uuid::new_v4();
        let task = Task::new("Test task", Some(non_existent_agent_id));

        let result = session_locked
            .run_task(Some(task), non_existent_agent_id)
            .await;
        assert!(result.is_err());
        assert!(matches!(
            result.unwrap_err(),
            crate::error::Error::SessionError(SessionError::AgentNotFound(_))
        ));
    }

    #[tokio::test]
    async fn test_session_run_task_with_none_task() {
        let (_session_id, session) = Session::new(100);
        let session_locked = session;

        let agent = Arc::new(MockAgent::new("test_agent"));
        let agent_id = Uuid::new_v4();
        session_locked.register_agent_with_id(agent_id, agent).await;

        let result = session_locked.run_task(None, agent_id).await;
        assert!(result.is_err());
        assert!(matches!(
            result.unwrap_err(),
            crate::error::Error::SessionError(SessionError::EmptyTask)
        ));
    }

    #[tokio::test]
    async fn test_session_run_with_failing_agent() {
        let (_session_id, session) = Session::new(100);
        let session_locked = session;

        let agent = Arc::new(MockAgent::new_failing("failing_agent"));
        let agent_id = Uuid::new_v4();
        session_locked.register_agent_with_id(agent_id, agent).await;

        let task = Task::new("Test task", Some(agent_id));
        session_locked.add_task(task).await.unwrap();

        let result = session_locked.run(agent_id).await;
        assert!(result.is_err());
    }

    #[tokio::test]
    async fn test_session_run_all() {
        let (_session_id, session) = Session::new(100);
        let session_locked = session;

        let agent = Arc::new(MockAgent::new("test_agent"));
        let agent_id = Uuid::new_v4();
        session_locked.register_agent_with_id(agent_id, agent).await;

        // Add multiple tasks
        for i in 1..=3 {
            let task = Task::new(format!("Task {}", i), Some(agent_id));
            session_locked.add_task(task).await.unwrap();
        }

        let results = session_locked.run_all(agent_id).await;
        assert!(results.is_ok());
        assert_eq!(results.unwrap().len(), 3);
    }

    #[tokio::test]
    async fn test_session_set_current_task() {
        let (_session_id, session) = Session::new(100);
        let session_locked = session;

        let task = Task::new("Current task", None);
        session_locked.set_current_task(task.clone()).await;

        let current_task = session_locked.get_current_task().await;
        assert!(current_task.is_some());
        assert_eq!(current_task.unwrap().prompt, "Current task");
    }

    #[tokio::test]
    async fn test_session_manager_create_session() {
        let manager = SessionManager::new();
        let (session_id, session_arc) = manager.create_session(100).await;

        let session = session_arc.lock().await;
        assert_eq!(session.id, session_id);
    }

    #[tokio::test]
    async fn test_session_manager_get_session() {
        let manager = SessionManager::new();
        let (session_id, _) = manager.create_session(100).await;

        let retrieved_session = manager.get_session(&session_id).await;
        assert!(retrieved_session.is_some());

        let non_existent_id = Uuid::new_v4();
        let missing_session = manager.get_session(&non_existent_id).await;
        assert!(missing_session.is_none());
    }

    #[tokio::test]
    async fn test_session_manager_get_session_mut() {
        let manager = SessionManager::new();
        let (session_id, _) = manager.create_session(100).await;

        let retrieved_session = manager.get_session_mut(&session_id).await;
        assert!(retrieved_session.is_some());

        let non_existent_id = Uuid::new_v4();
        let missing_session = manager.get_session_mut(&non_existent_id).await;
        assert!(missing_session.is_none());
    }

    #[tokio::test]
    async fn test_session_event_sender() {
        let (_session_id, session) = Session::new(100);
        let session_locked = session;

        let sender = session_locked.event_sender();

        // Test that we can send an event
        let event = Event::NewTask {
            sub_id: Uuid::new_v4(),
            agent_id: Some(Uuid::new_v4()),
            prompt: "Test prompt".to_string(),
        };

        let result = sender.send(event).await;
        assert!(result.is_ok());
    }

    #[tokio::test]
    async fn test_session_take_event_receiver() {
        let (_session_id, mut session) = Session::new(100);

        let receiver = session.take_event_receiver();
        assert!(receiver.is_some());

        // Second call should return None
        let receiver2 = session.take_event_receiver();
        assert!(receiver2.is_none());
    }

    #[test]
    fn test_session_error_display() {
        let error = SessionError::AgentNotFound(Uuid::new_v4());
        assert!(error.to_string().contains("Agent not found"));

        let error = SessionError::NoTaskSet(Uuid::new_v4());
        assert!(error.to_string().contains("No task set for agent"));

        let error = SessionError::EmptyTask;
        assert_eq!(error.to_string(), "Task is None");
    }

    #[test]
    fn test_task_serialization() {
        let task = Task::new("Test task", Some(Uuid::new_v4()));
        let serialized = serde_json::to_string(&task).unwrap();
        let deserialized: Task = serde_json::from_str(&serialized).unwrap();

        assert_eq!(task.prompt, deserialized.prompt);
        assert_eq!(task.submission_id, deserialized.submission_id);
        assert_eq!(task.completed, deserialized.completed);
    }
}
