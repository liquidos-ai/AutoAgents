use crate::agent::error::RunnableAgentError;
use crate::agent::result::AgentRunResult;
use crate::agent::runnable::RunnableAgent;
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

    pub async fn add_task(&self, task: Task) -> Result<(), SessionError> {
        let task_clone = task.clone();
        self.tx_event
            .send(Event::NewTask {
                sub_id: task_clone.submission_id,
                agent_id: task_clone.agent_id,
                prompt: task_clone.prompt,
            })
            .await?;
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
    ) -> Result<AgentRunResult, SessionError> {
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
        let result = join_handle.await?;
        Ok(result?)
    }

    pub async fn run(&self, agent_id: Uuid) -> Result<AgentRunResult, SessionError> {
        let task = self.get_top_task().await;
        self.run_task(task, agent_id).await
    }

    pub async fn run_all(&self, agent_id: Uuid) -> Result<Vec<AgentRunResult>, SessionError> {
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
