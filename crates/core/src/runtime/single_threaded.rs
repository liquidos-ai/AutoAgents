use super::{Runtime, RuntimeError, State, Task};
use crate::{
    agent::{AgentRunResult, RunnableAgent},
    error::Error,
    protocol::{AgentID, Event, RuntimeID, SubmissionId},
};
use async_trait::async_trait;
use std::{
    collections::{HashMap, VecDeque},
    sync::{atomic::AtomicBool, Arc},
};
use tokio::sync::{mpsc, Mutex, Notify, RwLock};
use tokio_stream::wrappers::ReceiverStream;
use uuid::Uuid;

pub struct SingleThreadedRuntime {
    pub id: RuntimeID,
    tx_event: Mutex<Option<mpsc::Sender<Event>>>,
    rx_event: Mutex<Option<mpsc::Receiver<Event>>>,
    state: Arc<Mutex<State>>,
    agents: Arc<RwLock<HashMap<AgentID, Arc<dyn RunnableAgent>>>>,
    subscriptions: Arc<RwLock<HashMap<String, Vec<AgentID>>>>,
    shutdown_flag: Arc<AtomicBool>,
    event_queue: Mutex<VecDeque<Event>>,
    event_notify: Notify,
}

impl SingleThreadedRuntime {
    pub fn new(channel_buffer: usize) -> Arc<Self> {
        let id = Uuid::new_v4();
        let (tx_event, rx_event) = mpsc::channel(channel_buffer);
        Arc::new(Self {
            id,
            tx_event: Mutex::new(Some(tx_event)),
            rx_event: Mutex::new(Some(rx_event)),
            state: Arc::new(Mutex::new(State::default())),
            agents: Arc::new(RwLock::new(HashMap::new())),
            subscriptions: Arc::new(RwLock::new(HashMap::new())),
            shutdown_flag: Arc::new(AtomicBool::new(false)),
            event_queue: Mutex::new(VecDeque::new()),
            event_notify: Notify::new(),
        })
    }

    async fn tx_event(&self) -> Result<mpsc::Sender<Event>, Error> {
        let tx_lock = self.tx_event.lock().await;
        if let Some(sender) = tx_lock.as_ref() {
            Ok(sender.clone())
        } else {
            Err(RuntimeError::EmptyTask.into())
        }
    }

    async fn add_agent(&self, agent: Arc<dyn RunnableAgent>) {
        println!("Add agent");
        self.agents.write().await.insert(agent.id(), agent.clone());
    }

    pub async fn add_task(&self, task: Task) -> Result<(), Error> {
        let task_clone = task.clone();
        self.tx_event()
            .await?
            .send(Event::NewTask {
                sub_id: task_clone.submission_id,
                agent_id: task_clone.agent_id,
                prompt: task_clone.prompt,
            })
            .await
            .map_err(RuntimeError::EventError)?;
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

    pub async fn event_sender(&self) -> mpsc::Sender<Event> {
        self.tx_event().await.unwrap()
    }

    async fn take_event_receiver(&self) -> Option<ReceiverStream<Event>> {
        let mut guard = self.rx_event.lock().await;
        guard.take().map(ReceiverStream::new)
    }

    pub async fn run_task(
        &self,
        task: Option<Task>,
        agent_id: AgentID,
    ) -> Result<AgentRunResult, Error> {
        let task = task.ok_or_else(|| RuntimeError::EmptyTask)?;
        let agent = self
            .agents
            .read()
            .await
            .get(&agent_id)
            .ok_or(RuntimeError::AgentNotFound(agent_id))?
            .clone();

        let join_handle = agent.spawn_task(task, self.tx_event().await?);

        // Await the task completion:
        let result = join_handle.await.map_err(RuntimeError::TaskJoinError);
        result?
    }

    pub async fn run(&self, agent_id: Uuid) -> Result<Vec<AgentRunResult>, Error> {
        let mut results = Vec::new();
        while !self.is_task_queue_empty().await {
            let task = self.get_top_task().await;
            let result = self.run_task(task, agent_id).await?;
            results.push(result);
        }
        Ok(results)
    }
}

#[async_trait]
impl Runtime for SingleThreadedRuntime {
    fn id(&self) -> RuntimeID {
        self.id
    }

    async fn publish_message(&self, message: String, topic: String) -> Result<(), Error> {
        let subscriptions = self.subscriptions.read().await;

        if let Some(agents) = subscriptions.get(&topic) {
            let mut queue = self.event_queue.lock().await;

            for agent_id in agents {
                let task = Task {
                    prompt: message.clone(),
                    submission_id: Uuid::new_v4(),
                    completed: false,
                    result: None,
                    agent_id: None,
                };

                let event = Event::RuntimeTask {
                    agent_id: agent_id.clone(),
                    task,
                };

                queue.push_back(event);
                self.event_notify.notify_one();
            }
        }

        Ok(())
    }

    async fn send_message(&self, message: String, agent_id: AgentID) -> Result<(), Error> {
        let task = Task {
            prompt: message.clone(),
            submission_id: Uuid::new_v4(),
            completed: false,
            result: None,
            agent_id: None,
        };
        if let Some(agent) = self.agents.read().await.get(&agent_id) {
            agent.clone().spawn_task(task, self.tx_event().await?);
        } else {
            return Err(RuntimeError::EmptyTask.into());
        }
        Ok(())
    }
    async fn event_sender(&self) -> mpsc::Sender<Event> {
        self.event_sender().await
    }
    async fn register_agent(&self, agent: Arc<dyn RunnableAgent>) -> Result<(), Error> {
        self.add_agent(agent).await;
        Ok(())
    }

    async fn subscribe(&self, agent_id: AgentID, topic: String) -> Result<(), Error> {
        let mut subscribed_agents = self
            .subscriptions
            .read()
            .await
            .get(&topic)
            .cloned()
            .unwrap_or_default();
        subscribed_agents.push(agent_id);
        self.subscriptions
            .write()
            .await
            .insert(topic, subscribed_agents);
        Ok(())
    }

    async fn take_event_receiver(&self) -> Option<ReceiverStream<Event>> {
        self.take_event_receiver().await
    }

    async fn run(&self) -> Result<(), Error> {
        println!("Runtime event loop starting...");

        loop {
            if self.shutdown_flag.load(std::sync::atomic::Ordering::SeqCst) {
                println!("Shutdown flag detected.");
                break;
            }
            let event = {
                let mut queue = self.event_queue.lock().await;
                queue.pop_front()
            };
            match event {
                Some(Event::RuntimeTask { agent_id, task }) => {
                    if let Some(agent) = self.agents.read().await.get(&agent_id) {
                        let _ = agent.clone().run(task, self.tx_event().await?).await;
                    }
                }
                Some(e) => {
                    println!("Unhandled Event: {:?}", e);
                }
                None => {
                    self.event_notify.notified().await;
                }
            }
        }

        println!("Draining remaining events...");

        let mut queue = self.event_queue.lock().await;
        while let Some(event) = queue.pop_front() {
            match event {
                Event::RuntimeTask { agent_id, task } => {
                    if let Some(agent) = self.agents.read().await.get(&agent_id) {
                        let _ = agent.clone().run(task, self.tx_event().await?).await;
                    }
                }
                _ => {}
            }
        }

        println!("All events processed, shutting down.");
        Ok(())
    }

    async fn stop(&self) -> Result<(), Error> {
        self.shutdown_flag
            .store(true, std::sync::atomic::Ordering::SeqCst);
        Ok(())
    }
}
