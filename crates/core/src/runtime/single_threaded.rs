use super::{Runtime, RuntimeError, Task};
use crate::{
    agent::RunnableAgent,
    error::Error,
    protocol::{AgentID, Event, RuntimeID},
};
use async_trait::async_trait;
use log::{debug, error};
use std::{
    collections::{HashMap, VecDeque},
    sync::{atomic::AtomicBool, Arc},
};
use tokio::sync::{mpsc, Mutex, Notify, RwLock};
use tokio_stream::wrappers::ReceiverStream;
use uuid::Uuid;

const DEFAULT_CHANEL_BUFFER: usize = 100;

#[derive(Debug)]
pub struct SingleThreadedRuntime {
    pub id: RuntimeID,
    tx_event: Mutex<Option<mpsc::Sender<Event>>>,
    rx_event: Mutex<Option<mpsc::Receiver<Event>>>,
    agents: Arc<RwLock<HashMap<AgentID, Arc<dyn RunnableAgent>>>>,
    subscriptions: Arc<RwLock<HashMap<String, Vec<AgentID>>>>,
    shutdown_flag: Arc<AtomicBool>,
    event_queue: Mutex<VecDeque<Event>>,
    event_notify: Notify,
}

impl SingleThreadedRuntime {
    pub fn new(channel_buffer: Option<usize>) -> Arc<Self> {
        let id = Uuid::new_v4();
        let (tx_event, rx_event) = mpsc::channel(channel_buffer.unwrap_or(DEFAULT_CHANEL_BUFFER));
        Arc::new(Self {
            id,
            tx_event: Mutex::new(Some(tx_event)),
            rx_event: Mutex::new(Some(rx_event)),
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
        self.agents.write().await.insert(agent.id(), agent.clone());
    }

    async fn take_event_receiver(&self) -> Option<ReceiverStream<Event>> {
        let mut guard = self.rx_event.lock().await;
        guard.take().map(ReceiverStream::new)
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
                let task = Task::new(message.clone(), Some(*agent_id));
                let event = Event::NewTask {
                    agent_id: *agent_id,
                    task,
                };
                queue.push_back(event);
                self.event_notify.notify_one();
            }
        }

        Ok(())
    }

    async fn send_message(&self, message: String, agent_id: AgentID) -> Result<(), Error> {
        let task = Task::new(message.clone(), Some(agent_id));
        if let Some(agent) = self.agents.read().await.get(&agent_id) {
            agent.clone().spawn_task(task, self.tx_event().await?);
        } else {
            return Err(RuntimeError::AgentNotFound(agent_id).into());
        }
        Ok(())
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
        debug!("Runtime event loop startng");
        loop {
            if self.shutdown_flag.load(std::sync::atomic::Ordering::SeqCst) {
                break;
            }
            let event = {
                let mut queue = self.event_queue.lock().await;
                queue.pop_front()
            };
            match event {
                Some(Event::NewTask { agent_id, task }) => {
                    if let Some(agent) = self.agents.read().await.get(&agent_id) {
                        agent.clone().run(task, self.tx_event().await?).await?;
                    }
                }
                Some(e) => {
                    error!("Unhandled Runtime Task: {e:?}");
                }
                None => {
                    self.event_notify.notified().await;
                }
            }
        }
        debug!("Draning remainging tasks");
        let mut queue = self.event_queue.lock().await;
        while let Some(event) = queue.pop_front() {
            if let Event::NewTask { agent_id, task } = event {
                if let Some(agent) = self.agents.read().await.get(&agent_id) {
                    agent.clone().run(task, self.tx_event().await?).await?;
                }
            }
        }
        Ok(())
    }

    async fn stop(&self) -> Result<(), Error> {
        self.shutdown_flag
            .store(true, std::sync::atomic::Ordering::SeqCst);
        // wake the loop if it's waiting
        self.event_notify.notify_one();
        Ok(())
    }
}
