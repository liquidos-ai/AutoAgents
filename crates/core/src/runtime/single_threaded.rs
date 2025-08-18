use super::{Runtime, RuntimeError};
use crate::{
    actor::ActorMessage,
    error::Error,
    protocol::{ActorID, Event, RuntimeID},
};
use async_trait::async_trait;
use log::{debug, error, info, warn};
use ractor::ActorRef;
use std::{
    collections::HashMap,
    sync::{
        atomic::{AtomicBool, Ordering},
        Arc,
    },
};
use tokio::sync::{mpsc, Mutex, Notify, RwLock};
use tokio_stream::wrappers::ReceiverStream;
use uuid::Uuid;
use crate::agent::task::Task;

const DEFAULT_CHANNEL_BUFFER: usize = 100;
const DEFAULT_INTERNAL_BUFFER: usize = 1000;

/// Internal events that are processed within the runtime
#[derive(Debug, Clone)]
pub enum InternalEvent {
    /// An event from an agent that needs processing
    AgentEvent(Event),
    /// Shutdown signal
    Shutdown,
}

/// Single-threaded runtime implementation with internal event routing
#[derive(Debug)]
pub struct SingleThreadedRuntime {
    pub id: RuntimeID,
    // External event channel for application consumption
    external_tx: mpsc::Sender<Event>,
    external_rx: Mutex<Option<mpsc::Receiver<Event>>>,
    // Internal event channel for runtime processing
    internal_tx: mpsc::Sender<InternalEvent>,
    internal_rx: Mutex<Option<mpsc::Receiver<InternalEvent>>>,
    // actor and subscription management
    actors: Arc<RwLock<HashMap<ActorID, ActorRef<ActorMessage>>>>,
    subscriptions: Arc<RwLock<HashMap<String, Vec<ActorID>>>>,
    // Runtime state
    shutdown_flag: Arc<AtomicBool>,
    shutdown_notify: Arc<Notify>,
}

impl SingleThreadedRuntime {
    pub fn new(channel_buffer: Option<usize>) -> Arc<Self> {
        let id = Uuid::new_v4();
        let buffer_size = channel_buffer.unwrap_or(DEFAULT_CHANNEL_BUFFER);

        // Create channels
        let (external_tx, external_rx) = mpsc::channel(buffer_size);
        let (internal_tx, internal_rx) = mpsc::channel(DEFAULT_INTERNAL_BUFFER);

        Arc::new(Self {
            id,
            external_tx,
            external_rx: Mutex::new(Some(external_rx)),
            internal_tx,
            internal_rx: Mutex::new(Some(internal_rx)),
            actors: Arc::new(RwLock::new(HashMap::new())),
            subscriptions: Arc::new(RwLock::new(HashMap::new())),
            shutdown_flag: Arc::new(AtomicBool::new(false)),
            shutdown_notify: Arc::new(Notify::new()),
        })
    }

    /// Creates an event sender that intercepts specific events for internal processing
    fn create_intercepting_sender(&self) -> mpsc::Sender<Event> {
        let internal_tx = self.internal_tx.clone();
        let (interceptor_tx, mut interceptor_rx) = mpsc::channel(DEFAULT_CHANNEL_BUFFER);

        tokio::spawn(async move {
            while let Some(event) = interceptor_rx.recv().await {
                if let Err(e) = internal_tx.send(InternalEvent::AgentEvent(event)).await {
                    error!("Failed to forward event to internal channel: {e}");
                    break;
                }
            }
        });

        interceptor_tx
    }

    async fn process_internal_event(&self, event: InternalEvent) -> Result<(), Error> {
        match event {
            InternalEvent::AgentEvent(event) => {
                self.process_agent_event(event).await?;
            }
            InternalEvent::Shutdown => {
                self.shutdown_flag.store(true, Ordering::SeqCst);
                self.shutdown_notify.notify_waiters();
            }
        }
        Ok(())
    }

    async fn process_agent_event(&self, event: Event) -> Result<(), Error> {
        match event {
            Event::PublishMessage { topic, message } => {
                debug!("Processing publish message to topic: {topic}");
                self.handle_publish_message(topic, message).await?;
            }
            Event::SendMessage { actor_id, message } => {
                debug!("Processing send message to agent: {actor_id:?}");
                self.handle_send_message(actor_id, message).await?;
            }
            _ => {
                // All other events are forwarded to external channel
                self.external_tx
                    .send(event)
                    .await
                    .map_err(RuntimeError::EventError)?;
            }
        }
        Ok(())
    }

    async fn handle_publish_message(&self, topic: String, message: String) -> Result<(), Error> {
        let subscriptions = self.subscriptions.read().await;

        if let Some(agents) = subscriptions.get(&topic) {
            debug!(
                "Publishing message to topic '{}' with {} subscribers",
                topic,
                agents.len()
            );

            for agent_id in agents {
                let task = Task::new(message.clone(), Some(*agent_id));
                self.execute_task_on_agent(*agent_id, task).await?;
            }
        } else {
            debug!("No subscribers for topic: {topic}");
        }

        Ok(())
    }

    async fn handle_send_message(&self, actor_id: ActorID, message: String) -> Result<(), Error> {
        let task = Task::new(message, Some(actor_id));
        self.execute_task_on_agent(actor_id, task).await
    }

    async fn execute_task_on_agent(&self, actor_id: ActorID, task: Task) -> Result<(), Error> {
        let actors = self.actors.read().await;

        if let Some(actor) = actors.get(&actor_id) {
            debug!("Executing task on agent: {actor_id:?}");

            // Create a new task event and send it to external channel first
            self.external_tx
                .send(Event::NewTask {
                    actor_id,
                    task: task.clone(),
                })
                .await
                .map_err(RuntimeError::EventError)?;

            // Create intercepting sender for this agent
            let tx = self.create_intercepting_sender();

            // Use spawn_task for async execution
            let _ = actor.send_message(ActorMessage {
                task: Box::new(task),
                tx,
            });
        } else {
            warn!("Agent not found: {actor_id:?}");
            return Err(RuntimeError::AgentNotFound(actor_id).into());
        }

        Ok(())
    }
}

#[async_trait]
impl Runtime for SingleThreadedRuntime {
    fn id(&self) -> RuntimeID {
        self.id
    }

    async fn publish_message(&self, message: String, topic: String) -> Result<(), Error> {
        debug!(
            "Runtime received publish_message request for topic: {}",
            topic
        );

        // Send the publish event through internal channel
        self.internal_tx
            .send(InternalEvent::AgentEvent(Event::PublishMessage {
                topic,
                message,
            }))
            .await
            .map_err(RuntimeError::InternalEventError)?;

        Ok(())
    }

    async fn send_message(&self, message: String, actor_id: ActorID) -> Result<(), Error> {
        debug!(
            "Runtime received send_message request to agent: {:?}",
            actor_id
        );

        // Send the event through internal channel
        self.internal_tx
            .send(InternalEvent::AgentEvent(Event::SendMessage {
                actor_id,
                message,
            }))
            .await
            .map_err(RuntimeError::InternalEventError)?;

        Ok(())
    }

    async fn register_agent(&self, id: ActorID, agent: ActorRef<ActorMessage>) -> Result<(), Error> {
        info!("Registering agent: {:?}", id);
        self.actors.write().await.insert(id, agent);
        Ok(())
    }

    async fn subscribe(&self, actor_id: ActorID, topic: String) -> Result<(), Error> {
        info!("Agent {:?} subscribing to topic: {}", actor_id, topic);

        let mut subscriptions = self.subscriptions.write().await;
        let agents = subscriptions.entry(topic).or_insert_with(Vec::new);

        if !agents.contains(&actor_id) {
            agents.push(actor_id);
        }

        Ok(())
    }

    async fn take_event_receiver(&self) -> Option<ReceiverStream<Event>> {
        self.external_rx
            .lock()
            .await
            .take()
            .map(ReceiverStream::new)
    }

    async fn run(&self) -> Result<(), Error> {
        info!("Runtime starting");

        // Take the internal receiver
        let mut internal_rx = self
            .internal_rx
            .lock()
            .await
            .take()
            .ok_or(RuntimeError::EmptyTask)?;

        // Process events until shutdown
        loop {
            tokio::select! {
                // Process internal events
                Some(event) = internal_rx.recv() => {
                    debug!("Processing internal event: {event:?}");
                    if let Err(e) = self.process_internal_event(event).await {
                        error!("Error processing internal event: {e}");
                    }
                }
                // Check for shutdown
                _ = self.shutdown_notify.notified() => {
                    if self.shutdown_flag.load(Ordering::SeqCst) {
                        info!("Runtime received shutdown signal");
                        break;
                    }
                }
            }
        }

        // Drain remaining events
        info!("Draining remaining events before shutdown");
        while let Ok(event) = internal_rx.try_recv() {
            if let Err(e) = self.process_internal_event(event).await {
                error!("Error processing event during shutdown: {e}");
            }
        }

        info!("Runtime stopped");
        Ok(())
    }

    async fn stop(&self) -> Result<(), Error> {
        info!("Initiating runtime shutdown");

        // Send shutdown signal
        let _ = self.internal_tx.send(InternalEvent::Shutdown).await;

        Ok(())
    }
}

/*
#[cfg(test)]
mod tests {
    use super::*;
    use crate::protocol::TaskResult;
    use crate::{agent::AgentDeriveT, memory::MemoryProvider};
    use tokio::time::{sleep, Duration};

    #[derive(Debug, Clone)]
    struct MockAgent {
        id: AgentID,
    }

    #[async_trait]
    impl<T: AgentDeriveT> RunnableAgent<T> for MockAgent {
        fn id(&self) -> AgentID {
            self.id
        }

        fn name(&self) -> &'static str {
            "test"
        }

        fn description(&self) -> &'static str {
            "test"
        }

        fn memory(&self) -> Option<Arc<RwLock<Box<dyn MemoryProvider>>>> {
            None
        }

        async fn run(self: Arc<Self>, task: Task, tx: mpsc::Sender<Event>) -> Result<(), Error> {
            // Send task started event
            tx.send(Event::TaskStarted {
                sub_id: task.submission_id,
                agent_id: self.id,
                task_description: task.prompt.clone(),
            })
            .await
            .unwrap();

            // Simulate some work
            sleep(Duration::from_millis(10)).await;

            // Send task complete event
            tx.send(Event::TaskComplete {
                sub_id: task.submission_id,
                result: TaskResult::Value(serde_json::json!({
                    "message": "Task completed successfully"
                })),
            })
            .await
            .unwrap();

            Ok(())
        }
    }

    #[tokio::test]
    async fn test_runtime_creation() {
        let runtime = SingleThreadedRuntime::new(None);
        assert_ne!(runtime.id(), Uuid::nil());
    }

    #[tokio::test]
    async fn test_agent_registration() {
        let runtime = SingleThreadedRuntime::new(None);
        let agent = Arc::new(MockAgent { id: Uuid::new_v4() });

        runtime.register_agent(agent.clone()).await.unwrap();

        let agents = runtime.agents.read().await;
        assert!(agents.contains_key(&agent.id()));
    }

    #[tokio::test]
    async fn test_subscription() {
        let runtime = SingleThreadedRuntime::new(None);
        let agent_id = Uuid::new_v4();
        let topic = "test_topic".to_string();

        runtime.subscribe(agent_id, topic.clone()).await.unwrap();

        let subscriptions = runtime.subscriptions.read().await;
        assert!(subscriptions.contains_key(&topic));
        assert!(subscriptions.get(&topic).unwrap().contains(&agent_id));
    }
}
*/
