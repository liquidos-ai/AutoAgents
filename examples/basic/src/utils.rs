use autoagents::core::utils::BoxEventStream;
use autoagents::protocol::Event;
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
                    println!("✅ Task Completed: {:?}", result);
                }
                _ => {
                    // Ignore other events for this example
                }
            }
        }
    });
}
