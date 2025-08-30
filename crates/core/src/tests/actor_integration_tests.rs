#[cfg(test)]
mod tests {
    use crate::actor::{
        ActorMessage, AnyActor, CloneableMessage, LocalTransport, SharedMessage, Topic, Transport,
    };
    use async_trait::async_trait;
    use std::any::Any;

    use std::sync::{Arc, Mutex};

    // Test message types for integration testing
    #[derive(Debug, Clone, PartialEq)]
    #[cfg_attr(feature = "cluster", derive(serde::Serialize, serde::Deserialize))]
    struct ChatMessage {
        user: String,
        content: String,
        timestamp: u64,
    }

    #[cfg(feature = "cluster")]
    impl ractor::BytesConvertable for ChatMessage {
        fn into_bytes(self) -> Vec<u8> {
            serde_json::to_vec(&self).expect("Failed to serialize ChatMessage")
        }

        fn from_bytes(data: Vec<u8>) -> Self {
            serde_json::from_slice(&data).expect("Failed to deserialize ChatMessage")
        }
    }

    impl ActorMessage for ChatMessage {}
    impl CloneableMessage for ChatMessage {}

    #[derive(Debug, PartialEq)]
    #[cfg_attr(feature = "cluster", derive(serde::Serialize, serde::Deserialize))]
    struct FileData {
        filename: String,
        content: Vec<u8>,
        size: usize,
    }

    #[cfg(feature = "cluster")]
    impl ractor::BytesConvertable for FileData {
        fn into_bytes(self) -> Vec<u8> {
            serde_json::to_vec(&self).expect("Failed to serialize FileData")
        }

        fn from_bytes(data: Vec<u8>) -> Self {
            serde_json::from_slice(&data).expect("Failed to deserialize FileData")
        }
    }

    impl ActorMessage for FileData {}

    #[derive(Debug, PartialEq)]
    #[cfg_attr(feature = "cluster", derive(serde::Serialize, serde::Deserialize))]
    struct NotificationMessage {
        title: String,
        body: String,
        priority: u8,
    }

    #[cfg(feature = "cluster")]
    impl ractor::BytesConvertable for NotificationMessage {
        fn into_bytes(self) -> Vec<u8> {
            serde_json::to_vec(&self).expect("Failed to serialize NotificationMessage")
        }

        fn from_bytes(data: Vec<u8>) -> Self {
            serde_json::from_slice(&data).expect("Failed to deserialize NotificationMessage")
        }
    }

    impl ActorMessage for NotificationMessage {}

    // Mock chat room actor
    #[derive(Debug)]
    struct ChatRoomActor {
        room_name: String,
        messages: Arc<Mutex<Vec<ChatMessage>>>,
        user_count: Arc<Mutex<usize>>,
    }

    impl ChatRoomActor {
        fn new(room_name: String) -> Self {
            Self {
                room_name,
                messages: Arc::new(Mutex::new(Vec::new())),
                user_count: Arc::new(Mutex::new(0)),
            }
        }

        fn get_messages(&self) -> Vec<ChatMessage> {
            self.messages.lock().unwrap().clone()
        }

        fn get_user_count(&self) -> usize {
            *self.user_count.lock().unwrap()
        }
    }

    #[async_trait]
    impl AnyActor for ChatRoomActor {
        async fn send_any(
            &self,
            msg: Arc<dyn Any + Send + Sync>,
        ) -> Result<(), Box<dyn std::error::Error + Send + Sync>> {
            if let Some(chat_msg) = msg.downcast_ref::<ChatMessage>() {
                self.messages.lock().unwrap().push(chat_msg.clone());
                *self.user_count.lock().unwrap() += 1;
                Ok(())
            } else {
                Err(format!(
                    "Chat room {} cannot handle this message type",
                    self.room_name
                )
                .into())
            }
        }

        fn as_any(&self) -> &dyn Any {
            self
        }
    }

    // Mock file processor actor
    #[derive(Debug)]
    struct FileProcessorActor {
        processed_files: Arc<Mutex<Vec<String>>>,
        total_size: Arc<Mutex<usize>>,
        processing_failures: Arc<Mutex<Vec<String>>>,
    }

    impl FileProcessorActor {
        fn new() -> Self {
            Self {
                processed_files: Arc::new(Mutex::new(Vec::new())),
                total_size: Arc::new(Mutex::new(0)),
                processing_failures: Arc::new(Mutex::new(Vec::new())),
            }
        }

        fn get_processed_files(&self) -> Vec<String> {
            self.processed_files.lock().unwrap().clone()
        }

        fn get_total_size(&self) -> usize {
            *self.total_size.lock().unwrap()
        }

        fn get_failures(&self) -> Vec<String> {
            self.processing_failures.lock().unwrap().clone()
        }
    }

    #[async_trait]
    impl AnyActor for FileProcessorActor {
        async fn send_any(
            &self,
            msg: Arc<dyn Any + Send + Sync>,
        ) -> Result<(), Box<dyn std::error::Error + Send + Sync>> {
            if let Some(shared_file) = msg.downcast_ref::<SharedMessage<FileData>>() {
                let file_data = shared_file.inner();

                // Simulate processing based on file extension
                if file_data.filename.ends_with(".corrupted") {
                    self.processing_failures
                        .lock()
                        .unwrap()
                        .push(file_data.filename.clone());
                    return Err("File processing failed".into());
                }

                self.processed_files
                    .lock()
                    .unwrap()
                    .push(file_data.filename.clone());
                *self.total_size.lock().unwrap() += file_data.size;
                Ok(())
            } else {
                Err("File processor can only handle SharedMessage<FileData>".into())
            }
        }

        fn as_any(&self) -> &dyn Any {
            self
        }
    }

    #[tokio::test]
    async fn test_topic_based_messaging_integration() {
        // Create topics for different message types
        let chat_topic = Topic::<ChatMessage>::new("general_chat");
        let file_topic = Topic::<SharedMessage<FileData>>::new("file_processing");

        // Verify topics have correct properties
        assert_eq!(chat_topic.name(), "general_chat");
        assert_eq!(file_topic.name(), "file_processing");
        assert_ne!(chat_topic.type_id(), file_topic.type_id());

        // Test topic cloning
        let chat_topic_clone = chat_topic.clone();
        assert_eq!(chat_topic.name(), chat_topic_clone.name());
        assert_eq!(chat_topic.id(), chat_topic_clone.id());
        assert_eq!(chat_topic.type_id(), chat_topic_clone.type_id());
    }

    #[tokio::test]
    async fn test_transport_with_multiple_actor_types() {
        let transport = LocalTransport;

        // Create different types of actors
        let chat_actor = ChatRoomActor::new("lobby".to_string());
        let file_actor = FileProcessorActor::new();

        // Send appropriate messages to each actor
        let chat_msg = ChatMessage {
            user: "alice".to_string(),
            content: "Hello everyone!".to_string(),
            timestamp: 1234567890,
        };
        let chat_arc_msg: Arc<dyn Any + Send + Sync> = Arc::new(chat_msg.clone());

        let file_data = FileData {
            filename: "document.pdf".to_string(),
            content: vec![1, 2, 3, 4, 5],
            size: 5,
        };
        let shared_file = SharedMessage::new(file_data);
        let file_arc_msg: Arc<dyn Any + Send + Sync> = Arc::new(shared_file);

        // Send messages through transport
        let chat_result = transport.send(&chat_actor, chat_arc_msg).await;
        let file_result = transport.send(&file_actor, file_arc_msg).await;

        // Verify results
        assert!(chat_result.is_ok());
        assert!(file_result.is_ok());

        // Check actor states
        assert_eq!(chat_actor.get_messages().len(), 1);
        assert_eq!(chat_actor.get_messages()[0], chat_msg);
        assert_eq!(chat_actor.get_user_count(), 1);

        assert_eq!(file_actor.get_processed_files(), vec!["document.pdf"]);
        assert_eq!(file_actor.get_total_size(), 5);
    }

    #[tokio::test]
    async fn test_shared_message_zero_copy_semantics() {
        let transport = LocalTransport;

        // Create multiple file processors
        let processor1 = FileProcessorActor::new();
        let processor2 = FileProcessorActor::new();
        let processor3 = FileProcessorActor::new();

        // Create large file data that we don't want to copy
        let large_file = FileData {
            filename: "large_video.mp4".to_string(),
            content: vec![0u8; 10000], // 10KB of data
            size: 10000,
        };

        // Wrap in SharedMessage to enable zero-copy sharing
        let shared_file = SharedMessage::new(large_file);
        let arc_msg: Arc<dyn Any + Send + Sync> = Arc::new(shared_file);

        // Send same message to all processors (should not copy the inner data)
        let result1 = transport.send(&processor1, arc_msg.clone()).await;
        let result2 = transport.send(&processor2, arc_msg.clone()).await;
        let result3 = transport.send(&processor3, arc_msg.clone()).await;

        // All should succeed
        assert!(result1.is_ok());
        assert!(result2.is_ok());
        assert!(result3.is_ok());

        // All processors should have processed the same file
        assert_eq!(processor1.get_processed_files(), vec!["large_video.mp4"]);
        assert_eq!(processor2.get_processed_files(), vec!["large_video.mp4"]);
        assert_eq!(processor3.get_processed_files(), vec!["large_video.mp4"]);

        // All should report the same size
        assert_eq!(processor1.get_total_size(), 10000);
        assert_eq!(processor2.get_total_size(), 10000);
        assert_eq!(processor3.get_total_size(), 10000);
    }

    #[tokio::test]
    async fn test_error_handling_in_transport_chain() {
        let transport = LocalTransport;

        // Create file processor
        let processor = FileProcessorActor::new();

        // Create some files, including a corrupted one
        let good_file = FileData {
            filename: "good_file.txt".to_string(),
            content: vec![1, 2, 3],
            size: 3,
        };
        let bad_file = FileData {
            filename: "bad_file.corrupted".to_string(),
            content: vec![],
            size: 0,
        };

        let good_shared = SharedMessage::new(good_file);
        let bad_shared = SharedMessage::new(bad_file);

        let good_msg: Arc<dyn Any + Send + Sync> = Arc::new(good_shared);
        let bad_msg: Arc<dyn Any + Send + Sync> = Arc::new(bad_shared);

        // Send messages
        let good_result = transport.send(&processor, good_msg).await;
        let bad_result = transport.send(&processor, bad_msg).await;

        // Good should succeed, bad should fail
        assert!(good_result.is_ok());
        assert!(bad_result.is_err());

        // Check final state
        assert_eq!(processor.get_processed_files(), vec!["good_file.txt"]);
        assert_eq!(processor.get_failures(), vec!["bad_file.corrupted"]);
        assert_eq!(processor.get_total_size(), 3);
    }

    #[tokio::test]
    async fn test_concurrent_message_processing() {
        let transport = Arc::new(LocalTransport);
        let chat_actor = Arc::new(ChatRoomActor::new("busy_room".to_string()));

        // Send many messages concurrently
        let handles = (0..50)
            .map(|i| {
                let transport = Arc::clone(&transport);
                let actor = Arc::clone(&chat_actor);

                tokio::spawn(async move {
                    let msg = ChatMessage {
                        user: format!("user_{i}"),
                        content: format!("Message number {i}"),
                        timestamp: 1234567890 + i,
                    };
                    let arc_msg: Arc<dyn Any + Send + Sync> = Arc::new(msg);
                    transport.send(actor.as_ref(), arc_msg).await
                })
            })
            .collect::<Vec<_>>();

        // Wait for all messages to be processed
        let results = futures::future::join_all(handles).await;

        // All should succeed
        for result in results {
            assert!(result.is_ok());
            assert!(result.unwrap().is_ok());
        }

        // Should have received all messages
        assert_eq!(chat_actor.get_messages().len(), 50);
        assert_eq!(chat_actor.get_user_count(), 50);

        // Verify message content variety
        let messages = chat_actor.get_messages();
        let unique_users: std::collections::HashSet<_> = messages.iter().map(|m| &m.user).collect();
        assert_eq!(unique_users.len(), 50);
    }

    #[tokio::test]
    async fn test_mixed_message_types_routing() {
        let transport = LocalTransport;

        // Create actors that handle different message types
        let chat_actor = ChatRoomActor::new("mixed_room".to_string());
        let file_actor = FileProcessorActor::new();

        // Create mixed message types
        let chat_msg = ChatMessage {
            user: "admin".to_string(),
            content: "System message".to_string(),
            timestamp: 1111111111,
        };

        let file_data = FileData {
            filename: "system_log.txt".to_string(),
            content: vec![10, 20, 30],
            size: 3,
        };

        // Try sending wrong message type to each actor
        let chat_arc_msg: Arc<dyn Any + Send + Sync> = Arc::new(chat_msg.clone());
        let shared_file = SharedMessage::new(file_data);
        let file_arc_msg: Arc<dyn Any + Send + Sync> = Arc::new(shared_file);

        // Correct routing should work
        let correct_chat = transport.send(&chat_actor, chat_arc_msg.clone()).await;
        let correct_file = transport.send(&file_actor, file_arc_msg.clone()).await;

        assert!(correct_chat.is_ok());
        assert!(correct_file.is_ok());

        // Wrong routing should fail
        let wrong_chat = transport.send(&chat_actor, file_arc_msg).await;
        let wrong_file = transport.send(&file_actor, chat_arc_msg).await;

        assert!(wrong_chat.is_err());
        assert!(wrong_file.is_err());

        // Verify only correct messages were processed
        assert_eq!(chat_actor.get_messages().len(), 1);
        assert_eq!(file_actor.get_processed_files().len(), 1);
    }

    #[test]
    fn test_topic_type_safety_compilation() {
        // This test verifies compile-time type safety
        let chat_topic: Topic<ChatMessage> = Topic::new("chat");
        let file_topic: Topic<SharedMessage<FileData>> = Topic::new("files");
        let notification_topic: Topic<NotificationMessage> = Topic::new("notifications");

        // Type IDs should be different for different message types
        assert_ne!(chat_topic.type_id(), file_topic.type_id());
        assert_ne!(chat_topic.type_id(), notification_topic.type_id());
        assert_ne!(file_topic.type_id(), notification_topic.type_id());

        // Same message types should have same type IDs
        let another_chat_topic: Topic<ChatMessage> = Topic::new("another_chat");
        assert_eq!(chat_topic.type_id(), another_chat_topic.type_id());
    }

    #[test]
    fn test_shared_message_memory_efficiency() {
        // Test that SharedMessage provides efficient memory sharing
        let large_data = FileData {
            filename: "huge_file.bin".to_string(),
            content: vec![42u8; 100000], // 100KB
            size: 100000,
        };

        let shared1 = SharedMessage::new(large_data);
        let shared2 = shared1.clone();
        let shared3 = shared2.clone();

        // All should reference the same inner data
        assert!(std::ptr::eq(shared1.inner(), shared2.inner()));
        assert!(std::ptr::eq(shared2.inner(), shared3.inner()));

        // We can test that the data is shared by checking content equality
        assert_eq!(shared1.inner().filename, shared2.inner().filename);
        assert_eq!(shared1.inner().size, shared3.inner().size);
    }

    #[tokio::test]
    async fn test_end_to_end_message_flow() {
        // Simulate a complete message flow from creation to processing
        let transport = LocalTransport;

        // Create a processing pipeline
        let input_processor = FileProcessorActor::new();
        let backup_processor = FileProcessorActor::new();

        // Create topic for the data flow
        let _processing_topic = Topic::<SharedMessage<FileData>>::new("processing_pipeline");

        // Create batch of files to process
        let files = vec![
            FileData {
                filename: "doc1.txt".to_string(),
                content: vec![1, 2],
                size: 2,
            },
            FileData {
                filename: "doc2.txt".to_string(),
                content: vec![3, 4, 5],
                size: 3,
            },
            FileData {
                filename: "doc3.txt".to_string(),
                content: vec![6],
                size: 1,
            },
        ];

        // Process files through both processors
        for file_data in files {
            let shared_file = SharedMessage::new(file_data);
            let arc_msg: Arc<dyn Any + Send + Sync> = Arc::new(shared_file);

            // Send to both processors (simulating fan-out)
            let result1 = transport.send(&input_processor, arc_msg.clone()).await;
            let result2 = transport.send(&backup_processor, arc_msg).await;

            assert!(result1.is_ok());
            assert!(result2.is_ok());
        }

        // Verify both processors handled all files
        assert_eq!(input_processor.get_processed_files().len(), 3);
        assert_eq!(backup_processor.get_processed_files().len(), 3);

        // Verify total sizes
        assert_eq!(input_processor.get_total_size(), 6); // 2 + 3 + 1
        assert_eq!(backup_processor.get_total_size(), 6);

        // Verify file names match
        let expected_files = vec!["doc1.txt", "doc2.txt", "doc3.txt"];
        assert_eq!(input_processor.get_processed_files(), expected_files);
        assert_eq!(backup_processor.get_processed_files(), expected_files);
    }
}
