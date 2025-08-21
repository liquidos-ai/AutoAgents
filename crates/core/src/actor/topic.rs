use crate::actor::messaging::ActorMessage;
use std::any::TypeId;
use std::fmt::Debug;
use std::marker::PhantomData;
use uuid::Uuid;

// Generic topic that is type-safe at compile time
#[derive(Clone)]
pub struct Topic<M: ActorMessage> {
    name: String,
    id: Uuid,
    _phantom: PhantomData<M>,
}
impl<M: ActorMessage> Topic<M> {
    pub fn new(name: impl Into<String>) -> Self {
        Topic {
            name: name.into(),
            id: Uuid::new_v4(),
            _phantom: PhantomData,
        }
    }

    pub fn name(&self) -> &str {
        &self.name
    }

    pub fn id(&self) -> Uuid {
        self.id
    }

    pub fn type_id(&self) -> TypeId {
        TypeId::of::<M>()
    }
}

impl<M: ActorMessage> Debug for Topic<M> {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.debug_struct("Topic")
            .field("name", &self.name)
            .field("id", &self.id)
            .finish()
    }
}