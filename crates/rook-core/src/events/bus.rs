//! Event bus using tokio broadcast channel
//!
//! Provides non-blocking event emission with multiple subscribers.
//! Slow subscribers will miss events rather than blocking senders.

use crate::events::MemoryLifecycleEvent;
use tokio::sync::broadcast;

/// Default channel capacity
const DEFAULT_CAPACITY: usize = 1024;

/// Event bus for memory lifecycle events
///
/// Uses tokio broadcast channel internally. Events are fire-and-forget;
/// if no subscribers are listening, events are simply dropped.
pub struct EventBus {
    sender: broadcast::Sender<MemoryLifecycleEvent>,
}

impl EventBus {
    /// Create a new event bus with default capacity
    pub fn new() -> Self {
        Self::with_capacity(DEFAULT_CAPACITY)
    }

    /// Create a new event bus with custom capacity
    pub fn with_capacity(capacity: usize) -> Self {
        let (sender, _) = broadcast::channel(capacity);
        Self { sender }
    }

    /// Subscribe to events
    ///
    /// Returns a receiver that will get all events emitted after subscription.
    /// If the receiver falls behind, it will miss events (lagged).
    pub fn subscribe(&self) -> EventSubscriber {
        EventSubscriber {
            receiver: self.sender.subscribe(),
        }
    }

    /// Emit an event to all subscribers
    ///
    /// This is non-blocking and will not fail. If there are no subscribers,
    /// the event is simply dropped.
    pub fn emit(&self, event: MemoryLifecycleEvent) {
        // send() returns the number of receivers, or error if none
        // We don't care about the result - fire and forget
        let _ = self.sender.send(event);
    }

    /// Get the number of active subscribers
    pub fn subscriber_count(&self) -> usize {
        self.sender.receiver_count()
    }
}

impl Default for EventBus {
    fn default() -> Self {
        Self::new()
    }
}

impl Clone for EventBus {
    fn clone(&self) -> Self {
        Self {
            sender: self.sender.clone(),
        }
    }
}

/// Subscriber to event bus
pub struct EventSubscriber {
    receiver: broadcast::Receiver<MemoryLifecycleEvent>,
}

impl EventSubscriber {
    /// Receive the next event
    ///
    /// Returns None if the bus was dropped.
    /// May return a Lagged error if this subscriber fell behind.
    pub async fn recv(&mut self) -> Option<MemoryLifecycleEvent> {
        loop {
            match self.receiver.recv().await {
                Ok(event) => return Some(event),
                Err(broadcast::error::RecvError::Closed) => return None,
                Err(broadcast::error::RecvError::Lagged(n)) => {
                    // Log the lag but continue receiving
                    tracing::warn!("Event subscriber lagged by {} events", n);
                    continue;
                }
            }
        }
    }

    /// Try to receive an event without blocking
    pub fn try_recv(&mut self) -> Option<MemoryLifecycleEvent> {
        match self.receiver.try_recv() {
            Ok(event) => Some(event),
            Err(_) => None,
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::events::{AccessType, MemoryAccessedEvent, MemoryCreatedEvent};

    #[tokio::test]
    async fn test_event_bus_basic() {
        let bus = EventBus::new();
        let mut sub = bus.subscribe();

        // Emit an event
        let event = MemoryLifecycleEvent::Created(MemoryCreatedEvent::new("mem-1", "Hello"));
        bus.emit(event);

        // Should receive it
        let received = sub.recv().await.unwrap();
        assert_eq!(received.memory_id(), "mem-1");
    }

    #[tokio::test]
    async fn test_multiple_subscribers() {
        let bus = EventBus::new();
        let mut sub1 = bus.subscribe();
        let mut sub2 = bus.subscribe();

        let event =
            MemoryLifecycleEvent::Accessed(MemoryAccessedEvent::new("mem-1", AccessType::Search));
        bus.emit(event);

        // Both should receive
        let r1 = sub1.recv().await.unwrap();
        let r2 = sub2.recv().await.unwrap();
        assert_eq!(r1.memory_id(), r2.memory_id());
    }

    #[tokio::test]
    async fn test_no_subscribers_no_panic() {
        let bus = EventBus::new();
        // Should not panic even with no subscribers
        let event = MemoryLifecycleEvent::Created(MemoryCreatedEvent::new("mem-1", "Test"));
        bus.emit(event);
    }

    #[test]
    fn test_subscriber_count() {
        let bus = EventBus::new();
        assert_eq!(bus.subscriber_count(), 0);

        let _sub1 = bus.subscribe();
        assert_eq!(bus.subscriber_count(), 1);

        let _sub2 = bus.subscribe();
        assert_eq!(bus.subscriber_count(), 2);
    }
}
