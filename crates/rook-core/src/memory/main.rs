//! Core Memory implementation.

use std::collections::HashMap;
use std::sync::{Arc, Mutex};
use tokio::sync::RwLock;
use uuid::Uuid;

use crate::config::MemoryConfig;
use crate::error::{RookError, RookResult};
use crate::events::{
    AccessType, EventBus, MemoryAccessedEvent, MemoryCreatedEvent, MemoryDeletedEvent,
    MemoryLifecycleEvent, MemoryUpdatedEvent, UpdateType,
};
use crate::ingestion::{
    IngestDecision, IngestResult, PredictionErrorGate, StrengthSignal, StrengthSignalProcessor,
};
use crate::traits::{
    Embedder, EmbeddingAction, GenerationOptions, GraphFilters, GraphStore, Llm, Reranker,
    ResponseFormat, VectorRecord, VectorSearchResult, VectorStore,
};
use crate::types::{
    AddResult, Filter, Grade, GraphRelation, MemoryEvent, MemoryItem, MemoryResult, MemoryType,
    Message, MessageInput, MessageRole, SearchResult,
};

use super::history::{HistoryEvent, HistoryStore};
use super::json_parser::{parse_facts, parse_memory_actions};
use super::prompts::{
    agent_memory_extraction_prompt, build_update_memory_message, classification_prompt,
    entity_extraction_prompt, find_entity_match, parse_classification, parse_entity_extraction,
    procedural_memory_prompt, user_memory_extraction_prompt, ClassificationResult, MergeConfig,
};
use super::session::SessionScope;
use super::telemetry::{process_telemetry_filters, Telemetry};

/// Main Memory struct - the core of rook.
pub struct Memory {
    config: MemoryConfig,
    llm: Arc<dyn Llm>,
    embedder: Arc<dyn Embedder>,
    vector_store: Arc<dyn VectorStore>,
    graph_store: Option<Arc<dyn GraphStore>>,
    reranker: Option<Arc<dyn Reranker>>,
    history: Arc<RwLock<HistoryStore>>,
    telemetry: Telemetry,
    prediction_error_gate: PredictionErrorGate,
    strength_processor: Mutex<StrengthSignalProcessor>,
    event_bus: Option<EventBus>,
}

impl Memory {
    /// Create a new Memory instance with the given configuration.
    ///
    /// Note: This method requires you to provide the provider implementations.
    /// Use the factory methods in rook-llm, rook-embeddings, and rook-vector-stores
    /// to create these.
    pub fn new(
        config: MemoryConfig,
        llm: Arc<dyn Llm>,
        embedder: Arc<dyn Embedder>,
        vector_store: Arc<dyn VectorStore>,
        graph_store: Option<Arc<dyn GraphStore>>,
        reranker: Option<Arc<dyn Reranker>>,
    ) -> RookResult<Self> {
        let history = Arc::new(RwLock::new(HistoryStore::new(&config.history_db_path)?));
        let telemetry = Telemetry::new(None);

        // Initialize prediction error gate with LLM for semantic layer
        let prediction_error_gate = PredictionErrorGate::new(Some(llm.clone()));
        let strength_processor = Mutex::new(StrengthSignalProcessor::new());

        Ok(Self {
            config,
            llm,
            embedder,
            vector_store,
            graph_store,
            reranker,
            history,
            telemetry,
            prediction_error_gate,
            strength_processor,
            event_bus: None,
        })
    }

    /// Set the event bus for emitting memory lifecycle events.
    ///
    /// When an EventBus is configured, Memory will emit:
    /// - MemoryCreatedEvent on add()
    /// - MemoryUpdatedEvent on update()
    /// - MemoryDeletedEvent on delete()
    /// - MemoryAccessedEvent on search() and get()
    pub fn with_event_bus(mut self, event_bus: EventBus) -> Self {
        self.event_bus = Some(event_bus);
        self
    }

    /// Add memories from messages.
    pub async fn add(
        &self,
        messages: impl Into<MessageInput>,
        user_id: Option<String>,
        agent_id: Option<String>,
        run_id: Option<String>,
        metadata: Option<HashMap<String, serde_json::Value>>,
        infer: bool,
        memory_type: Option<MemoryType>,
    ) -> RookResult<AddResult> {
        let messages = messages.into().normalize();
        let scope = SessionScope::new(user_id.clone(), agent_id.clone(), run_id.clone());
        scope.validate()?;

        let metadata = scope.to_metadata(metadata);
        let filters = scope.to_filters();

        // Handle procedural memory
        if agent_id.is_some() && memory_type == Some(MemoryType::ProceduralMemory) {
            return self.create_procedural_memory(&messages, &metadata).await;
        }

        // Add to vector store
        let vector_results = self
            .add_to_vector_store(&messages, &scope, &metadata, &filters, infer)
            .await?;

        // Add to graph store (if enabled)
        let graph_relations = if self.graph_store.is_some() {
            self.add_to_graph(&messages, &filters).await.ok()
        } else {
            None
        };

        // Telemetry
        let (keys, encoded_ids) = process_telemetry_filters(&filters);
        self.telemetry
            .capture_event(
                "rook.add",
                HashMap::from([
                    ("keys".to_string(), serde_json::to_value(&keys).unwrap()),
                    (
                        "encoded_ids".to_string(),
                        serde_json::to_value(&encoded_ids).unwrap(),
                    ),
                    ("infer".to_string(), serde_json::Value::Bool(infer)),
                ]),
            )
            .await;

        Ok(AddResult {
            results: vector_results,
            relations: graph_relations,
        })
    }

    /// Search for memories.
    ///
    /// If `key_memory.include_in_search` is enabled in config, key memories
    /// (is_key=true) are always included at the top of results, before
    /// similarity-ranked results. Duplicate key memories are deduplicated.
    pub async fn search(
        &self,
        query: &str,
        user_id: Option<String>,
        agent_id: Option<String>,
        run_id: Option<String>,
        limit: usize,
        filters: Option<HashMap<String, serde_json::Value>>,
        threshold: Option<f32>,
        rerank: bool,
    ) -> RookResult<SearchResult> {
        let scope = SessionScope::new(user_id.clone(), agent_id.clone(), run_id.clone());
        scope.validate()?;

        let mut effective_filters = scope.to_filters();
        if let Some(additional) = filters {
            effective_filters.extend(additional);
        }

        // Search vector store for similarity-ranked results
        let mut memories = self
            .search_vector_store(query, &effective_filters, limit, threshold)
            .await?;

        // Apply reranking if enabled
        if rerank {
            if let Some(ref reranker) = self.reranker {
                memories = reranker.rerank(query, memories, Some(limit)).await?;
            }
        }

        // Inject key memories at top if enabled
        if self.config.key_memory.include_in_search {
            let key_memories = self
                .get_key_memories(user_id.clone(), agent_id.clone(), run_id.clone())
                .await?;

            if !key_memories.is_empty() {
                memories = Self::merge_with_key_memories(key_memories, memories);
            }
        }

        // Search graph store (if enabled)
        let relations = if let Some(ref _graph) = self.graph_store {
            // TODO: Implement graph search
            None
        } else {
            None
        };

        // Emit accessed events for each memory in results
        if let Some(ref event_bus) = self.event_bus {
            for memory in &memories {
                let event = MemoryAccessedEvent::new(&memory.id, AccessType::Search)
                    .with_search_context(query, memory.score.unwrap_or(0.0));
                let event = if let Some(ref user_id) = user_id {
                    event.with_user(user_id)
                } else {
                    event
                };
                event_bus.emit(MemoryLifecycleEvent::Accessed(event));
            }
        }

        Ok(SearchResult {
            results: memories,
            relations,
        })
    }

    /// Get a specific memory by ID.
    pub async fn get(&self, memory_id: &str) -> RookResult<Option<MemoryItem>> {
        let record = self.vector_store.get(memory_id).await?;
        let result = record.map(|r| self.record_to_memory_item(r, None));

        // Emit accessed event if memory was found
        if let (Some(ref event_bus), Some(ref memory)) = (&self.event_bus, &result) {
            let event = MemoryAccessedEvent::new(&memory.id, AccessType::DirectGet);
            // Extract user_id from memory metadata if available
            let event = if let Some(user_id) = memory
                .metadata
                .as_ref()
                .and_then(|m| m.get("user_id"))
                .and_then(|v| v.as_str())
            {
                event.with_user(user_id)
            } else {
                event
            };
            event_bus.emit(MemoryLifecycleEvent::Accessed(event));
        }

        Ok(result)
    }

    /// Get all memories for a scope.
    pub async fn get_all(
        &self,
        user_id: Option<String>,
        agent_id: Option<String>,
        run_id: Option<String>,
        limit: Option<usize>,
    ) -> RookResult<Vec<MemoryItem>> {
        let scope = SessionScope::new(user_id, agent_id, run_id);
        scope.validate()?;

        let filters = scope.to_filters();
        let filter = self.build_filter(&filters);

        let records = self.vector_store.list(filter, limit).await?;

        Ok(records
            .into_iter()
            .map(|r| self.record_to_memory_item(r, None))
            .collect())
    }

    /// Update a memory.
    pub async fn update(&self, memory_id: &str, data: &str) -> RookResult<MemoryItem> {
        // Get existing memory
        let existing = self.vector_store.get(memory_id).await?.ok_or_else(|| {
            RookError::not_found(memory_id)
        })?;

        let prev_data = existing.get_data().map(|s| s.to_string());

        // Generate new embedding
        let embedding = self
            .embedder
            .embed(data, Some(EmbeddingAction::Update))
            .await?;

        // Update payload
        let mut payload = existing.payload.clone();
        payload.insert("data".to_string(), serde_json::Value::String(data.to_string()));
        payload.insert(
            "hash".to_string(),
            serde_json::Value::String(format!("{:x}", md5::compute(data.as_bytes()))),
        );
        let updated_at = chrono::Utc::now().to_rfc3339();
        payload.insert(
            "updated_at".to_string(),
            serde_json::Value::String(updated_at.clone()),
        );

        // Update in vector store
        self.vector_store
            .update(memory_id, Some(embedding), Some(payload.clone()))
            .await?;

        // Record history
        {
            let history = self.history.read().await;
            history.add(
                memory_id,
                prev_data.as_deref(),
                Some(data),
                HistoryEvent::Update,
                payload
                    .get("created_at")
                    .and_then(|v| v.as_str()),
                Some(&updated_at),
                payload.get("actor_id").and_then(|v| v.as_str()),
                payload.get("role").and_then(|v| v.as_str()),
            )?;
        }

        // Emit updated event
        if let Some(ref event_bus) = self.event_bus {
            let old_content = prev_data.clone().unwrap_or_default();
            let event = MemoryUpdatedEvent::new(
                memory_id,
                &old_content,
                data,
                UpdateType::Content,
                1, // Version tracking not yet integrated
            );
            let event = if let Some(user_id) = payload.get("user_id").and_then(|v| v.as_str()) {
                event.with_user(user_id)
            } else {
                event
            };
            event_bus.emit(MemoryLifecycleEvent::Updated(event));
        }

        Ok(MemoryItem {
            id: memory_id.to_string(),
            memory: data.to_string(),
            hash: Some(format!("{:x}", md5::compute(data.as_bytes()))),
            score: None,
            metadata: Some(payload.clone()),
            created_at: existing
                .payload
                .get("created_at")
                .and_then(|v| v.as_str())
                .map(|s| s.to_string()),
            updated_at: Some(updated_at),
            category: payload
                .get("category")
                .and_then(|v| v.as_str())
                .map(|s| s.to_string()),
            is_key: payload
                .get("is_key")
                .and_then(|v| v.as_bool())
                .unwrap_or(false),
            memory_state: None,
            dual_strength: None,
        })
    }

    /// Delete a memory.
    pub async fn delete(&self, memory_id: &str) -> RookResult<()> {
        // Get existing memory for history
        let existing = self.vector_store.get(memory_id).await?;
        let prev_data = existing.as_ref().and_then(|r| r.get_data().map(|s| s.to_string()));

        // Delete from vector store
        self.vector_store.delete(memory_id).await?;

        // Record history
        {
            let history = self.history.read().await;
            history.add(
                memory_id,
                prev_data.as_deref(),
                None,
                HistoryEvent::Delete,
                None,
                None,
                existing
                    .as_ref()
                    .and_then(|r| r.payload.get("actor_id"))
                    .and_then(|v| v.as_str()),
                existing
                    .as_ref()
                    .and_then(|r| r.payload.get("role"))
                    .and_then(|v| v.as_str()),
            )?;
        }

        // Emit deleted event
        if let Some(ref event_bus) = self.event_bus {
            let event = MemoryDeletedEvent::new(memory_id, false); // hard delete
            let event = if let Some(user_id) = existing
                .as_ref()
                .and_then(|r| r.payload.get("user_id"))
                .and_then(|v| v.as_str())
            {
                event.with_user(user_id)
            } else {
                event
            };
            event_bus.emit(MemoryLifecycleEvent::Deleted(event));
        }

        Ok(())
    }

    /// Delete all memories for a scope.
    pub async fn delete_all(
        &self,
        user_id: Option<String>,
        agent_id: Option<String>,
        run_id: Option<String>,
    ) -> RookResult<()> {
        let scope = SessionScope::new(user_id, agent_id, run_id);
        scope.validate()?;

        // Get all memories for this scope
        let memories = self
            .get_all(
                scope.user_id.clone(),
                scope.agent_id.clone(),
                scope.run_id.clone(),
                None,
            )
            .await?;

        // Delete each one
        for memory in memories {
            self.delete(&memory.id).await?;
        }

        Ok(())
    }

    /// Get history for a memory.
    pub async fn history(
        &self,
        memory_id: &str,
    ) -> RookResult<Vec<super::history::HistoryRecord>> {
        let history = self.history.read().await;
        history.get(memory_id)
    }

    /// Reset all memories.
    pub async fn reset(&self) -> RookResult<()> {
        self.vector_store.reset().await?;

        {
            let history = self.history.read().await;
            history.reset()?;
        }

        Ok(())
    }

    /// Intelligently ingest new content using prediction error gating.
    ///
    /// Unlike `add()` which always creates or updates memories based on LLM
    /// extraction, `smart_ingest()` uses multi-layer detection to decide:
    /// - Skip: Content is duplicate/redundant
    /// - Create: Content is novel
    /// - Update: Content refines existing memory
    /// - Supersede: Content contradicts existing memory
    ///
    /// Returns IngestResult with the decision, affected memory IDs,
    /// surprise value, and reasoning.
    pub async fn smart_ingest(
        &self,
        content: &str,
        user_id: Option<String>,
        agent_id: Option<String>,
        run_id: Option<String>,
        metadata: Option<HashMap<String, serde_json::Value>>,
    ) -> RookResult<IngestResult> {
        let scope = SessionScope::new(user_id.clone(), agent_id.clone(), run_id.clone());
        scope.validate()?;

        let filters = scope.to_filters();
        let filter = self.build_filter(&filters);

        // Get existing memories for comparison (scoped to user/agent)
        let existing_memories = self.vector_store.list(filter, None).await?;

        // Run prediction error gating
        let gate_result = self
            .prediction_error_gate
            .evaluate(content, &existing_memories, self.embedder.as_ref())
            .await?;

        // Execute decision and build result
        let result = match gate_result.decision {
            IngestDecision::Skip => IngestResult {
                decision: IngestDecision::Skip,
                memory_id: None,
                previous_content: None,
                related_memory_id: gate_result.related_memory_id,
                surprise: gate_result.surprise,
                decided_at_layer: gate_result.layer,
                reason: gate_result.reason,
            },
            IngestDecision::Create => {
                let metadata = scope.to_metadata(metadata);
                let memory_id = self.create_memory(content, &metadata).await?;

                IngestResult {
                    decision: IngestDecision::Create,
                    memory_id: Some(memory_id),
                    previous_content: None,
                    related_memory_id: None,
                    surprise: gate_result.surprise,
                    decided_at_layer: gate_result.layer,
                    reason: gate_result.reason,
                }
            }
            IngestDecision::Update => {
                let related_id = gate_result.related_memory_id.ok_or_else(|| {
                    RookError::internal("Update decision requires related memory ID")
                })?;

                // Get previous content
                let previous = self.get(&related_id).await?.map(|m| m.memory);

                // Update the memory
                self.update(&related_id, content).await?;

                IngestResult {
                    decision: IngestDecision::Update,
                    memory_id: Some(related_id.clone()),
                    previous_content: previous,
                    related_memory_id: Some(related_id),
                    surprise: gate_result.surprise,
                    decided_at_layer: gate_result.layer,
                    reason: gate_result.reason,
                }
            }
            IngestDecision::Supersede => {
                let superseded_id = gate_result.related_memory_id.ok_or_else(|| {
                    RookError::internal("Supersede decision requires related memory ID")
                })?;

                // Get previous content
                let previous = self.get(&superseded_id).await?.map(|m| m.memory);

                // Create new memory with higher initial strength due to surprise
                let metadata = scope.to_metadata(metadata);
                let new_memory_id = self.create_memory(content, &metadata).await?;

                // Mark old memory as superseded (update metadata)
                if let Some(mut record) = self.vector_store.get(&superseded_id).await? {
                    record.payload.insert(
                        "superseded_by".to_string(),
                        serde_json::Value::String(new_memory_id.clone()),
                    );
                    record.payload.insert(
                        "superseded_at".to_string(),
                        serde_json::Value::String(chrono::Utc::now().to_rfc3339()),
                    );
                    self.vector_store
                        .update(&superseded_id, None, Some(record.payload))
                        .await?;
                }

                // Process contradiction strength signal
                {
                    let mut processor = self.strength_processor.lock().unwrap();
                    processor.process(StrengthSignal::Contradiction {
                        winner_id: new_memory_id.clone(),
                        loser_id: superseded_id.clone(),
                    });
                }

                IngestResult {
                    decision: IngestDecision::Supersede,
                    memory_id: Some(new_memory_id),
                    previous_content: previous,
                    related_memory_id: Some(superseded_id),
                    surprise: gate_result.surprise,
                    decided_at_layer: gate_result.layer,
                    reason: gate_result.reason,
                }
            }
        };

        // Telemetry
        let (keys, encoded_ids) = process_telemetry_filters(&filters);
        self.telemetry
            .capture_event(
                "rook.smart_ingest",
                HashMap::from([
                    ("keys".to_string(), serde_json::to_value(&keys).unwrap()),
                    (
                        "encoded_ids".to_string(),
                        serde_json::to_value(&encoded_ids).unwrap(),
                    ),
                    (
                        "decision".to_string(),
                        serde_json::to_value(&result.decision).unwrap(),
                    ),
                    (
                        "layer".to_string(),
                        serde_json::to_value(format!("{:?}", result.decided_at_layer)).unwrap(),
                    ),
                    (
                        "surprise".to_string(),
                        serde_json::Value::Number(
                            serde_json::Number::from_f64(result.surprise as f64).unwrap(),
                        ),
                    ),
                ]),
            )
            .await;

        Ok(result)
    }

    /// Get pending strength signal updates
    ///
    /// Returns the updates collected since last clear. Used by
    /// external FSRS integration to apply grade updates.
    pub fn get_pending_strength_updates(&self) -> HashMap<String, Grade> {
        let processor = self.strength_processor.lock().unwrap();
        processor.get_pending_updates()
    }

    /// Clear pending strength signal updates
    pub fn clear_strength_updates(&self) {
        let mut processor = self.strength_processor.lock().unwrap();
        processor.clear();
    }

    /// Process a strength signal
    pub fn process_strength_signal(&self, signal: StrengthSignal) {
        let mut processor = self.strength_processor.lock().unwrap();
        processor.process(signal);
    }

    /// Get key memories for a scope.
    ///
    /// Key memories are high-importance memories marked with `is_key=true`.
    /// They are exempt from decay and archival, and are always included in
    /// search results (up to `max_key_memories` config limit).
    pub async fn get_key_memories(
        &self,
        user_id: Option<String>,
        agent_id: Option<String>,
        run_id: Option<String>,
    ) -> RookResult<Vec<MemoryItem>> {
        let scope = SessionScope::new(user_id, agent_id, run_id);
        scope.validate()?;

        let mut filters = scope.to_filters();
        // Add is_key=true filter
        filters.insert("is_key".to_string(), serde_json::Value::Bool(true));

        let filter = self.build_filter(&filters);
        let limit = Some(self.config.key_memory.max_key_memories);

        let records = self.vector_store.list(filter, limit).await?;

        Ok(records
            .into_iter()
            .map(|r| {
                let mut item = self.record_to_memory_item(r, None);
                item.is_key = true;
                item
            })
            .collect())
    }

    /// Merge key memories with search results.
    ///
    /// Key memories are prepended to search results, with deduplication
    /// to ensure a memory doesn't appear twice (once as key, once from search).
    fn merge_with_key_memories(
        key_memories: Vec<MemoryItem>,
        search_results: Vec<MemoryItem>,
    ) -> Vec<MemoryItem> {
        // Collect IDs of key memories for deduplication
        let key_ids: std::collections::HashSet<_> =
            key_memories.iter().map(|m| m.id.clone()).collect();

        // Filter search results to remove any that are already in key memories
        let deduplicated_results: Vec<MemoryItem> = search_results
            .into_iter()
            .filter(|m| !key_ids.contains(&m.id))
            .collect();

        // Prepend key memories to search results
        let mut merged = key_memories;
        merged.extend(deduplicated_results);
        merged
    }

    /// Classify a memory using LLM.
    ///
    /// Uses the category configuration to generate a prompt with valid categories,
    /// calls the LLM to classify the memory, and returns a ClassificationResult
    /// with the category, is_key flag, and confidence score.
    ///
    /// If the LLM returns an invalid category, it falls back to "misc".
    async fn classify_memory(&self, content: &str) -> RookResult<ClassificationResult> {
        let valid_categories: Vec<String> = self
            .config
            .category
            .valid_categories()
            .into_iter()
            .collect();

        let prompt = classification_prompt(&valid_categories);
        let messages = vec![Message::system(prompt), Message::user(content)];

        let response = self.llm.generate(&messages, None).await?;
        let result = parse_classification(response.content_or_empty(), &valid_categories);

        Ok(result)
    }

    // Private helper methods

    async fn add_to_vector_store(
        &self,
        messages: &[Message],
        scope: &SessionScope,
        metadata: &HashMap<String, serde_json::Value>,
        filters: &HashMap<String, serde_json::Value>,
        infer: bool,
    ) -> RookResult<Vec<MemoryResult>> {
        if !infer {
            // Without inference, just store raw messages
            return self.store_raw_messages(messages, metadata).await;
        }

        // Extract facts using LLM
        let new_facts = self.extract_facts(messages, scope).await?;
        if new_facts.is_empty() {
            return Ok(vec![]);
        }

        // Get existing memories for comparison
        let existing_memories = self.get_existing_memories(filters, &new_facts).await?;

        // Get LLM decision on memory actions
        let actions = self
            .get_memory_update_actions(&existing_memories, &new_facts)
            .await?;

        // Execute memory actions
        let mut results = Vec::new();
        for action in actions {
            if action.text.is_empty() {
                continue;
            }

            match action.event {
                MemoryEvent::Add => {
                    let memory_id = self.create_memory(&action.text, metadata).await?;
                    results.push(MemoryResult {
                        id: memory_id,
                        memory: action.text,
                        event: MemoryEvent::Add,
                        previous_memory: None,
                    });
                }
                MemoryEvent::Update => {
                    // Map the index back to real ID
                    if let Some((real_id, _)) = existing_memories
                        .iter()
                        .find(|(id, _)| *id == action.id || id.ends_with(&action.id))
                    {
                        self.update(real_id, &action.text).await?;
                        results.push(MemoryResult {
                            id: real_id.clone(),
                            memory: action.text,
                            event: MemoryEvent::Update,
                            previous_memory: action.old_memory,
                        });
                    }
                }
                MemoryEvent::Delete => {
                    if let Some((real_id, _)) = existing_memories
                        .iter()
                        .find(|(id, _)| *id == action.id || id.ends_with(&action.id))
                    {
                        self.delete(real_id).await?;
                        results.push(MemoryResult {
                            id: real_id.clone(),
                            memory: action.text,
                            event: MemoryEvent::Delete,
                            previous_memory: None,
                        });
                    }
                }
                MemoryEvent::None => {
                    // No action needed
                }
            }
        }

        Ok(results)
    }

    async fn extract_facts(
        &self,
        messages: &[Message],
        scope: &SessionScope,
    ) -> RookResult<Vec<String>> {
        let mut all_facts = Vec::new();

        // Extract user facts
        let user_messages: Vec<_> = messages
            .iter()
            .filter(|m| matches!(m.role, MessageRole::User))
            .cloned()
            .collect();

        if !user_messages.is_empty() {
            let user_facts = self.extract_user_facts(&user_messages).await?;
            all_facts.extend(user_facts);
        }

        // Extract agent facts if agent_id is present
        if scope.should_use_agent_extraction(messages) {
            let agent_messages: Vec<_> = messages
                .iter()
                .filter(|m| matches!(m.role, MessageRole::Assistant))
                .cloned()
                .collect();

            if !agent_messages.is_empty() {
                let agent_facts = self.extract_agent_facts(&agent_messages).await?;
                all_facts.extend(agent_facts);
            }
        }

        Ok(all_facts)
    }

    async fn extract_user_facts(&self, messages: &[Message]) -> RookResult<Vec<String>> {
        let prompt = self
            .config
            .custom_fact_extraction_prompt
            .clone()
            .unwrap_or_else(user_memory_extraction_prompt);

        let formatted_messages = messages
            .iter()
            .map(|m| format!("User: {}", m.content))
            .collect::<Vec<_>>()
            .join("\n");

        let llm_messages = vec![
            Message::system(prompt),
            Message::user(formatted_messages),
        ];

        let response = self.llm.generate(&llm_messages, None).await?;
        parse_facts(response.content_or_empty())
    }

    async fn extract_agent_facts(&self, messages: &[Message]) -> RookResult<Vec<String>> {
        let prompt = agent_memory_extraction_prompt();

        let formatted_messages = messages
            .iter()
            .map(|m| format!("Assistant: {}", m.content))
            .collect::<Vec<_>>()
            .join("\n");

        let llm_messages = vec![
            Message::system(prompt),
            Message::user(formatted_messages),
        ];

        let response = self.llm.generate(&llm_messages, None).await?;
        parse_facts(response.content_or_empty())
    }

    async fn get_existing_memories(
        &self,
        filters: &HashMap<String, serde_json::Value>,
        new_facts: &[String],
    ) -> RookResult<Vec<(String, String)>> {
        let mut existing = HashMap::new();
        let filter = self.build_filter(filters);

        for fact in new_facts {
            let embedding = self.embedder.embed(fact, Some(EmbeddingAction::Search)).await?;
            let results = self.vector_store.search(&embedding, 5, filter.clone()).await?;

            for result in results {
                if let Some(data) = result.payload.get("data").and_then(|v| v.as_str()) {
                    existing.insert(result.id.clone(), data.to_string());
                }
            }
        }

        Ok(existing.into_iter().collect())
    }

    async fn get_memory_update_actions(
        &self,
        existing_memories: &[(String, String)],
        new_facts: &[String],
    ) -> RookResult<Vec<super::json_parser::MemoryAction>> {
        // Create indexed memories (using indices instead of real IDs to prevent hallucination)
        let indexed_memories: Vec<(String, String)> = existing_memories
            .iter()
            .enumerate()
            .map(|(idx, (_, data))| (idx.to_string(), data.clone()))
            .collect();

        let prompt = build_update_memory_message(
            &indexed_memories,
            new_facts,
            self.config.custom_update_memory_prompt.as_deref(),
        );

        let llm_messages = vec![Message::user(prompt)];
        let response = self.llm.generate(&llm_messages, None).await?;

        let mut actions = parse_memory_actions(response.content_or_empty())?;

        // Map indices back to real IDs
        for action in &mut actions {
            if let Ok(idx) = action.id.parse::<usize>() {
                if let Some((real_id, _)) = existing_memories.get(idx) {
                    action.id = real_id.clone();
                }
            }
        }

        Ok(actions)
    }

    async fn create_memory(
        &self,
        data: &str,
        metadata: &HashMap<String, serde_json::Value>,
    ) -> RookResult<String> {
        let embedding = self.embedder.embed(data, Some(EmbeddingAction::Add)).await?;
        let memory_id = Uuid::new_v4().to_string();
        let hash = format!("{:x}", md5::compute(data.as_bytes()));
        let created_at = chrono::Utc::now().to_rfc3339();

        // Classify the memory using LLM
        let classification = self.classify_memory(data).await?;

        let mut payload = metadata.clone();
        payload.insert("data".to_string(), serde_json::Value::String(data.to_string()));
        payload.insert("hash".to_string(), serde_json::Value::String(hash));
        payload.insert(
            "created_at".to_string(),
            serde_json::Value::String(created_at.clone()),
        );

        // Store classification results in payload
        payload.insert(
            "category".to_string(),
            serde_json::Value::String(classification.category),
        );
        payload.insert("is_key".to_string(), serde_json::Value::Bool(classification.is_key));
        payload.insert(
            "classification_confidence".to_string(),
            serde_json::json!(classification.confidence),
        );

        let record = VectorRecord::new(memory_id.clone(), embedding, payload.clone());
        self.vector_store.insert(vec![record]).await?;

        // Emit created event
        if let Some(ref event_bus) = self.event_bus {
            let event = MemoryCreatedEvent::new(&memory_id, data).with_metadata(payload.clone());
            let event = if let Some(user_id) = metadata.get("user_id").and_then(|v| v.as_str()) {
                event.with_user(user_id)
            } else {
                event
            };
            event_bus.emit(MemoryLifecycleEvent::Created(event));
        }

        // Record history
        {
            let history = self.history.read().await;
            history.add(
                &memory_id,
                None,
                Some(data),
                HistoryEvent::Add,
                Some(&created_at),
                None,
                metadata.get("actor_id").and_then(|v| v.as_str()),
                metadata.get("role").and_then(|v| v.as_str()),
            )?;
        }

        Ok(memory_id)
    }

    async fn store_raw_messages(
        &self,
        messages: &[Message],
        metadata: &HashMap<String, serde_json::Value>,
    ) -> RookResult<Vec<MemoryResult>> {
        let mut results = Vec::new();

        for message in messages {
            let memory_id = self.create_memory(&message.content, metadata).await?;
            results.push(MemoryResult {
                id: memory_id,
                memory: message.content.clone(),
                event: MemoryEvent::Add,
                previous_memory: None,
            });
        }

        Ok(results)
    }

    async fn search_vector_store(
        &self,
        query: &str,
        filters: &HashMap<String, serde_json::Value>,
        limit: usize,
        threshold: Option<f32>,
    ) -> RookResult<Vec<MemoryItem>> {
        let embedding = self
            .embedder
            .embed(query, Some(EmbeddingAction::Search))
            .await?;

        let filter = self.build_filter(filters);
        let results = self.vector_store.search(&embedding, limit, filter).await?;

        let memories: Vec<MemoryItem> = results
            .into_iter()
            .filter(|r| threshold.map_or(true, |t| r.score >= t))
            .map(|r| self.search_result_to_memory_item(r))
            .collect();

        Ok(memories)
    }

    /// Add entities and relationships to the graph store.
    ///
    /// This method:
    /// 1. Extracts entities and relationships from message content using LLM
    /// 2. Checks for existing similar entities using embedding-based merging (threshold 0.85)
    /// 3. Stores new entities with their embeddings
    /// 4. Creates relationships between entities
    /// 5. Returns GraphRelation objects for the created relationships
    async fn add_to_graph(
        &self,
        messages: &[Message],
        filters: &HashMap<String, serde_json::Value>,
    ) -> RookResult<Vec<GraphRelation>> {
        // Get graph store - return empty if not configured
        let graph_store = match &self.graph_store {
            Some(store) => store,
            None => return Ok(vec![]),
        };

        // Concatenate all message content for extraction
        let text: String = messages
            .iter()
            .map(|m| m.content.as_str())
            .collect::<Vec<_>>()
            .join("\n");

        if text.trim().is_empty() {
            return Ok(vec![]);
        }

        // Build graph filters from the filters HashMap
        let graph_filters = GraphFilters {
            user_id: filters.get("user_id").and_then(|v| v.as_str()).map(|s| s.to_string()),
            agent_id: filters.get("agent_id").and_then(|v| v.as_str()).map(|s| s.to_string()),
            run_id: filters.get("run_id").and_then(|v| v.as_str()).map(|s| s.to_string()),
        };

        // Extract entities and relationships using LLM
        let extraction_result = match self.extract_entities(&text).await {
            Ok(result) => result,
            Err(e) => {
                tracing::warn!("Entity extraction failed: {}", e);
                return Ok(vec![]);
            }
        };

        if extraction_result.is_empty() {
            tracing::debug!("No entities extracted from text");
            return Ok(vec![]);
        }

        tracing::debug!(
            "Extracted {} entities and {} relationships",
            extraction_result.entities.len(),
            extraction_result.relationships.len()
        );

        // Get existing entities for merge checking
        let existing_entities = graph_store.get_entities_for_merge(&graph_filters).await?;

        // Merge config with 0.85 threshold
        let merge_config = MergeConfig::default();

        // Track entity name -> ID mappings for relationship creation
        let mut entity_ids: HashMap<String, i64> = HashMap::new();

        // Process each extracted entity
        for entity in &extraction_result.entities {
            // Generate embedding for the entity name
            let embedding = match self
                .embedder
                .embed(&entity.name, Some(EmbeddingAction::Add))
                .await
            {
                Ok(emb) => emb,
                Err(e) => {
                    tracing::warn!("Failed to embed entity '{}': {}", entity.name, e);
                    continue;
                }
            };

            // Check for merge with existing entities
            let merge_result = find_entity_match(entity, &embedding, &existing_entities, &merge_config);

            let entity_id = if merge_result.matched {
                // Use existing entity ID
                let existing_id = merge_result.matched_entity_id.unwrap();
                tracing::debug!(
                    "Merged entity '{}' with existing '{}' (similarity: {:.3})",
                    entity.name,
                    merge_result.matched_entity_name.as_deref().unwrap_or("unknown"),
                    merge_result.similarity.unwrap_or(0.0)
                );
                existing_id
            } else {
                // Create new entity with embedding stored in properties
                let properties = serde_json::json!({
                    "description": entity.description,
                    "embedding": embedding,
                });

                match graph_store
                    .add_entity(
                        &entity.name,
                        entity.entity_type.as_str(),
                        &properties,
                        &graph_filters,
                    )
                    .await
                {
                    Ok(id) => {
                        tracing::debug!(
                            "Created entity '{}' (type: {}, id: {})",
                            entity.name,
                            entity.entity_type.as_str(),
                            id
                        );
                        id
                    }
                    Err(e) => {
                        tracing::warn!("Failed to add entity '{}': {}", entity.name, e);
                        continue;
                    }
                }
            };

            entity_ids.insert(entity.name.clone(), entity_id);
        }

        // Create relationships and build return value
        let mut relations: Vec<GraphRelation> = Vec::new();

        for rel in &extraction_result.relationships {
            let properties = serde_json::json!({
                "context": rel.context,
            });

            match graph_store
                .add_relationship(
                    &rel.source,
                    &rel.target,
                    rel.relationship_type.as_str(),
                    &properties,
                    &graph_filters,
                )
                .await
            {
                Ok(_) => {
                    tracing::debug!(
                        "Created relationship: {} --[{}]--> {}",
                        rel.source,
                        rel.relationship_type.as_str(),
                        rel.target
                    );
                    relations.push(GraphRelation {
                        source: rel.source.clone(),
                        relationship: rel.relationship_type.as_str().to_string(),
                        target: rel.target.clone(),
                    });
                }
                Err(e) => {
                    tracing::warn!(
                        "Failed to add relationship {} -> {}: {}",
                        rel.source,
                        rel.target,
                        e
                    );
                }
            }
        }

        Ok(relations)
    }

    /// Extract entities and relationships from text using LLM.
    async fn extract_entities(
        &self,
        text: &str,
    ) -> RookResult<super::prompts::ExtractionResult> {
        let prompt = entity_extraction_prompt();
        let messages = vec![
            Message::system(prompt),
            Message::user(format!("Extract entities and relationships from this text:\n\n{}", text)),
        ];

        // Request JSON response with deterministic temperature
        let options = GenerationOptions {
            temperature: Some(0.0),
            response_format: Some(ResponseFormat::Json),
            ..Default::default()
        };

        let response = self.llm.generate(&messages, Some(options)).await?;
        let content = response.content.unwrap_or_default();

        Ok(parse_entity_extraction(&content))
    }

    async fn create_procedural_memory(
        &self,
        messages: &[Message],
        metadata: &HashMap<String, serde_json::Value>,
    ) -> RookResult<AddResult> {
        let mut prompt_messages = vec![Message::system(procedural_memory_prompt())];
        prompt_messages.extend(messages.iter().cloned());
        prompt_messages.push(Message::user(
            "Create procedural memory of the above conversation.",
        ));

        let response = self.llm.generate(&prompt_messages, None).await?;
        let procedural_memory =
            super::json_parser::remove_code_blocks(response.content_or_empty());

        let mut proc_metadata = metadata.clone();
        proc_metadata.insert(
            "memory_type".to_string(),
            serde_json::Value::String("procedural_memory".to_string()),
        );

        let memory_id = self.create_memory(&procedural_memory, &proc_metadata).await?;

        Ok(AddResult {
            results: vec![MemoryResult {
                id: memory_id,
                memory: procedural_memory,
                event: MemoryEvent::Add,
                previous_memory: None,
            }],
            relations: None,
        })
    }

    fn build_filter(&self, filters: &HashMap<String, serde_json::Value>) -> Option<Filter> {
        if filters.is_empty() {
            return None;
        }

        let conditions: Vec<Filter> = filters
            .iter()
            .map(|(k, v)| Filter::eq(k.clone(), v.clone()))
            .collect();

        if conditions.len() == 1 {
            Some(conditions.into_iter().next().unwrap())
        } else {
            Some(Filter::And(conditions))
        }
    }

    fn record_to_memory_item(&self, record: VectorRecord, score: Option<f32>) -> MemoryItem {
        MemoryItem {
            id: record.id,
            memory: record
                .payload
                .get("data")
                .and_then(|v| v.as_str())
                .unwrap_or("")
                .to_string(),
            hash: record
                .payload
                .get("hash")
                .and_then(|v| v.as_str())
                .map(|s| s.to_string()),
            score: score.or(record.score),
            metadata: Some(record.payload.clone()),
            created_at: record
                .payload
                .get("created_at")
                .and_then(|v| v.as_str())
                .map(|s| s.to_string()),
            updated_at: record
                .payload
                .get("updated_at")
                .and_then(|v| v.as_str())
                .map(|s| s.to_string()),
            category: record
                .payload
                .get("category")
                .and_then(|v| v.as_str())
                .map(|s| s.to_string()),
            is_key: record
                .payload
                .get("is_key")
                .and_then(|v| v.as_bool())
                .unwrap_or(false),
            memory_state: None,
            dual_strength: None,
        }
    }

    fn search_result_to_memory_item(&self, result: VectorSearchResult) -> MemoryItem {
        MemoryItem {
            id: result.id,
            memory: result
                .payload
                .get("data")
                .and_then(|v| v.as_str())
                .unwrap_or("")
                .to_string(),
            hash: result
                .payload
                .get("hash")
                .and_then(|v| v.as_str())
                .map(|s| s.to_string()),
            score: Some(result.score),
            metadata: Some(result.payload.clone()),
            created_at: result
                .payload
                .get("created_at")
                .and_then(|v| v.as_str())
                .map(|s| s.to_string()),
            updated_at: result
                .payload
                .get("updated_at")
                .and_then(|v| v.as_str())
                .map(|s| s.to_string()),
            category: result
                .payload
                .get("category")
                .and_then(|v| v.as_str())
                .map(|s| s.to_string()),
            is_key: result
                .payload
                .get("is_key")
                .and_then(|v| v.as_bool())
                .unwrap_or(false),
            memory_state: None,
            dual_strength: None,
        }
    }

    /// Ingest multimodal content (PDF, DOCX, images) as memories.
    ///
    /// This is a convenience method that extracts text from the content,
    /// chunks if necessary, and stores as memories with provenance metadata.
    ///
    /// # Cross-Modal Retrieval
    ///
    /// When using multimodal ingestion, memories from different modalities
    /// (PDF, DOCX, images) are all stored as text with metadata tracking
    /// their source. This means:
    ///
    /// - Queries find relevant content regardless of original format
    /// - Results include `source_modality` in metadata for format awareness
    /// - No special handling needed - just search normally
    ///
    /// # Example
    ///
    /// ```ignore
    /// let pdf_bytes = std::fs::read("document.pdf")?;
    /// let result = memory.ingest_content(
    ///     &pdf_bytes,
    ///     "application/pdf",
    ///     Some("document.pdf"),
    ///     "user_123",
    ///     None,
    /// ).await?;
    ///
    /// println!("Created {} memories from PDF", result.memory_ids.len());
    ///
    /// // Search finds content from any modality
    /// let results = memory.search("meeting notes", Some("user_123".to_string()), None, None, 10, None, None, true).await?;
    ///
    /// // Check original modality in results
    /// for mem in &results.results {
    ///     if let Some(ref metadata) = mem.metadata {
    ///         if let Some(modality) = metadata.get("source_modality") {
    ///             println!("Found in {}: {}", modality, mem.memory);
    ///         }
    ///     }
    /// }
    /// ```
    #[cfg(feature = "multimodal")]
    pub async fn ingest_content(
        &self,
        content: &[u8],
        mime_type: &str,
        filename: Option<&str>,
        user_id: &str,
        additional_metadata: Option<std::collections::HashMap<String, serde_json::Value>>,
    ) -> crate::error::RookResult<crate::multimodal::MultimodalIngestResult> {
        let ingester = crate::multimodal::MultimodalIngester::new();
        ingester
            .ingest(self, content, mime_type, filename, user_id, additional_metadata)
            .await
    }

    /// Ingest multimodal content with custom configuration.
    ///
    /// Allows customizing chunk size, overlap, and page splitting behavior.
    ///
    /// # Example
    ///
    /// ```ignore
    /// use rook_core::MultimodalConfig;
    ///
    /// // Use page-level splitting for PDFs
    /// let config = MultimodalConfig::with_page_splitting();
    ///
    /// let result = memory.ingest_content_with_config(
    ///     &pdf_bytes,
    ///     "application/pdf",
    ///     Some("large_document.pdf"),
    ///     "user_123",
    ///     None,
    ///     config,
    /// ).await?;
    /// ```
    #[cfg(feature = "multimodal")]
    pub async fn ingest_content_with_config(
        &self,
        content: &[u8],
        mime_type: &str,
        filename: Option<&str>,
        user_id: &str,
        additional_metadata: Option<std::collections::HashMap<String, serde_json::Value>>,
        config: crate::multimodal::MultimodalConfig,
    ) -> crate::error::RookResult<crate::multimodal::MultimodalIngestResult> {
        let ingester = crate::multimodal::MultimodalIngester::with_config(config);
        ingester
            .ingest(self, content, mime_type, filename, user_id, additional_metadata)
            .await
    }
}

#[cfg(test)]
mod event_wiring_tests {
    use super::*;
    use crate::events::{EventBus, MemoryLifecycleEvent};

    /// Test that EventBus field exists and with_event_bus builder works
    #[test]
    fn test_event_bus_field_and_builder() {
        // This is a compile-time verification that the types work together
        // The actual runtime test requires full Memory construction which needs mocks

        let bus = EventBus::new();
        let subscriber = bus.subscribe();

        // Verify EventBus can be cloned (required for builder pattern)
        let _bus_clone = bus.clone();

        // Verify subscriber count works
        assert_eq!(bus.subscriber_count(), 1);

        drop(subscriber);
        assert_eq!(bus.subscriber_count(), 0);
    }

    /// Test that event types are correctly constructed
    #[test]
    fn test_event_construction() {
        use crate::events::{
            AccessType, MemoryAccessedEvent, MemoryCreatedEvent, MemoryDeletedEvent,
            MemoryUpdatedEvent, UpdateType,
        };

        // MemoryCreatedEvent
        let created = MemoryCreatedEvent::new("mem-1", "test content")
            .with_user("user-1")
            .with_metadata(HashMap::from([(
                "key".to_string(),
                serde_json::json!("value"),
            )]));
        assert_eq!(created.memory_id, "mem-1");
        assert_eq!(created.content, "test content");
        assert_eq!(created.user_id, Some("user-1".to_string()));

        // MemoryUpdatedEvent
        let updated =
            MemoryUpdatedEvent::new("mem-1", "old", "new", UpdateType::Content, 1).with_user("user-1");
        assert_eq!(updated.old_content, "old");
        assert_eq!(updated.new_content, "new");
        assert_eq!(updated.update_type, UpdateType::Content);

        // MemoryDeletedEvent
        let deleted = MemoryDeletedEvent::new("mem-1", false)
            .with_user("user-1")
            .with_reason("test deletion");
        assert!(!deleted.soft_delete);
        assert_eq!(deleted.reason, Some("test deletion".to_string()));

        // MemoryAccessedEvent - DirectGet
        let accessed_get = MemoryAccessedEvent::new("mem-1", AccessType::DirectGet).with_user("user-1");
        assert_eq!(accessed_get.access_type, AccessType::DirectGet);
        assert!(accessed_get.query.is_none());

        // MemoryAccessedEvent - Search
        let accessed_search = MemoryAccessedEvent::new("mem-1", AccessType::Search)
            .with_user("user-1")
            .with_search_context("test query", 0.95);
        assert_eq!(accessed_search.access_type, AccessType::Search);
        assert_eq!(accessed_search.query, Some("test query".to_string()));
        assert_eq!(accessed_search.relevance_score, Some(0.95));
    }

    /// Test event emission through EventBus
    #[tokio::test]
    async fn test_event_emission_through_bus() {
        use crate::events::MemoryCreatedEvent;

        let bus = EventBus::new();
        let mut subscriber = bus.subscribe();

        // Emit a created event
        let event = MemoryCreatedEvent::new("mem-1", "test content").with_user("user-1");
        bus.emit(MemoryLifecycleEvent::Created(event));

        // Verify it's received
        let received = subscriber.try_recv();
        assert!(received.is_some(), "Should receive emitted event");

        if let Some(MemoryLifecycleEvent::Created(e)) = received {
            assert_eq!(e.memory_id, "mem-1");
            assert_eq!(e.content, "test content");
            assert_eq!(e.user_id, Some("user-1".to_string()));
        } else {
            panic!("Expected MemoryCreatedEvent");
        }
    }
}

/// Tests for add_to_graph entity extraction flow
#[cfg(test)]
mod add_to_graph_tests {
    use super::*;
    use crate::memory::{
        parse_entity_extraction, EntityType, ExtractedEntity, ExtractedRelationship,
        RelationshipType,
    };
    use crate::traits::EntityWithEmbedding;

    #[test]
    fn test_graph_filters_from_hashmap() {
        // Verify GraphFilters can be built from HashMap (used in add_to_graph)
        let mut filters = HashMap::new();
        filters.insert("user_id".to_string(), serde_json::json!("user-123"));
        filters.insert("agent_id".to_string(), serde_json::json!("agent-456"));
        filters.insert("run_id".to_string(), serde_json::json!("run-789"));

        let graph_filters = GraphFilters {
            user_id: filters.get("user_id").and_then(|v| v.as_str()).map(|s| s.to_string()),
            agent_id: filters.get("agent_id").and_then(|v| v.as_str()).map(|s| s.to_string()),
            run_id: filters.get("run_id").and_then(|v| v.as_str()).map(|s| s.to_string()),
        };

        assert_eq!(graph_filters.user_id, Some("user-123".to_string()));
        assert_eq!(graph_filters.agent_id, Some("agent-456".to_string()));
        assert_eq!(graph_filters.run_id, Some("run-789".to_string()));
    }

    #[test]
    fn test_graph_relation_from_extracted_relationship() {
        // Verify GraphRelation can be built from ExtractedRelationship
        let rel = ExtractedRelationship::new("Alice", "Acme Corp", RelationshipType::WorksAt);

        let graph_relation = GraphRelation {
            source: rel.source.clone(),
            relationship: rel.relationship_type.as_str().to_string(),
            target: rel.target.clone(),
        };

        assert_eq!(graph_relation.source, "Alice");
        assert_eq!(graph_relation.relationship, "works_at");
        assert_eq!(graph_relation.target, "Acme Corp");
    }

    #[test]
    fn test_entity_properties_for_storage() {
        // Verify entity properties JSON structure for storage
        let entity = ExtractedEntity::new("Alice", EntityType::Person)
            .with_description("Software engineer");
        let embedding = vec![1.0, 0.5, 0.0];

        let properties = serde_json::json!({
            "description": entity.description,
            "embedding": embedding,
        });

        // Verify structure is valid JSON
        assert!(properties.is_object());
        assert_eq!(
            properties["description"].as_str(),
            Some("Software engineer")
        );
        assert_eq!(properties["embedding"].as_array().unwrap().len(), 3);
    }

    #[test]
    fn test_extraction_result_entity_iteration() {
        // Test that we can iterate over extracted entities and relationships
        // as done in add_to_graph
        let json = r#"{
            "entities": [
                {"name": "Alice", "entity_type": "person"},
                {"name": "Bob", "entity_type": "person"},
                {"name": "Acme Corp", "entity_type": "organization"}
            ],
            "relationships": [
                {"source": "Alice", "target": "Acme Corp", "relationship_type": "works_at"},
                {"source": "Alice", "target": "Bob", "relationship_type": "knows"}
            ]
        }"#;

        let result = parse_entity_extraction(json);

        // Verify entity iteration
        let entity_names: Vec<&str> = result.entities.iter().map(|e| e.name.as_str()).collect();
        assert_eq!(entity_names, vec!["Alice", "Bob", "Acme Corp"]);

        // Verify relationship iteration
        let rel_sources: Vec<&str> = result.relationships.iter().map(|r| r.source.as_str()).collect();
        assert_eq!(rel_sources, vec!["Alice", "Alice"]);
    }

    #[test]
    fn test_entity_id_tracking_hashmap() {
        // Test the HashMap<String, i64> pattern used for tracking entity IDs
        let mut entity_ids: HashMap<String, i64> = HashMap::new();

        // Simulate adding entities
        entity_ids.insert("Alice".to_string(), 1);
        entity_ids.insert("Bob".to_string(), 2);
        entity_ids.insert("Acme Corp".to_string(), 3);

        // Verify lookups work
        assert_eq!(entity_ids.get("Alice"), Some(&1));
        assert_eq!(entity_ids.get("Bob"), Some(&2));
        assert_eq!(entity_ids.get("Unknown"), None);
    }

    #[test]
    fn test_merge_with_existing_entities() {
        use crate::memory::prompts::{find_entity_match, MergeConfig};

        // Simulate the merge logic in add_to_graph
        let new_entity = ExtractedEntity::new("Alice Smith", EntityType::Person);
        let new_embedding = vec![1.0, 0.0, 0.0];

        let existing = vec![
            EntityWithEmbedding {
                id: 1,
                name: "Alice".to_string(),
                entity_type: "person".to_string(),
                embedding: Some(vec![0.99, 0.01, 0.0]), // Very similar
            },
            EntityWithEmbedding {
                id: 2,
                name: "Bob".to_string(),
                entity_type: "person".to_string(),
                embedding: Some(vec![0.0, 1.0, 0.0]),
            },
        ];

        let config = MergeConfig::default(); // threshold 0.85

        let result = find_entity_match(&new_entity, &new_embedding, &existing, &config);

        // Should merge with Alice (high similarity)
        assert!(result.matched, "Should find a match above 0.85 threshold");
        assert_eq!(result.matched_entity_id, Some(1));
        assert_eq!(result.matched_entity_name.as_deref(), Some("Alice"));
    }

    #[test]
    fn test_no_merge_below_threshold() {
        use crate::memory::prompts::{find_entity_match, MergeConfig};

        let new_entity = ExtractedEntity::new("Alice Smith", EntityType::Person);
        let new_embedding = vec![1.0, 0.0, 0.0];

        let existing = vec![EntityWithEmbedding {
            id: 1,
            name: "Bob".to_string(),
            entity_type: "person".to_string(),
            embedding: Some(vec![0.0, 1.0, 0.0]), // Orthogonal, 0.0 similarity
        }];

        let config = MergeConfig::default(); // threshold 0.85

        let result = find_entity_match(&new_entity, &new_embedding, &existing, &config);

        // Should NOT merge (0.0 similarity < 0.85 threshold)
        assert!(!result.matched);
    }

    #[test]
    fn test_message_content_concatenation() {
        // Test the message content concatenation used in add_to_graph
        let messages = vec![
            Message::user("Alice works at Acme Corp."),
            Message::assistant("I see! Alice is an employee at Acme Corp."),
            Message::user("She knows Bob who also works there."),
        ];

        let text: String = messages
            .iter()
            .map(|m| m.content.as_str())
            .collect::<Vec<_>>()
            .join("\n");

        assert!(text.contains("Alice works at Acme Corp."));
        assert!(text.contains("She knows Bob"));
        assert!(text.contains("\n")); // Joined with newlines
    }

    #[test]
    fn test_empty_messages_handling() {
        // Test handling of empty messages
        let messages: Vec<Message> = vec![];

        let text: String = messages
            .iter()
            .map(|m| m.content.as_str())
            .collect::<Vec<_>>()
            .join("\n");

        assert!(text.trim().is_empty());
    }
}
