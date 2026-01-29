//! Core Memory implementation.

use std::collections::HashMap;
use std::sync::{Arc, Mutex};
use tokio::sync::RwLock;
use uuid::Uuid;

use crate::config::MemoryConfig;
use crate::error::{RookError, RookResult};
use crate::ingestion::{
    IngestDecision, IngestResult, PredictionErrorGate, StrengthSignal, StrengthSignalProcessor,
};
use crate::traits::{
    Embedder, EmbeddingAction, GraphStore, Llm, Reranker, VectorRecord, VectorSearchResult,
    VectorStore,
};
use crate::types::{
    AddResult, Filter, Grade, GraphRelation, MemoryEvent, MemoryItem, MemoryResult, MemoryType,
    Message, MessageInput, MessageRole, SearchResult,
};

use super::history::{HistoryEvent, HistoryStore};
use super::json_parser::{parse_facts, parse_memory_actions};
use super::prompts::{
    agent_memory_extraction_prompt, build_update_memory_message, procedural_memory_prompt,
    user_memory_extraction_prompt,
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
        })
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
        let scope = SessionScope::new(user_id, agent_id, run_id);
        scope.validate()?;

        let mut effective_filters = scope.to_filters();
        if let Some(additional) = filters {
            effective_filters.extend(additional);
        }

        // Search vector store
        let mut memories = self
            .search_vector_store(query, &effective_filters, limit, threshold)
            .await?;

        // Apply reranking if enabled
        if rerank {
            if let Some(ref reranker) = self.reranker {
                memories = reranker.rerank(query, memories, Some(limit)).await?;
            }
        }

        // Search graph store (if enabled)
        let relations = if let Some(ref _graph) = self.graph_store {
            // TODO: Implement graph search
            None
        } else {
            None
        };

        Ok(SearchResult {
            results: memories,
            relations,
        })
    }

    /// Get a specific memory by ID.
    pub async fn get(&self, memory_id: &str) -> RookResult<Option<MemoryItem>> {
        let record = self.vector_store.get(memory_id).await?;
        Ok(record.map(|r| self.record_to_memory_item(r, None)))
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

        Ok(MemoryItem {
            id: memory_id.to_string(),
            memory: data.to_string(),
            hash: Some(format!("{:x}", md5::compute(data.as_bytes()))),
            score: None,
            metadata: Some(payload),
            created_at: existing
                .payload
                .get("created_at")
                .and_then(|v| v.as_str())
                .map(|s| s.to_string()),
            updated_at: Some(updated_at),
            memory_state: None,
            dual_strength: None,
            is_key: false,
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

        let mut payload = metadata.clone();
        payload.insert("data".to_string(), serde_json::Value::String(data.to_string()));
        payload.insert("hash".to_string(), serde_json::Value::String(hash));
        payload.insert(
            "created_at".to_string(),
            serde_json::Value::String(created_at.clone()),
        );

        let record = VectorRecord::new(memory_id.clone(), embedding, payload.clone());
        self.vector_store.insert(vec![record]).await?;

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

    async fn add_to_graph(
        &self,
        _messages: &[Message],
        _filters: &HashMap<String, serde_json::Value>,
    ) -> RookResult<Vec<GraphRelation>> {
        // TODO: Implement graph store integration
        Ok(vec![])
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
            memory_state: None,
            dual_strength: None,
            is_key: false,
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
            memory_state: None,
            dual_strength: None,
            is_key: false,
        }
    }
}
