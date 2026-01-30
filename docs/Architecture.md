# Architecture

Technical deep dive into Rook's architecture and design.

## System Overview

```
┌─────────────────────────────────────────────────────────────────┐
│                         Applications                             │
├─────────────┬─────────────┬─────────────┬─────────────┬─────────┤
│ REST API    │ Rust Lib    │ Python      │ MCP Server  │ Client  │
│ (rook-      │ (rook-core) │ (rook-      │ (rook-mcp)  │ (rook-  │
│  server)    │             │  python)    │             │  client)│
├─────────────┴─────────────┴─────────────┴─────────────┴─────────┤
│                         rook-core                                │
│  ┌─────────┐ ┌─────────┐ ┌─────────┐ ┌─────────┐ ┌─────────┐   │
│  │ Memory  │ │ FSRS-6  │ │Ingestion│ │ Search  │ │ Events  │   │
│  │ Engine  │ │Scheduler│ │ Gate    │ │ Engine  │ │  Bus    │   │
│  └─────────┘ └─────────┘ └─────────┘ └─────────┘ └─────────┘   │
│  ┌─────────┐ ┌─────────┐ ┌─────────┐ ┌─────────┐               │
│  │Category │ │Consolid-│ │Intention│ │ Graph   │               │
│  │Classify │ │ ation   │ │ Engine  │ │ Memory  │               │
│  └─────────┘ └─────────┘ └─────────┘ └─────────┘               │
├─────────────────────────────────────────────────────────────────┤
│                       Provider Traits                            │
│  ┌─────────┐ ┌─────────┐ ┌─────────┐ ┌─────────┐ ┌─────────┐   │
│  │   Llm   │ │Embedder │ │ Vector  │ │ Graph   │ │Reranker │   │
│  │         │ │         │ │ Store   │ │ Store   │ │         │   │
│  └─────────┘ └─────────┘ └─────────┘ └─────────┘ └─────────┘   │
├─────────────────────────────────────────────────────────────────┤
│                     Provider Implementations                     │
│  ┌─────────────────┐ ┌─────────────────┐ ┌─────────────────┐   │
│  │    rook-llm     │ │ rook-embeddings │ │rook-vector-     │   │
│  │ OpenAI,Anthropic│ │ OpenAI, Cohere  │ │stores: Qdrant,  │   │
│  │ Ollama, Azure   │ │ Ollama, HF      │ │pgvector, etc.   │   │
│  └─────────────────┘ └─────────────────┘ └─────────────────┘   │
│  ┌─────────────────┐ ┌─────────────────┐                        │
│  │rook-graph-stores│ │ rook-rerankers  │                        │
│  │Embedded, Neo4j  │ │ Cohere, LLM     │                        │
│  └─────────────────┘ └─────────────────┘                        │
└─────────────────────────────────────────────────────────────────┘
```

## Crate Structure

### Core Crates

| Crate | Description | Key Types |
|-------|-------------|-----------|
| `rook-core` | Memory engine, algorithms, types | `Memory`, `MemoryItem`, `FsrsScheduler` |
| `rook-llm` | LLM provider implementations | `Llm` trait, `OpenAiLlm`, `AnthropicLlm` |
| `rook-embeddings` | Embedding providers | `Embedder` trait, `OpenAiEmbedder` |
| `rook-vector-stores` | Vector storage backends | `VectorStore` trait, `QdrantVectorStore` |
| `rook-graph-stores` | Graph storage backends | `GraphStore` trait, `EmbeddedGraphStore` |
| `rook-rerankers` | Result reranking | `Reranker` trait, `CohereReranker` |

### Application Crates

| Crate | Description |
|-------|-------------|
| `rook-server` | Axum REST API server |
| `rook-client` | HTTP client library |
| `rook-python` | PyO3 Python bindings |
| `rook-mcp` | MCP server for Claude Code |
| `rook-extractors` | Document/image extraction |

## Key Traits

### Llm

```rust
#[async_trait]
pub trait Llm: Send + Sync {
    async fn generate(
        &self,
        messages: &[Message],
        options: Option<GenerationOptions>,
    ) -> Result<LlmResponse, RookError>;

    async fn generate_stream(
        &self,
        messages: &[Message],
        options: Option<GenerationOptions>,
    ) -> Result<LlmStream, RookError>;
}
```

Implementations: `OpenAiLlm`, `AnthropicLlm`, `OllamaLlm`, `AzureOpenAiLlm`, `GroqLlm`, `TogetherLlm`

### Embedder

```rust
#[async_trait]
pub trait Embedder: Send + Sync {
    async fn embed(&self, texts: &[String]) -> Result<Vec<Vec<f32>>, RookError>;
    fn dimensions(&self) -> usize;
}
```

Implementations: `OpenAiEmbedder`, `CohereEmbedder`, `OllamaEmbedder`, `HuggingFaceEmbedder`

### VectorStore

```rust
#[async_trait]
pub trait VectorStore: Send + Sync {
    async fn insert(&self, items: &[VectorItem]) -> Result<(), RookError>;
    async fn search(&self, query: &[f32], limit: usize, filter: Option<Filter>)
        -> Result<Vec<SearchResult>, RookError>;
    async fn get(&self, id: &str) -> Result<Option<VectorItem>, RookError>;
    async fn update(&self, id: &str, item: &VectorItem) -> Result<(), RookError>;
    async fn delete(&self, id: &str) -> Result<(), RookError>;
}
```

Implementations: `QdrantVectorStore`, `PgVectorStore`, `PineconeVectorStore`, `WeaviateVectorStore`, etc.

### GraphStore

```rust
#[async_trait]
pub trait GraphStore: Send + Sync {
    async fn add_node(&self, node: &GraphNode) -> Result<String, RookError>;
    async fn add_edge(&self, edge: &GraphEdge) -> Result<(), RookError>;
    async fn get_neighbors(&self, node_id: &str) -> Result<Vec<GraphNode>, RookError>;
    async fn traverse(&self, start: &str, depth: usize) -> Result<Vec<GraphNode>, RookError>;
}
```

Implementations: `EmbeddedGraphStore`, `Neo4jGraphStore`

### Reranker

```rust
#[async_trait]
pub trait Reranker: Send + Sync {
    async fn rerank(
        &self,
        query: &str,
        documents: &[String],
        top_k: usize,
    ) -> Result<Vec<RerankResult>, RookError>;
}
```

Implementations: `CohereReranker`, `LlmReranker`

## Core Components

### Memory Engine

The central component that coordinates all operations:

```rust
pub struct Memory {
    llm: Box<dyn Llm>,
    embedder: Box<dyn Embedder>,
    vector_store: Box<dyn VectorStore>,
    graph_store: Option<Box<dyn GraphStore>>,
    reranker: Option<Box<dyn Reranker>>,
    fsrs: FsrsScheduler,
    classifier: Classifier,
    event_bus: Option<EventBus>,
    // ...
}
```

### FSRS Scheduler

Manages memory strength calculations:

```rust
pub struct FsrsScheduler {
    decay_rate: f64,
    stability_base: f64,
    difficulty_default: f64,
}

impl FsrsScheduler {
    pub fn retrievability(&self, state: &FsrsState) -> f64;
    pub fn schedule_review(&self, state: &mut FsrsState, grade: Grade);
    pub fn should_forget(&self, state: &FsrsState, threshold: f64) -> bool;
}
```

### Prediction Error Gate

Handles smart ingestion decisions:

```rust
pub struct PredictionErrorGate {
    llm: Arc<dyn Llm>,
    embedder: Arc<dyn Embedder>,
    vector_store: Arc<dyn VectorStore>,
}

impl PredictionErrorGate {
    pub async fn evaluate(&self, input: &str, existing: &[MemoryItem])
        -> Result<IngestDecision, RookError>;
}

pub enum IngestDecision {
    Skip { reason: String },
    Create,
    Update { target: String },
    Supersede { target: String },
}
```

### Hybrid Search

Combines multiple retrieval methods:

```rust
pub struct HybridRetriever {
    vector_store: Arc<dyn VectorStore>,
    graph_store: Option<Arc<dyn GraphStore>>,
    text_index: Option<TantivyIndex>,
    reranker: Option<Arc<dyn Reranker>>,
}

impl HybridRetriever {
    pub async fn search(
        &self,
        query: &str,
        mode: RetrievalMode,
        limit: usize,
    ) -> Result<Vec<SearchResult>, RookError>;
}
```

## Data Flow

### Add Memory

```
Input messages
      ↓
┌─────────────────┐
│  Fact Extractor │ ← LLM extracts facts from messages
└────────┬────────┘
         ↓
┌─────────────────┐
│ Prediction Error│ ← Compare to existing memories
│     Gate        │
└────────┬────────┘
         ↓
    ┌────┴────┐
    │Decision │
    └────┬────┘
    ┌────┼────┬────┐
    ↓    ↓    ↓    ↓
  Skip Create Update Supersede
         ↓    ↓
    ┌────┴────┴────┐
    │  Classifier  │ ← Assign category
    └──────┬───────┘
           ↓
    ┌──────┴───────┐
    │Entity Extract│ ← Extract entities for graph
    └──────┬───────┘
           ↓
    ┌──────┴───────┐
    │    Embed     │ ← Generate vector embedding
    └──────┬───────┘
           ↓
    ┌──────┴───────┐
    │    Store     │ ← Save to vector + graph stores
    └──────┬───────┘
           ↓
    ┌──────┴───────┐
    │  Emit Event  │ ← Notify subscribers
    └──────────────┘
```

### Search

```
Query
  ↓
┌─────────────────┐
│   Embed Query   │
└────────┬────────┘
         ↓
┌─────────────────┐
│  Vector Search  │ ← Initial candidates
└────────┬────────┘
         ↓ (if Standard/Cognitive mode)
┌─────────────────┐
│   Spreading     │ ← Activate related via graph
│   Activation    │
└────────┬────────┘
         ↓ (if Standard/Precise mode)
┌─────────────────┐
│  Keyword/BM25   │ ← Full-text search
└────────┬────────┘
         ↓
┌─────────────────┐
│     Fusion      │ ← Combine with RRF
└────────┬────────┘
         ↓ (if Cognitive mode)
┌─────────────────┐
│  FSRS Weighting │ ← Boost by retrievability
└────────┬────────┘
         ↓ (if Precise mode)
┌─────────────────┐
│    Reranking    │ ← LLM/Cohere rerank
└────────┬────────┘
         ↓
┌─────────────────┐
│  Deduplication  │
└────────┬────────┘
         ↓
     Results
```

## Storage Schema

### Vector Store

Each memory item stored with:

```json
{
  "id": "uuid",
  "vector": [0.1, 0.2, ...],
  "payload": {
    "memory": "User likes Rust programming",
    "user_id": "user123",
    "agent_id": null,
    "run_id": null,
    "category": "preference",
    "is_key": false,
    "metadata": {},
    "fsrs_state": {
      "stability": 1.5,
      "difficulty": 0.3,
      "last_review": "2025-01-30T10:00:00Z",
      "review_count": 2
    },
    "created_at": "2025-01-30T09:00:00Z",
    "updated_at": "2025-01-30T10:00:00Z"
  }
}
```

### Graph Store

Nodes:
```json
{
  "id": "node_uuid",
  "name": "Alice",
  "type": "person",
  "properties": {},
  "activation": 0.0
}
```

Edges:
```json
{
  "source": "alice_id",
  "target": "acme_id",
  "relationship": "works_at",
  "weight": 1.0,
  "memory_id": "memory_uuid"
}
```

## Background Processes

### Consolidation Scheduler

```rust
pub struct ConsolidationScheduler {
    memory: Arc<Memory>,
    interval: Duration,
}

// Runs periodically to:
// 1. Check synaptic tags
// 2. Consolidate tagged memories
// 3. Archive forgotten memories
```

### Intention Scheduler

```rust
pub struct IntentionScheduler {
    memory: Arc<Memory>,
    check_interval: usize,  // Every N messages
}

// Checks for:
// 1. Keyword triggers (via bloom filter)
// 2. Scheduled time triggers
// 3. Semantic topic matches
```

## Extension Points

### Adding a New LLM Provider

1. Implement the `Llm` trait in `rook-llm`
2. Add to the factory in `rook-server`

```rust
// rook-llm/src/my_provider.rs
pub struct MyLlm { /* ... */ }

#[async_trait]
impl Llm for MyLlm {
    async fn generate(&self, messages: &[Message], options: Option<GenerationOptions>)
        -> Result<LlmResponse, RookError> {
        // Implementation
    }
}
```

### Adding a New Vector Store

1. Implement the `VectorStore` trait in `rook-vector-stores`
2. Add to the factory

### Custom Ingestion Logic

Extend `PredictionErrorGate` or implement custom detection layers:

```rust
pub trait ConflictDetector: Send + Sync {
    async fn detect(&self, input: &str, existing: &MemoryItem)
        -> Result<ConflictResult, RookError>;
}
```

## Performance Considerations

### Embedding Caching
- Embeddings are cached per session
- Consider external cache for production

### Connection Pooling
- Use `pgvector_pooled` for PostgreSQL
- Qdrant has built-in connection management

### Batch Operations
- Add memories in batches when possible
- Vector stores support batch insert

### Index Optimization
- HNSW indexes for vector stores
- Bloom filters for intention keywords
- Tantivy for full-text search

## Security

### API Authentication
- Bearer token authentication
- Configurable via `ROOK_API_KEY`

### Data Isolation
- Memories scoped by `user_id`
- No cross-user data access

### Secrets
- API keys via environment variables
- No secrets in logs
