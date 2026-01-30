# Rust Library Guide

Using rook-core directly in your Rust applications.

## Installation

Add the required crates to your `Cargo.toml`:

```toml
[dependencies]
rook-core = "0.1"
rook-llm = "0.1"
rook-embeddings = "0.1"
rook-vector-stores = "0.1"
rook-graph-stores = "0.1"  # Optional, for graph features
tokio = { version = "1", features = ["full"] }
anyhow = "1.0"
```

## Basic Usage

### Creating a Memory Instance

```rust
use rook_core::memory::{Memory, MemoryBuilder};
use rook_core::types::Message;
use rook_llm::openai::OpenAiLlm;
use rook_embeddings::openai::OpenAiEmbedder;
use rook_vector_stores::qdrant::QdrantVectorStore;

#[tokio::main]
async fn main() -> anyhow::Result<()> {
    let api_key = std::env::var("OPENAI_API_KEY")?;

    // Create providers
    let llm = OpenAiLlm::new("gpt-4-turbo", &api_key)?;
    let embedder = OpenAiEmbedder::new("text-embedding-3-small", &api_key)?;
    let vector_store = QdrantVectorStore::new(
        "http://localhost:6333",
        "rook",
        1536
    ).await?;

    // Build memory
    let memory = MemoryBuilder::new()
        .llm(Box::new(llm))
        .embedder(Box::new(embedder))
        .vector_store(Box::new(vector_store))
        .build()?;

    Ok(())
}
```

### Adding Memories

```rust
// From conversation messages
let messages = vec![
    Message::user("My name is Alice and I'm a software engineer"),
    Message::assistant("Nice to meet you, Alice!"),
];

let results = memory.add(
    &messages,
    Some("user123"),  // user_id
    None,             // agent_id
    None,             // run_id
    None,             // metadata
    true,             // infer (extract facts)
).await?;

for result in results {
    println!("Added: {} (event: {:?})", result.memory, result.event);
}
```

### Searching Memories

```rust
// Basic search
let results = memory.search(
    "What is the user's profession?",
    Some("user123"),  // user_id
    None,             // agent_id
    None,             // run_id
    Some(10),         // limit
).await?;

for result in results {
    println!("{}: {} (score: {:.2})",
        result.id,
        result.memory,
        result.score.unwrap_or(0.0)
    );
}
```

### Hybrid Search

```rust
use rook_core::memory::RetrievalMode;

// Use hybrid retrieval with spreading activation
let results = memory.hybrid_search(
    "programming languages",
    Some("user123"),
    None,
    None,
    Some(10),
    RetrievalMode::Cognitive,  // Spreading activation + FSRS weighting
).await?;
```

**Retrieval Modes:**
- `Quick` - Vector search only (fastest)
- `Standard` - Vector + spreading activation + keyword
- `Precise` - All methods + reranking
- `Cognitive` - Spreading activation + FSRS weighting

### CRUD Operations

```rust
// Get a specific memory
let item = memory.get("memory-id-here").await?;

// Update a memory
memory.update("memory-id-here", "Updated content").await?;

// Delete a memory
memory.delete("memory-id-here").await?;

// Delete all memories for a user
memory.delete_all(Some("user123"), None, None).await?;
```

## Advanced Features

### Smart Ingestion

Use prediction error gating to intelligently handle new information:

```rust
use rook_core::ingestion::{PredictionErrorGate, IngestResult};

let gate = PredictionErrorGate::new(
    memory.llm(),
    memory.embedder(),
    memory.vector_store(),
);

let result = gate.smart_ingest(
    "Alice now works at BigCorp",
    Some("user123"),
    None,
).await?;

match result {
    IngestResult::Created(item) => println!("New memory: {}", item.memory),
    IngestResult::Updated(item) => println!("Updated: {}", item.memory),
    IngestResult::Superseded { old, new } => println!("Replaced old memory"),
    IngestResult::Skipped(reason) => println!("Skipped: {}", reason),
}
```

### FSRS Memory Strength

```rust
use rook_core::fsrs::Grade;

// Promote a memory (user confirmed it's correct)
memory.promote("memory-id", Grade::Good).await?;

// Demote a memory (user corrected it)
memory.demote("memory-id", Grade::Again).await?;

// Process a review (used in response)
memory.process_review("memory-id", Grade::Easy).await?;

// Get retrievability (0.0-1.0)
let item = memory.get("memory-id").await?;
let retrievability = item.fsrs_state.retrievability();
println!("Retrievability: {:.1}%", retrievability * 100.0);
```

### Classification

```rust
use rook_core::classification::Category;

// Memories are auto-classified on add
// Categories: Personal, Preference, Fact, Belief, Skill, Experience,
//             Social, Work, Health, Other

// Filter by category
let work_memories = memory.search_by_category(
    "projects",
    Category::Work,
    Some("user123"),
    Some(10),
).await?;

// Mark as key memory (always retrieved)
memory.set_key_memory("memory-id", true).await?;
```

### Graph Memory

```rust
use rook_graph_stores::embedded::EmbeddedGraphStore;

// Add graph store to memory
let graph_store = EmbeddedGraphStore::new("./rook_graph.db").await?;

let memory = MemoryBuilder::new()
    .llm(Box::new(llm))
    .embedder(Box::new(embedder))
    .vector_store(Box::new(vector_store))
    .graph_store(Box::new(graph_store))
    .build()?;

// Entities are automatically extracted on add
// Search uses spreading activation through the graph
```

### Consolidation

```rust
use rook_core::consolidation::ConsolidationManager;

// Run consolidation (typically on a schedule)
let manager = ConsolidationManager::new(&memory);
let consolidated = manager.consolidate().await?;
println!("Consolidated {} memories", consolidated);

// Tag a memory as novel (boosts nearby memories)
manager.tag_as_novel("memory-id").await?;
```

### Intentions

```rust
use rook_core::intentions::{Intention, TriggerCondition};

// Create an intention
let intention = Intention::new(
    "Remind about meeting",
    TriggerCondition::KeywordMention(vec!["meeting", "schedule"]),
    chrono::Utc::now() + chrono::Duration::days(7),
);

memory.add_intention(intention, Some("user123")).await?;

// Check for triggered intentions
let triggered = memory.check_intentions(
    "Can you help me schedule something?",
    Some("user123"),
).await?;
```

### Events

```rust
use rook_core::events::{EventBus, MemoryEvent};

// Subscribe to memory events
let event_bus = EventBus::new();

event_bus.subscribe(|event: MemoryEvent| {
    match event {
        MemoryEvent::Created(item) => println!("Created: {}", item.id),
        MemoryEvent::Updated(item) => println!("Updated: {}", item.id),
        MemoryEvent::Deleted(id) => println!("Deleted: {}", id),
        MemoryEvent::Accessed(id) => println!("Accessed: {}", id),
    }
});

let memory = MemoryBuilder::new()
    // ... providers
    .event_bus(event_bus)
    .build()?;
```

### Multimodal Extraction

```rust
use rook_extractors::{PdfExtractor, DocxExtractor, ImageExtractor};

// Extract text from PDF
let pdf_extractor = PdfExtractor::new();
let text = pdf_extractor.extract("document.pdf")?;

// Add extracted text as memories
let messages = vec![Message::user(&text)];
memory.add(&messages, Some("user123"), None, None, None, true).await?;

// Extract from images (OCR + vision LLM)
let image_extractor = ImageExtractor::new(llm.clone());
let description = image_extractor.extract("photo.jpg").await?;
```

## Provider Options

### LLM Providers

```rust
use rook_llm::{openai::OpenAiLlm, anthropic::AnthropicLlm, ollama::OllamaLlm};

// OpenAI
let llm = OpenAiLlm::new("gpt-4-turbo", &api_key)?;

// Anthropic
let llm = AnthropicLlm::new("claude-3-opus", &api_key)?;

// Ollama (local)
let llm = OllamaLlm::new("http://localhost:11434", "llama2")?;
```

### Embedding Providers

```rust
use rook_embeddings::{openai::OpenAiEmbedder, cohere::CohereEmbedder};

// OpenAI
let embedder = OpenAiEmbedder::new("text-embedding-3-small", &api_key)?;

// Cohere
let embedder = CohereEmbedder::new("embed-english-v3.0", &api_key)?;
```

### Vector Stores

```rust
use rook_vector_stores::{qdrant::QdrantVectorStore, pgvector::PgVectorStore};

// Qdrant
let store = QdrantVectorStore::new("http://localhost:6333", "rook", 1536).await?;

// PostgreSQL + pgvector
let store = PgVectorStore::new("postgresql://localhost/rook", "memories", 1536).await?;
```

## Error Handling

```rust
use rook_core::error::RookError;

match memory.add(&messages, user_id, None, None, None, true).await {
    Ok(results) => println!("Added {} memories", results.len()),
    Err(RookError::LlmError(e)) => eprintln!("LLM failed: {}", e),
    Err(RookError::EmbeddingError(e)) => eprintln!("Embedding failed: {}", e),
    Err(RookError::VectorStoreError(e)) => eprintln!("Storage failed: {}", e),
    Err(e) => eprintln!("Other error: {}", e),
}
```

## Testing

```rust
#[cfg(test)]
mod tests {
    use super::*;
    use rook_core::memory::MockMemory;

    #[tokio::test]
    async fn test_search() {
        let memory = MockMemory::new();
        // ... test with mock
    }
}
```
