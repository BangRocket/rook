# mem0-rs Usage Guide

This guide covers three ways to use mem0-rs:
1. **Embedded Library** - Integrate directly into your Rust application
2. **REST Server** - Run as a standalone memory service
3. **Remote Client** - Connect to hosted Mem0 API

---

## Table of Contents

- [Prerequisites](#prerequisites)
- [Method 1: Embedded Library](#method-1-embedded-library)
- [Method 2: REST Server](#method-2-rest-server)
- [Method 3: Remote Client](#method-3-remote-client)
- [Configuration Reference](#configuration-reference)

---

## Prerequisites

Add the required crates to your `Cargo.toml`:

```toml
# For embedded library usage
[dependencies]
mem0-core = { path = "path/to/mem0-rs/crates/mem0-core" }
mem0-llm = { path = "path/to/mem0-rs/crates/mem0-llm" }
mem0-embeddings = { path = "path/to/mem0-rs/crates/mem0-embeddings" }
mem0-vector-stores = { path = "path/to/mem0-rs/crates/mem0-vector-stores", features = ["qdrant"] }
tokio = { version = "1", features = ["full"] }

# For running the server
mem0-server = { path = "path/to/mem0-rs/crates/mem0-server" }

# For remote client usage
mem0-client = { path = "path/to/mem0-rs/crates/mem0-client" }
```

---

## Method 1: Embedded Library

Use mem0-rs directly in your application for maximum performance and control.

### Basic Setup

```rust
use std::sync::Arc;
use mem0_core::config::{
    EmbedderProviderConfig, LlmProvider, LlmProviderConfig, MemoryConfig,
};
use mem0_core::memory::Memory;
use mem0_core::traits::{
    EmbedderConfig, EmbedderProvider, LlmConfig, VectorStoreConfig, VectorStoreProvider,
};
use mem0_llm::OpenAIProvider;
use mem0_embeddings::OpenAIEmbedder;
use mem0_vector_stores::QdrantVectorStore;

#[tokio::main]
async fn main() -> Result<(), Box<dyn std::error::Error>> {
    // 1. Configure LLM
    let llm_config = LlmProviderConfig {
        provider: LlmProvider::OpenAI,
        config: LlmConfig {
            model: "gpt-4o-mini".to_string(),
            api_key: Some(std::env::var("OPENAI_API_KEY")?),
            temperature: 0.7,
            max_tokens: 4096,
            ..Default::default()
        },
    };

    // 2. Configure Embedder
    let embedder_config = EmbedderProviderConfig {
        provider: EmbedderProvider::OpenAI,
        config: EmbedderConfig {
            model: "text-embedding-3-small".to_string(),
            api_key: Some(std::env::var("OPENAI_API_KEY")?),
            ..Default::default()
        },
    };

    // 3. Configure Vector Store
    let vector_store_config = VectorStoreConfig {
        provider: VectorStoreProvider::Qdrant,
        collection_name: "memories".to_string(),
        embedding_model_dims: 1536,
        config: serde_json::json!({
            "url": "http://localhost:6334"
        }),
    };

    // 4. Create providers
    let llm = Arc::new(OpenAIProvider::new(llm_config.config)?);
    let embedder = Arc::new(OpenAIEmbedder::new(embedder_config.config)?);
    let vector_store = Arc::new(QdrantVectorStore::new(vector_store_config.clone()).await?);

    // 5. Create Memory instance
    let config = MemoryConfig {
        llm: llm_config,
        embedder: embedder_config,
        vector_store: vector_store_config,
        graph_store: None,
        reranker: None,
        custom_prompt: None,
        history_db_path: None,
    };

    let memory = Memory::new(config, llm, embedder, vector_store, None, None)?;

    // Now use the memory!
    println!("Memory system initialized successfully!");

    Ok(())
}
```

### Adding Memories

```rust
use std::collections::HashMap;

// Add a simple text memory
let result = memory
    .add(
        "I love playing tennis on weekends",
        Some("user_123".to_string()),  // user_id
        None,                           // agent_id
        None,                           // run_id
        None,                           // metadata
        true,                           // infer (extract facts)
        None,                           // memory_type
    )
    .await?;

println!("Added {} memories", result.results.len());
for mem in &result.results {
    println!("  - {}: {}", mem.id, mem.memory);
}

// Add with metadata
let mut metadata = HashMap::new();
metadata.insert("category".to_string(), serde_json::json!("preferences"));
metadata.insert("importance".to_string(), serde_json::json!("high"));

let result = memory
    .add(
        "My favorite programming language is Rust",
        Some("user_123".to_string()),
        None,
        None,
        Some(metadata),
        true,
        None,
    )
    .await?;
```

### Searching Memories

```rust
// Basic search
let results = memory
    .search(
        "What sports do I like?",
        Some("user_123".to_string()),  // user_id
        None,                           // agent_id
        None,                           // run_id
        10,                             // limit
        None,                           // filters
    )
    .await?;

for result in &results.results {
    println!("Memory: {} (score: {:.3})", result.memory, result.score.unwrap_or(0.0));
}

// Search with filters
use mem0_core::types::Filter;

let filter = Filter::eq("category", "preferences");
let results = memory
    .search(
        "programming",
        Some("user_123".to_string()),
        None,
        None,
        5,
        Some(filter),
    )
    .await?;
```

### Getting All Memories

```rust
// Get all memories for a user
let all_memories = memory
    .get_all(
        Some("user_123".to_string()),
        None,
        None,
        None,  // limit
    )
    .await?;

println!("User has {} memories", all_memories.results.len());
```

### Updating Memories

```rust
// Update an existing memory
let memory_id = "mem_abc123";
memory
    .update(
        memory_id,
        "I love playing tennis and basketball on weekends",
    )
    .await?;
```

### Deleting Memories

```rust
// Delete a specific memory
memory.delete("mem_abc123").await?;

// Delete all memories for a user
memory.delete_all(Some("user_123".to_string()), None, None).await?;

// Reset entire memory store
memory.reset().await?;
```

### Using with Conversations

```rust
use mem0_core::types::{Message, MessageRole};

// Add memories from a conversation
let messages = vec![
    Message {
        role: MessageRole::User,
        content: "I'm planning a trip to Japan next month".to_string(),
    },
    Message {
        role: MessageRole::Assistant,
        content: "That's exciting! What cities are you planning to visit?".to_string(),
    },
    Message {
        role: MessageRole::User,
        content: "Tokyo and Kyoto. I'm really interested in temples and ramen!".to_string(),
    },
];

let result = memory
    .add(
        messages,  // Pass Vec<Message> directly
        Some("user_123".to_string()),
        None,
        None,
        None,
        true,
        None,
    )
    .await?;
```

### With Graph Store (Optional)

```rust
use mem0_core::traits::{GraphStoreConfig, GraphStoreProvider};
use mem0_graph_stores::Neo4jGraphStore;

// Configure graph store for relationship extraction
let graph_config = GraphStoreConfig {
    provider: GraphStoreProvider::Neo4j,
    url: "bolt://localhost:7687".to_string(),
    username: Some("neo4j".to_string()),
    password: Some("password".to_string()),
    database: None,
};

let graph_store = Arc::new(Neo4jGraphStore::new(graph_config.clone()).await?);

// Include in Memory creation
let memory = Memory::new(
    config,
    llm,
    embedder,
    vector_store,
    Some(graph_store),  // Enable graph storage
    None,
)?;
```

### With Reranker (Optional)

```rust
use mem0_core::traits::{RerankerConfig, RerankerProvider};
use mem0_rerankers::CohereReranker;

// Configure reranker for better search results
let reranker_config = RerankerConfig {
    provider: RerankerProvider::Cohere,
    model: "rerank-english-v2.0".to_string(),
    api_key: Some(std::env::var("COHERE_API_KEY")?),
    top_n: Some(5),
};

let reranker = Arc::new(CohereReranker::new(reranker_config)?);

// Include in Memory creation
let memory = Memory::new(
    config,
    llm,
    embedder,
    vector_store,
    None,
    Some(reranker),  // Enable reranking
)?;
```

---

## Method 2: REST Server

Run mem0-rs as a standalone HTTP service that other applications can connect to.

### Running the Server

```rust
// src/main.rs
use mem0_server::{create_server, AppState};
use tokio::net::TcpListener;

#[tokio::main]
async fn main() -> Result<(), Box<dyn std::error::Error>> {
    // Initialize tracing for logging
    tracing_subscriber::fmt::init();

    // Create application state
    let state = AppState::new();

    // Create the server with all routes and middleware
    let app = create_server(state);

    // Bind to address
    let addr = "0.0.0.0:8080";
    println!("Starting mem0 server on {}", addr);

    let listener = TcpListener::bind(addr).await?;
    axum::serve(listener, app).await?;

    Ok(())
}
```

Or use the pre-built binary:

```bash
cd mem0-rs
cargo run --package mem0-server
```

### API Endpoints

#### Configure Memory

```bash
# POST /configure
curl -X POST http://localhost:8080/configure \
  -H "Content-Type: application/json" \
  -d '{
    "llm": {
      "provider": "openai",
      "model": "gpt-4o-mini",
      "api_key": "sk-..."
    },
    "embedder": {
      "provider": "openai",
      "model": "text-embedding-3-small",
      "api_key": "sk-..."
    },
    "vector_store": {
      "provider": "qdrant",
      "url": "http://localhost:6334",
      "collection_name": "memories"
    }
  }'
```

#### Add Memory

```bash
# POST /memories
curl -X POST http://localhost:8080/memories \
  -H "Content-Type: application/json" \
  -d '{
    "messages": [
      {"role": "user", "content": "I love hiking in the mountains"}
    ],
    "user_id": "user_123",
    "infer": true
  }'
```

Response:
```json
{
  "results": [
    {
      "id": "mem_abc123",
      "memory": "User loves hiking in the mountains",
      "event": "ADD"
    }
  ]
}
```

#### Search Memories

```bash
# POST /search
curl -X POST http://localhost:8080/search \
  -H "Content-Type: application/json" \
  -d '{
    "query": "outdoor activities",
    "user_id": "user_123",
    "limit": 10
  }'
```

Response:
```json
{
  "results": [
    {
      "id": "mem_abc123",
      "memory": "User loves hiking in the mountains",
      "score": 0.89,
      "metadata": {}
    }
  ]
}
```

#### Get All Memories

```bash
# GET /memories?user_id=user_123
curl "http://localhost:8080/memories?user_id=user_123"
```

#### Get Specific Memory

```bash
# GET /memories/{id}
curl http://localhost:8080/memories/mem_abc123
```

#### Update Memory

```bash
# PUT /memories/{id}
curl -X PUT http://localhost:8080/memories/mem_abc123 \
  -H "Content-Type: application/json" \
  -d '{
    "text": "User loves hiking and camping in the mountains"
  }'
```

#### Delete Memory

```bash
# DELETE /memories/{id}
curl -X DELETE http://localhost:8080/memories/mem_abc123
```

#### Delete All Memories

```bash
# DELETE /memories?user_id=user_123
curl -X DELETE "http://localhost:8080/memories?user_id=user_123"
```

#### Health Check

```bash
# GET /health
curl http://localhost:8080/health
```

### Docker Deployment

```dockerfile
# Dockerfile
FROM rust:1.75 as builder
WORKDIR /app
COPY . .
RUN cargo build --release --package mem0-server

FROM debian:bookworm-slim
RUN apt-get update && apt-get install -y ca-certificates && rm -rf /var/lib/apt/lists/*
COPY --from=builder /app/target/release/mem0-server /usr/local/bin/
EXPOSE 8080
CMD ["mem0-server"]
```

```yaml
# docker-compose.yml
version: '3.8'
services:
  mem0:
    build: .
    ports:
      - "8080:8080"
    environment:
      - OPENAI_API_KEY=${OPENAI_API_KEY}
    depends_on:
      - qdrant

  qdrant:
    image: qdrant/qdrant:latest
    ports:
      - "6333:6333"
      - "6334:6334"
    volumes:
      - qdrant_data:/qdrant/storage

volumes:
  qdrant_data:
```

### Using the Server from Rust

```rust
use reqwest::Client;
use serde::{Deserialize, Serialize};

#[derive(Serialize)]
struct AddRequest {
    messages: Vec<MessageInput>,
    user_id: Option<String>,
    infer: bool,
}

#[derive(Serialize)]
struct MessageInput {
    role: String,
    content: String,
}

#[derive(Deserialize)]
struct AddResponse {
    results: Vec<MemoryResult>,
}

#[derive(Deserialize)]
struct MemoryResult {
    id: String,
    memory: String,
    event: String,
}

async fn add_memory(client: &Client, text: &str, user_id: &str) -> Result<AddResponse, reqwest::Error> {
    let request = AddRequest {
        messages: vec![MessageInput {
            role: "user".to_string(),
            content: text.to_string(),
        }],
        user_id: Some(user_id.to_string()),
        infer: true,
    };

    client
        .post("http://localhost:8080/memories")
        .json(&request)
        .send()
        .await?
        .json()
        .await
}
```

---

## Method 3: Remote Client

Connect to the hosted Mem0 API service using the client library.

### Setup

```rust
use mem0_client::MemoryClient;

#[tokio::main]
async fn main() -> Result<(), Box<dyn std::error::Error>> {
    // Create client with API key
    let client = MemoryClient::new("your-mem0-api-key")?;

    // Or with custom options (base URL, org ID, project ID)
    let client = MemoryClient::with_options(
        "your-api-key",
        Some("https://your-mem0-instance.com/v1"),  // custom base URL
        Some("org-123"),                             // organization ID
        Some("proj-456"),                            // project ID
    )?;

    // Or from environment variables (MEM0_API_KEY, MEM0_BASE_URL, etc.)
    let client = MemoryClient::from_env()?;

    Ok(())
}
```

### Adding Memories

```rust
use mem0_client::MemoryClient;
use std::collections::HashMap;

let client = MemoryClient::new("your-api-key")?;

// Simple add with user_id
let result = client
    .add(
        "I prefer morning meetings",
        Some("user_123"),  // user_id
        None,              // agent_id
        None,              // run_id
        None,              // metadata
    )
    .await?;

println!("Added {} memories", result.len());
for mem in &result {
    println!("  - {}: {}", mem.id, mem.memory);
}

// Add with metadata
let mut metadata = HashMap::new();
metadata.insert("category".to_string(), serde_json::json!("preferences"));

let result = client
    .add(
        "My favorite programming language is Rust",
        Some("user_123"),
        None,
        None,
        Some(metadata),
    )
    .await?;
```

### Searching Memories

```rust
// Basic search
let results = client
    .search(
        "meeting preferences",
        Some("user_123"),  // user_id
        None,              // agent_id
        None,              // run_id
        Some(10),          // limit
    )
    .await?;

for memory in &results {
    println!("Found: {} (score: {:.2})",
        memory.memory,
        memory.score.unwrap_or(0.0)
    );
}
```

### Getting Memories

```rust
// Get all memories for a user
let memories = client
    .get_all(
        Some("user_123"),  // user_id
        None,              // agent_id
        None,              // run_id
    )
    .await?;

println!("Found {} memories", memories.len());

// Get specific memory by ID
if let Some(memory) = client.get("mem_abc123").await? {
    println!("Memory: {}", memory.memory);
} else {
    println!("Memory not found");
}
```

### Updating Memories

```rust
let updated = client
    .update("mem_abc123", "Updated memory content")
    .await?;

println!("Updated: {}", updated.memory);
```

### Deleting Memories

```rust
// Delete specific memory
client.delete("mem_abc123").await?;

// Delete all for user
client.delete_all(
    Some("user_123"),  // user_id
    None,              // agent_id
    None,              // run_id
).await?;

// Reset all memories
client.reset().await?;
```

### Getting Memory History

```rust
let history = client.history("mem_abc123").await?;

for entry in history {
    // History entries are returned as JSON values
    println!("Entry: {}", serde_json::to_string_pretty(&entry)?);
}
```

### Complete Example

```rust
use mem0_client::MemoryClient;
use std::collections::HashMap;

#[tokio::main]
async fn main() -> Result<(), Box<dyn std::error::Error>> {
    // Create client from environment
    let client = MemoryClient::from_env()?;
    let user_id = "user_123";

    // Add some memories
    let memories_to_add = vec![
        "I work as a software engineer at a tech startup",
        "My favorite food is sushi",
        "I have a golden retriever named Max",
        "I'm learning to play guitar",
    ];

    for memory in &memories_to_add {
        client
            .add(memory, Some(user_id), None, None, None)
            .await?;
        println!("Added: {}", memory);
    }

    // Search for relevant memories
    let queries = vec!["work", "pets", "hobbies"];

    for query in queries {
        println!("\nSearching for '{}':", query);
        let results = client
            .search(query, Some(user_id), None, None, Some(3))
            .await?;

        for result in results {
            println!("  - {} (score: {:.2})",
                result.memory,
                result.score.unwrap_or(0.0)
            );
        }
    }

    // Get all memories
    println!("\nAll memories for user:");
    let all = client.get_all(Some(user_id), None, None).await?;
    for mem in all {
        println!("  [{}] {}", mem.id, mem.memory);
    }

    Ok(())
}
```

---

## Configuration Reference

### LLM Providers

| Provider | Model Examples | Environment Variable |
|----------|---------------|---------------------|
| `openai` | `gpt-4o`, `gpt-4o-mini`, `gpt-4-turbo` | `OPENAI_API_KEY` |
| `anthropic` | `claude-3-opus`, `claude-3-sonnet` | `ANTHROPIC_API_KEY` |
| `ollama` | `llama2`, `mistral`, `codellama` | N/A (local) |

### Embedder Providers

| Provider | Model Examples | Dimensions |
|----------|---------------|------------|
| `openai` | `text-embedding-3-small`, `text-embedding-3-large` | 1536 / 3072 |
| `ollama` | `nomic-embed-text`, `mxbai-embed-large` | Varies |

### Vector Store Providers

| Provider | Use Case | Configuration |
|----------|----------|---------------|
| `qdrant` | Production, self-hosted | `url`, `api_key` |
| `pinecone` | Managed cloud service | `api_key`, `environment` |
| `chroma` | Development, embedded | `url` |
| `pgvector` | PostgreSQL integration | `url` (connection string) |
| `redis` | Redis Stack | `url` |
| `weaviate` | Semantic search | `url`, `api_key` |
| `milvus` | Large-scale vectors | `url` |

### Graph Store Providers

| Provider | Use Case |
|----------|----------|
| `neo4j` | Production graph database |
| `memgraph` | In-memory graph |

### Reranker Providers

| Provider | Model Examples |
|----------|---------------|
| `cohere` | `rerank-english-v2.0`, `rerank-multilingual-v2.0` |

---

## Best Practices

1. **Use environment variables** for API keys, never hardcode them
2. **Enable inference** (`infer: true`) for automatic fact extraction
3. **Use user_id consistently** to scope memories per user
4. **Add metadata** for better filtering and organization
5. **Use reranking** for improved search relevance in production
6. **Enable graph store** when relationships between entities matter
7. **Batch operations** when adding multiple memories for efficiency

---

## Troubleshooting

### Connection Issues

```rust
// Test vector store connection
match vector_store.list_collections().await {
    Ok(collections) => println!("Connected! Collections: {:?}", collections),
    Err(e) => eprintln!("Connection failed: {}", e),
}
```

### Memory Not Found

```rust
match memory.get("mem_abc123").await {
    Ok(Some(mem)) => println!("Found: {}", mem.memory),
    Ok(None) => println!("Memory not found"),
    Err(e) => eprintln!("Error: {}", e),
}
```

### Debug Logging

```rust
// Enable debug logging
std::env::set_var("RUST_LOG", "mem0=debug");
tracing_subscriber::fmt::init();
```
