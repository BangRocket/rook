# Getting Started

This guide walks you through installing and using Rook for the first time.

## Prerequisites

Before you begin, you'll need:

1. **A Vector Store** - Rook needs somewhere to store embeddings
   - [Qdrant](https://qdrant.tech/) - Recommended for development (easy Docker setup)
   - PostgreSQL + pgvector - Recommended for production
   - Other options: Pinecone, Weaviate, Milvus, etc.

2. **An LLM API Key** - For fact extraction and classification
   - OpenAI (`OPENAI_API_KEY`)
   - Anthropic (`ANTHROPIC_API_KEY`)
   - Or use Ollama for local models

3. **An Embedding API Key** - For semantic similarity
   - OpenAI embeddings (same key as LLM)
   - Cohere (`COHERE_API_KEY`)
   - Or use Ollama for local embeddings

## Quick Setup with Qdrant

The fastest way to get started is with Qdrant and OpenAI:

```bash
# Start Qdrant
docker run -p 6333:6333 -p 6334:6334 qdrant/qdrant

# Set your API key
export OPENAI_API_KEY="sk-..."
```

## Installation

### Option 1: REST API Server

```bash
# Clone the repository
git clone https://github.com/heidornj/rook.git
cd rook

# Build and run the server
cargo run --release --bin rook-server
```

The server starts on `http://localhost:8080`.

### Option 2: Rust Library

Add to your `Cargo.toml`:

```toml
[dependencies]
rook-core = "0.1"
rook-llm = "0.1"
rook-embeddings = "0.1"
rook-vector-stores = "0.1"
tokio = { version = "1", features = ["full"] }
```

### Option 3: Python

```bash
# From the repository
cd crates/rook-python
pip install maturin
maturin develop --release

# Or when published to PyPI
pip install rook-rs
```

## First Steps

### Using the REST API

**1. Check the server is running:**

```bash
curl http://localhost:8080/health
```

Response:
```json
{"status": "healthy", "configured": false, "version": "0.1.0"}
```

**2. Configure the memory system:**

```bash
curl -X POST http://localhost:8080/configure \
  -H "Content-Type: application/json" \
  -d '{
    "llm": {
      "provider": "openai",
      "api_key": "sk-..."
    },
    "embedder": {
      "provider": "openai",
      "api_key": "sk-..."
    },
    "vector_store": {
      "provider": "qdrant",
      "url": "http://localhost:6333"
    }
  }'
```

**3. Add your first memory:**

```bash
curl -X POST http://localhost:8080/memories \
  -H "Content-Type: application/json" \
  -d '{
    "messages": [
      {"role": "user", "content": "My name is Alice and I work as a software engineer at Acme Corp"}
    ],
    "user_id": "alice"
  }'
```

Response:
```json
{
  "results": [
    {"id": "...", "memory": "Alice works as a software engineer", "event": "ADD"},
    {"id": "...", "memory": "Alice works at Acme Corp", "event": "ADD"}
  ]
}
```

**4. Search memories:**

```bash
curl -X POST http://localhost:8080/search \
  -H "Content-Type: application/json" \
  -d '{
    "query": "Where does Alice work?",
    "user_id": "alice"
  }'
```

### Using the Rust Library

```rust
use rook_core::memory::{Memory, MemoryBuilder};
use rook_core::types::Message;
use rook_llm::openai::OpenAiLlm;
use rook_embeddings::openai::OpenAiEmbedder;
use rook_vector_stores::qdrant::QdrantVectorStore;

#[tokio::main]
async fn main() -> anyhow::Result<()> {
    // Initialize providers
    let api_key = std::env::var("OPENAI_API_KEY")?;

    let llm = OpenAiLlm::new("gpt-4-turbo", &api_key)?;
    let embedder = OpenAiEmbedder::new("text-embedding-3-small", &api_key)?;
    let vector_store = QdrantVectorStore::new(
        "http://localhost:6333",
        "rook",
        1536
    ).await?;

    // Build memory instance
    let memory = MemoryBuilder::new()
        .llm(Box::new(llm))
        .embedder(Box::new(embedder))
        .vector_store(Box::new(vector_store))
        .build()?;

    // Add memories from a conversation
    let messages = vec![
        Message::user("I'm Alice, a software engineer at Acme Corp"),
    ];

    let results = memory.add(
        &messages,
        Some("alice"),  // user_id
        None,           // agent_id
        None,           // run_id
        None,           // metadata
        true,           // infer facts
    ).await?;

    println!("Added {} memories", results.len());

    // Search
    let results = memory.search(
        "Where does the user work?",
        Some("alice"),
        None,
        None,
        Some(5),
    ).await?;

    for result in results {
        println!("- {} (score: {:.2})", result.memory, result.score.unwrap_or(0.0));
    }

    Ok(())
}
```

### Using Python

```python
import asyncio
import os
from rook_rs import Memory, MemoryConfig

async def main():
    # Configure memory
    config = MemoryConfig(
        llm_provider="openai",
        llm_api_key=os.environ["OPENAI_API_KEY"],
        embedder_provider="openai",
        embedder_api_key=os.environ["OPENAI_API_KEY"],
        vector_store_provider="qdrant",
        vector_store_url="http://localhost:6333"
    )

    memory = Memory(config)

    # Add a memory
    results = await memory.add(
        messages=[
            {"role": "user", "content": "I'm Alice, a software engineer at Acme Corp"}
        ],
        user_id="alice"
    )
    print(f"Added {len(results)} memories")

    # Search
    results = await memory.search(
        query="Where does the user work?",
        user_id="alice",
        limit=5
    )

    for r in results:
        print(f"- {r.memory} (score: {r.score:.2f})")

asyncio.run(main())
```

## What's Next?

- [REST API Reference](REST-API.md) - Complete endpoint documentation
- [Rust Library Guide](Rust-Library.md) - Advanced Rust usage
- [Python Guide](Python-Bindings.md) - Python-specific features
- [Configuration](Configuration.md) - All configuration options
- [Concepts](Concepts.md) - Understanding FSRS, prediction error, and spreading activation

## Common Issues

### "Memory not configured"

You need to call `/configure` before using memory operations.

### "Connection refused" to Qdrant

Make sure Qdrant is running:
```bash
docker run -p 6333:6333 qdrant/qdrant
```

### "Invalid API key"

Check that your environment variables are set:
```bash
echo $OPENAI_API_KEY
```

### Python import errors

Make sure you built the Python bindings:
```bash
cd crates/rook-python
maturin develop --release
```
