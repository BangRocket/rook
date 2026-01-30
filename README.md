# Rook

**Cognitive science-based memory for AI assistants**

[![License](https://img.shields.io/badge/license-Apache%202.0-blue.svg)](LICENSE)
[![Rust](https://img.shields.io/badge/rust-1.75%2B-orange.svg)](https://www.rust-lang.org/)

Rook is a long-term memory system that applies cognitive science principles to help AI assistants remember, forget, and retrieve information like humans do. Unlike simple vector databases, Rook uses spaced repetition, prediction error gating, and spreading activation to create memory that strengthens with use and naturally decays over time.

## Key Features

- **FSRS-6 Memory Dynamics** - Memories track strength using spaced repetition with power-law forgetting curves
- **Smart Ingestion** - Prediction error gating decides whether to Skip, Create, Update, or Supersede based on novelty
- **Graph Memory** - LLM-based entity extraction with spreading activation for associative retrieval
- **Hybrid Retrieval** - Four modes (Quick/Standard/Precise/Cognitive) combining vector search, full-text, and graph traversal
- **Classification** - 10 cognitive categories with LLM auto-classification and key memory tier
- **Consolidation** - Synaptic tagging marks memories for consolidation based on behavioral novelty
- **Intentions & Events** - Proactive triggers and lifecycle webhooks for memory operations
- **Multimodal** - Extract memories from PDF, DOCX, and images (OCR + vision LLM)
- **Production Ready** - PostgreSQL+pgvector backend, Python bindings, MCP server for Claude Code

## Quick Start

### REST API

Start the server and make requests:

```bash
# Start with Docker (coming soon)
docker run -p 8080:8080 ghcr.io/heidornj/rook-server

# Or build from source
cargo run --bin rook-server
```

Configure and add memories:

```bash
# Configure with OpenAI
curl -X POST http://localhost:8080/configure \
  -H "Content-Type: application/json" \
  -d '{
    "llm": {"provider": "openai", "api_key": "sk-..."},
    "embedder": {"provider": "openai", "api_key": "sk-..."},
    "vector_store": {"provider": "qdrant", "url": "http://localhost:6333"}
  }'

# Add a memory
curl -X POST http://localhost:8080/memories \
  -H "Content-Type: application/json" \
  -d '{
    "messages": [
      {"role": "user", "content": "I'm a software engineer who loves Rust"}
    ],
    "user_id": "user123"
  }'

# Search memories
curl -X POST http://localhost:8080/search \
  -H "Content-Type: application/json" \
  -d '{"query": "programming languages", "user_id": "user123"}'
```

### Rust Library

Add to your `Cargo.toml`:

```toml
[dependencies]
rook-core = "0.1"
rook-llm = "0.1"
rook-embeddings = "0.1"
rook-vector-stores = "0.1"
tokio = { version = "1", features = ["full"] }
```

Use in your code:

```rust
use rook_core::memory::{Memory, MemoryBuilder};
use rook_llm::openai::OpenAiLlm;
use rook_embeddings::openai::OpenAiEmbedder;
use rook_vector_stores::qdrant::QdrantVectorStore;

#[tokio::main]
async fn main() -> Result<(), Box<dyn std::error::Error>> {
    // Create providers
    let llm = OpenAiLlm::new("gpt-4-turbo", std::env::var("OPENAI_API_KEY")?)?;
    let embedder = OpenAiEmbedder::new("text-embedding-3-small", std::env::var("OPENAI_API_KEY")?)?;
    let vector_store = QdrantVectorStore::new("http://localhost:6333", "rook", 1536).await?;

    // Build memory instance
    let memory = MemoryBuilder::new()
        .llm(Box::new(llm))
        .embedder(Box::new(embedder))
        .vector_store(Box::new(vector_store))
        .build()?;

    // Add a memory
    let messages = vec![
        Message::user("I'm learning Rust and really enjoying it"),
    ];
    let results = memory.add(&messages, Some("user123"), None, None, None, true).await?;

    // Search memories
    let results = memory.search("programming", Some("user123"), None, None, Some(10)).await?;
    for result in results {
        println!("{}: {}", result.id, result.memory);
    }

    Ok(())
}
```

### Python

Install the package:

```bash
pip install rook-rs
```

Use in your code:

```python
import asyncio
from rook_rs import Memory, MemoryConfig, SearchConfig

async def main():
    # Create memory instance
    config = MemoryConfig(
        llm_provider="openai",
        llm_api_key="sk-...",
        embedder_provider="openai",
        embedder_api_key="sk-...",
        vector_store_provider="qdrant",
        vector_store_url="http://localhost:6333"
    )
    memory = Memory(config)

    # Add a memory
    result = await memory.add(
        messages=[{"role": "user", "content": "I love Python and Rust"}],
        user_id="user123"
    )
    print(f"Added: {result}")

    # Search memories
    results = await memory.search(
        query="programming languages",
        user_id="user123",
        limit=10
    )
    for r in results:
        print(f"{r.id}: {r.memory} (score: {r.score})")

asyncio.run(main())
```

## Installation

### Prerequisites

- Rust 1.75+ (for building from source)
- A vector store (Qdrant, PostgreSQL+pgvector, or others)
- An LLM API key (OpenAI, Anthropic, or others)
- An embedding API key (OpenAI, Cohere, or others)

### From Source

```bash
# Clone the repository
git clone https://github.com/BangRocket/rook.git
cd rook

# Build all crates
cargo build --release

# Run the server
./target/release/rook-server

# Or run tests
cargo test
```

### Python Bindings

```bash
cd crates/rook-python
pip install maturin
maturin develop --release
```

## Configuration

### Environment Variables

| Variable | Default | Description |
|----------|---------|-------------|
| `ROOK_HOST` | `0.0.0.0` | Server host address |
| `ROOK_PORT` | `8080` | Server port |
| `ROOK_API_KEY` | - | API key for authentication |
| `ROOK_REQUIRE_AUTH` | - | Enable API key auth (set any value) |
| `OPENAI_API_KEY` | - | OpenAI API key |
| `ANTHROPIC_API_KEY` | - | Anthropic API key |

### Supported Providers

**LLM Providers:**
- OpenAI (GPT-4, GPT-4 Turbo, GPT-3.5)
- Anthropic (Claude 3, Claude 2)
- Ollama (local models)
- Azure OpenAI
- Groq, Together, DeepSeek, Gemini

**Embedding Providers:**
- OpenAI (text-embedding-3-small/large, ada-002)
- Cohere (embed-english-v3.0)
- Ollama (local embeddings)
- HuggingFace, Vertex AI

**Vector Stores:**
- Qdrant (recommended for development)
- PostgreSQL + pgvector (recommended for production)
- Pinecone, Weaviate, Milvus
- Redis, Elasticsearch, MongoDB
- And more...

**Graph Stores:**
- Embedded (SQLite + petgraph) - default
- Neo4j
- Memgraph

## Architecture

```
rook/
├── crates/
│   ├── rook-core/         # Memory engine, FSRS-6, classification, consolidation
│   ├── rook-llm/          # LLM provider abstractions (OpenAI, Anthropic, etc.)
│   ├── rook-embeddings/   # Embedding provider abstractions
│   ├── rook-vector-stores/# Vector store backends (Qdrant, pgvector, etc.)
│   ├── rook-graph-stores/ # Graph store backends (embedded, Neo4j)
│   ├── rook-rerankers/    # Reranker integrations (Cohere, LLM-based)
│   ├── rook-extractors/   # Document extraction (PDF, DOCX, images)
│   ├── rook-client/       # HTTP client library
│   ├── rook-server/       # Axum REST API server
│   ├── rook-python/       # Python bindings via PyO3
│   └── rook-mcp/          # MCP server for Claude Code
```

### Key Traits

The system is built on trait abstractions for flexibility:

- `Llm` - LLM provider interface
- `Embedder` - Embedding provider interface
- `VectorStore` - Vector storage interface
- `GraphStore` - Graph storage interface
- `Reranker` - Result reranking interface

## Documentation

- [Getting Started](docs/Getting-Started.md) - Installation and first steps
- [REST API Reference](docs/REST-API.md) - Complete HTTP API documentation
- [Rust Library Guide](docs/Rust-Library.md) - Using rook-core directly
- [Python Guide](docs/Python-Bindings.md) - Python bindings usage
- [Configuration](docs/Configuration.md) - All configuration options
- [Concepts](docs/Concepts.md) - Cognitive science background (FSRS, spreading activation)
- [MCP Server](docs/MCP-Server.md) - Claude Code integration
- [Architecture](docs/Architecture.md) - Technical deep dive

## Cognitive Science Background

Rook applies research from cognitive psychology and neuroscience:

### FSRS-6 (Free Spaced Repetition Scheduler)
Memories track stability and difficulty using the FSRS-6 algorithm. Frequently accessed memories strengthen; unused memories naturally decay following power-law forgetting curves.

### Prediction Error Gating
New information is evaluated against existing memories. High prediction error (surprising information) leads to new memory creation; low prediction error (redundant information) is skipped or merged.

### Spreading Activation (ACT-R)
Related memories activate each other through graph connections. When you search for "coffee", memories about "morning routine" and "favorite cafe" also receive activation boosts.

### Synaptic Tagging & Consolidation
Novel events "tag" temporally adjacent memories for consolidation, mimicking how surprising experiences enhance memory formation for surrounding events.

## Contributing

Contributions are welcome! Please see our contributing guidelines.

1. Fork the repository
2. Create a feature branch (`git checkout -b feature/amazing-feature`)
3. Commit your changes (`git commit -m 'Add amazing feature'`)
4. Push to the branch (`git push origin feature/amazing-feature`)
5. Open a Pull Request

## License

Apache 2.0 - see [LICENSE](LICENSE) for details.

## Acknowledgments

- [FSRS](https://github.com/open-spaced-repetition/fsrs-rs) - Spaced repetition algorithm
- [mem0](https://github.com/mem0ai/mem0) - Inspiration for the memory layer concept
- ACT-R cognitive architecture research
- Synaptic tagging and capture research
