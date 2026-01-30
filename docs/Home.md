# Rook Documentation

Welcome to the Rook documentation. Rook is a cognitive science-based memory system for AI assistants that applies principles from spaced repetition, prediction error gating, and spreading activation to create human-like memory behavior.

## Quick Navigation

| Guide | Description |
|-------|-------------|
| [Getting Started](Getting-Started.md) | Installation, prerequisites, and first steps |
| [REST API](REST-API.md) | Complete HTTP API reference |
| [Rust Library](Rust-Library.md) | Using rook-core directly in Rust |
| [Python Bindings](Python-Bindings.md) | Python usage guide |
| [Configuration](Configuration.md) | Environment variables and provider setup |
| [Concepts](Concepts.md) | Cognitive science background |
| [MCP Server](MCP-Server.md) | Claude Code integration |
| [Architecture](Architecture.md) | Technical deep dive |

## What is Rook?

Rook is a long-term memory layer for AI assistants that goes beyond simple vector databases. Instead of treating all information equally, Rook applies cognitive science principles:

- **Memories strengthen with use** - Frequently accessed memories become more stable
- **Memories naturally decay** - Unused memories fade following power-law forgetting curves
- **Surprising information is prioritized** - Prediction error gating filters redundant information
- **Related memories activate each other** - Spreading activation enables associative retrieval

## Core Concepts

### Memory Lifecycle

```
Input → Prediction Error Check → Store/Skip/Update/Supersede
                                      ↓
                              FSRS Strength Tracking
                                      ↓
                              Consolidation (if tagged)
                                      ↓
                              Retrieval (with activation spreading)
```

### Usage Methods

Rook can be used in three ways:

1. **REST API** - HTTP endpoints for any language/platform
2. **Rust Library** - Direct integration via `rook-core`
3. **Python Bindings** - Native Python package via PyO3

## Feature Overview

| Feature | Description |
|---------|-------------|
| FSRS-6 | Spaced repetition memory dynamics |
| Smart Ingestion | Prediction error gating |
| Graph Memory | Entity extraction + spreading activation |
| Hybrid Search | Vector + full-text + graph retrieval |
| Classification | 10 cognitive categories |
| Consolidation | Synaptic tagging |
| Intentions | Proactive memory triggers |
| Events | Lifecycle webhooks |
| Multimodal | PDF, DOCX, image extraction |

## Supported Providers

**LLMs:** OpenAI, Anthropic, Ollama, Azure OpenAI, Groq, Together, DeepSeek, Gemini

**Embeddings:** OpenAI, Cohere, Ollama, HuggingFace, Vertex AI

**Vector Stores:** Qdrant, PostgreSQL+pgvector, Pinecone, Weaviate, Milvus, Redis, Elasticsearch, MongoDB

**Graph Stores:** Embedded (SQLite+petgraph), Neo4j, Memgraph

## Getting Help

- [GitHub Issues](https://github.com/heidornj/rook/issues) - Bug reports and feature requests
- [Discussions](https://github.com/heidornj/rook/discussions) - Questions and community

## License

Rook is licensed under Apache 2.0. See the [LICENSE](https://github.com/heidornj/rook/blob/main/LICENSE) file for details.
