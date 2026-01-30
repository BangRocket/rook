# Configuration Reference

Complete configuration options for Rook.

## Environment Variables

### Server Configuration

| Variable | Default | Description |
|----------|---------|-------------|
| `ROOK_HOST` | `0.0.0.0` | Server bind address |
| `ROOK_PORT` | `8080` | Server port |
| `ROOK_REQUIRE_AUTH` | (unset) | Enable API key authentication (set any value) |
| `ROOK_API_KEY` | (empty) | API key for authentication |
| `RUST_LOG` | `info` | Log level (trace, debug, info, warn, error) |

### Feature Toggles

| Variable | Default | Description |
|----------|---------|-------------|
| `ROOK_DISABLE_GRAPH` | (unset) | Disable graph memory features |
| `ROOK_DISABLE_CLASSIFICATION` | (unset) | Disable auto-classification |
| `ROOK_DISABLE_CONSOLIDATION` | (unset) | Disable background consolidation |
| `ROOK_DISABLE_INTENTIONS` | (unset) | Disable intention checking |

### Provider API Keys

| Variable | Provider |
|----------|----------|
| `OPENAI_API_KEY` | OpenAI LLM and embeddings |
| `ANTHROPIC_API_KEY` | Anthropic Claude models |
| `COHERE_API_KEY` | Cohere embeddings and reranking |
| `AZURE_OPENAI_API_KEY` | Azure OpenAI |
| `AZURE_OPENAI_ENDPOINT` | Azure OpenAI endpoint |
| `GROQ_API_KEY` | Groq |
| `TOGETHER_API_KEY` | Together AI |
| `DEEPSEEK_API_KEY` | DeepSeek |
| `GEMINI_API_KEY` | Google Gemini |
| `HUGGINGFACE_API_KEY` | HuggingFace |
| `PINECONE_API_KEY` | Pinecone vector store |
| `QDRANT_API_KEY` | Qdrant Cloud |
| `WEAVIATE_API_KEY` | Weaviate Cloud |

## Provider Configuration

### LLM Providers

#### OpenAI

```json
{
  "llm": {
    "provider": "openai",
    "model": "gpt-4-turbo",
    "api_key": "sk-...",
    "base_url": null,
    "temperature": 0.7,
    "max_tokens": 4096
  }
}
```

**Models:** `gpt-4-turbo`, `gpt-4`, `gpt-4o`, `gpt-3.5-turbo`

#### Anthropic

```json
{
  "llm": {
    "provider": "anthropic",
    "model": "claude-3-opus-20240229",
    "api_key": "sk-ant-..."
  }
}
```

**Models:** `claude-3-opus-20240229`, `claude-3-sonnet-20240229`, `claude-3-haiku-20240307`

#### Ollama (Local)

```json
{
  "llm": {
    "provider": "ollama",
    "model": "llama2",
    "base_url": "http://localhost:11434"
  }
}
```

**Models:** Any model available in your Ollama installation

#### Azure OpenAI

```json
{
  "llm": {
    "provider": "azure_openai",
    "model": "gpt-4",
    "api_key": "...",
    "base_url": "https://your-resource.openai.azure.com"
  }
}
```

### Embedding Providers

#### OpenAI

```json
{
  "embedder": {
    "provider": "openai",
    "model": "text-embedding-3-small",
    "api_key": "sk-..."
  }
}
```

**Models:**
- `text-embedding-3-small` (1536 dims) - Recommended
- `text-embedding-3-large` (3072 dims)
- `text-embedding-ada-002` (1536 dims)

#### Cohere

```json
{
  "embedder": {
    "provider": "cohere",
    "model": "embed-english-v3.0",
    "api_key": "..."
  }
}
```

#### Ollama

```json
{
  "embedder": {
    "provider": "ollama",
    "model": "nomic-embed-text",
    "base_url": "http://localhost:11434"
  }
}
```

### Vector Store Providers

#### Qdrant

```json
{
  "vector_store": {
    "provider": "qdrant",
    "url": "http://localhost:6333",
    "api_key": null,
    "collection_name": "rook",
    "embedding_dims": 1536
  }
}
```

**Setup:**
```bash
docker run -p 6333:6333 -p 6334:6334 qdrant/qdrant
```

#### PostgreSQL + pgvector

```json
{
  "vector_store": {
    "provider": "pgvector",
    "url": "postgresql://user:pass@localhost/rook",
    "collection_name": "memories",
    "embedding_dims": 1536
  }
}
```

**Setup:**
```sql
CREATE EXTENSION vector;
```

#### Pooled PostgreSQL

For production with connection pooling:

```json
{
  "vector_store": {
    "provider": "pgvector_pooled",
    "url": "postgresql://user:pass@localhost/rook",
    "collection_name": "memories",
    "embedding_dims": 1536,
    "pool_size": 16
  }
}
```

#### Pinecone

```json
{
  "vector_store": {
    "provider": "pinecone",
    "api_key": "...",
    "environment": "us-east-1-aws",
    "index_name": "rook"
  }
}
```

#### Weaviate

```json
{
  "vector_store": {
    "provider": "weaviate",
    "url": "http://localhost:8080",
    "api_key": null,
    "collection_name": "Memory"
  }
}
```

#### Milvus

```json
{
  "vector_store": {
    "provider": "milvus",
    "url": "http://localhost:19530",
    "collection_name": "rook"
  }
}
```

### Graph Store Providers

#### Embedded (Default)

```json
{
  "graph_store": {
    "provider": "embedded",
    "path": "./rook_graph.db"
  }
}
```

Uses SQLite + petgraph for zero-dependency graph storage.

#### Neo4j

```json
{
  "graph_store": {
    "provider": "neo4j",
    "url": "bolt://localhost:7687",
    "username": "neo4j",
    "password": "password"
  }
}
```

### Reranker Providers

#### Cohere

```json
{
  "reranker": {
    "provider": "cohere",
    "model": "rerank-english-v2.0",
    "api_key": "..."
  }
}
```

#### LLM-based

```json
{
  "reranker": {
    "provider": "llm"
  }
}
```

Uses the configured LLM for reranking.

## FSRS Configuration

Memory strength parameters:

| Parameter | Default | Description |
|-----------|---------|-------------|
| `fsrs_decay_rate` | 0.9 | Power-law decay exponent |
| `fsrs_stability_base` | 1.0 | Initial stability for new memories |
| `fsrs_difficulty_default` | 0.3 | Default difficulty (0-1) |
| `fsrs_forgetting_threshold` | 0.1 | Archive when retrievability below this |

## Classification Configuration

| Parameter | Default | Description |
|-----------|---------|-------------|
| `classification_enabled` | true | Enable auto-classification |
| `key_memory_cap` | 15 | Max key memories per search |

**Categories:**
- Personal, Preference, Fact, Belief, Skill
- Experience, Social, Work, Health, Other

## Consolidation Configuration

| Parameter | Default | Description |
|-----------|---------|-------------|
| `consolidation_interval` | 3600 | Seconds between consolidation runs |
| `synaptic_tag_decay` | 3600 | Tag decay time constant (seconds) |
| `behavioral_window_before` | 1800 | Novelty boost window before (30 min) |
| `behavioral_window_after` | 7200 | Novelty boost window after (2 hours) |

## Intention Configuration

| Parameter | Default | Description |
|-----------|---------|-------------|
| `intention_check_interval` | 10 | Check every N messages |
| `keyword_bloom_fpr` | 0.001 | Bloom filter false positive rate |
| `semantic_threshold` | 0.75 | Topic match threshold |

## Event Configuration

| Parameter | Default | Description |
|-----------|---------|-------------|
| `events_enabled` | true | Emit memory lifecycle events |
| `webhook_timeout` | 30 | Webhook timeout (seconds) |
| `webhook_retries` | 3 | Retry count for failed webhooks |

## Retrieval Configuration

| Parameter | Default | Description |
|-----------|---------|-------------|
| `default_retrieval_mode` | Standard | Default search mode |
| `spreading_decay` | 0.5 | Activation decay per hop |
| `spreading_threshold` | 0.1 | Minimum activation to propagate |
| `spreading_max_depth` | 3 | Maximum propagation depth |
| `dedup_threshold` | 0.95 | Similarity threshold for deduplication |

## Example Configurations

### Development (Minimal)

```json
{
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
}
```

### Production (Full)

```json
{
  "llm": {
    "provider": "openai",
    "model": "gpt-4-turbo",
    "api_key": "sk-...",
    "temperature": 0.3
  },
  "embedder": {
    "provider": "openai",
    "model": "text-embedding-3-large",
    "api_key": "sk-..."
  },
  "vector_store": {
    "provider": "pgvector_pooled",
    "url": "postgresql://user:pass@db.example.com/rook",
    "embedding_dims": 3072,
    "pool_size": 32
  },
  "graph_store": {
    "provider": "neo4j",
    "url": "bolt://neo4j.example.com:7687",
    "username": "neo4j",
    "password": "..."
  },
  "reranker": {
    "provider": "cohere",
    "api_key": "..."
  }
}
```

### Local/Offline

```json
{
  "llm": {
    "provider": "ollama",
    "model": "llama2",
    "base_url": "http://localhost:11434"
  },
  "embedder": {
    "provider": "ollama",
    "model": "nomic-embed-text",
    "base_url": "http://localhost:11434"
  },
  "vector_store": {
    "provider": "qdrant",
    "url": "http://localhost:6333"
  },
  "graph_store": {
    "provider": "embedded"
  }
}
```
