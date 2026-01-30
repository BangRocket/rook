# REST API Reference

Complete reference for the Rook HTTP API.

**Base URL:** `http://localhost:8080` (default)

## Authentication

Authentication is optional. Enable it with environment variables:

```bash
ROOK_REQUIRE_AUTH=1
ROOK_API_KEY=your-secret-key
```

When enabled, include the API key in requests:

```bash
curl -H "Authorization: Bearer your-secret-key" http://localhost:8080/health
```

## Endpoints

### Health Check

Check server status and configuration.

```
GET /health
```

**Response:**
```json
{
  "status": "healthy",
  "configured": true,
  "version": "0.1.0"
}
```

---

### Configure Memory

Initialize the memory system with providers.

```
POST /configure
```

**Request Body:**
```json
{
  "llm": {
    "provider": "openai",
    "model": "gpt-4-turbo",
    "api_key": "sk-...",
    "temperature": 0.7,
    "max_tokens": 4096
  },
  "embedder": {
    "provider": "openai",
    "model": "text-embedding-3-small",
    "api_key": "sk-..."
  },
  "vector_store": {
    "provider": "qdrant",
    "url": "http://localhost:6333",
    "collection_name": "rook",
    "embedding_dims": 1536
  },
  "graph_store": {
    "provider": "embedded"
  },
  "reranker": {
    "provider": "cohere",
    "api_key": "..."
  }
}
```

**LLM Providers:** `openai`, `anthropic`, `ollama`, `azure_openai`, `groq`, `together`, `deepseek`, `gemini`

**Embedder Providers:** `openai`, `ollama`, `huggingface`, `cohere`, `vertex_ai`, `azure_openai`

**Vector Store Providers:** `qdrant`, `pgvector`, `pgvector_pooled`, `pinecone`, `weaviate`, `milvus`, `redis`, `elasticsearch`, `mongodb`, `chroma`, `faiss`

**Graph Store Providers:** `embedded`, `neo4j`, `memgraph`

**Reranker Providers:** `cohere`, `llm`, `huggingface`

**Response:**
```json
{
  "message": "Memory configured successfully",
  "configured": true
}
```

---

### Add Memories

Extract and store memories from messages.

```
POST /memories
```

**Request Body:**
```json
{
  "messages": [
    {"role": "user", "content": "My name is Alice"},
    {"role": "assistant", "content": "Nice to meet you, Alice!"}
  ],
  "user_id": "user123",
  "agent_id": "agent456",
  "run_id": "session789",
  "metadata": {"source": "chat"},
  "infer": true
}
```

| Field | Type | Required | Description |
|-------|------|----------|-------------|
| `messages` | array | Yes | Conversation messages |
| `user_id` | string | No | User identifier |
| `agent_id` | string | No | Agent identifier |
| `run_id` | string | No | Session identifier |
| `metadata` | object | No | Custom metadata |
| `infer` | boolean | No | Extract facts (default: true) |

**Response:**
```json
{
  "results": [
    {
      "id": "550e8400-e29b-41d4-a716-446655440000",
      "memory": "The user's name is Alice",
      "event": "ADD"
    }
  ]
}
```

**Event Types:**
- `ADD` - New memory created
- `UPDATE` - Existing memory updated
- `DELETE` - Memory superseded/deleted
- `NONE` - No action (duplicate/skipped)

---

### List Memories

Retrieve all memories with optional filters.

```
GET /memories
```

**Query Parameters:**
| Parameter | Type | Description |
|-----------|------|-------------|
| `user_id` | string | Filter by user |
| `agent_id` | string | Filter by agent |
| `run_id` | string | Filter by session |
| `limit` | integer | Max results (default: 100) |

**Example:**
```bash
curl "http://localhost:8080/memories?user_id=alice&limit=10"
```

**Response:**
```json
{
  "results": [
    {
      "id": "550e8400-e29b-41d4-a716-446655440000",
      "memory": "Alice is a software engineer",
      "score": null,
      "user_id": "alice",
      "agent_id": null,
      "run_id": null,
      "metadata": {},
      "created_at": "2025-01-30T10:00:00Z",
      "updated_at": "2025-01-30T10:00:00Z"
    }
  ]
}
```

---

### Get Memory

Retrieve a specific memory by ID.

```
GET /memories/:id
```

**Response:**
```json
{
  "id": "550e8400-e29b-41d4-a716-446655440000",
  "memory": "Alice is a software engineer",
  "score": null,
  "user_id": "alice",
  "metadata": {}
}
```

**Errors:**
- `404 Not Found` - Memory doesn't exist

---

### Update Memory

Update the content of a memory.

```
PUT /memories/:id
```

**Request Body:**
```json
{
  "text": "Alice is a senior software engineer"
}
```

**Response:**
```json
{
  "id": "550e8400-e29b-41d4-a716-446655440000",
  "memory": "Alice is a senior software engineer",
  "updated_at": "2025-01-30T11:00:00Z"
}
```

---

### Delete Memory

Delete a specific memory.

```
DELETE /memories/:id
```

**Response:**
```json
{
  "message": "Memory deleted successfully"
}
```

---

### Delete All Memories

Delete memories matching filters.

```
DELETE /memories
```

**Request Body:**
```json
{
  "user_id": "alice",
  "agent_id": null,
  "run_id": null
}
```

**Response:**
```json
{
  "message": "All memories deleted successfully"
}
```

---

### Get Memory History

Retrieve version history for a memory.

```
GET /memories/:id/history
```

**Response:**
```json
{
  "history": [
    {
      "version": 1,
      "content": "Alice is a software engineer",
      "timestamp": "2025-01-30T10:00:00Z",
      "event": "ADD"
    },
    {
      "version": 2,
      "content": "Alice is a senior software engineer",
      "timestamp": "2025-01-30T11:00:00Z",
      "event": "UPDATE"
    }
  ]
}
```

---

### Search Memories

Semantic search across memories.

```
POST /search
```

**Request Body:**
```json
{
  "query": "What does Alice do for work?",
  "user_id": "alice",
  "agent_id": null,
  "run_id": null,
  "limit": 10,
  "threshold": 0.7,
  "filters": {},
  "rerank": true
}
```

| Field | Type | Required | Description |
|-------|------|----------|-------------|
| `query` | string | Yes | Search query |
| `user_id` | string | No | Filter by user |
| `agent_id` | string | No | Filter by agent |
| `run_id` | string | No | Filter by session |
| `limit` | integer | No | Max results (default: 10) |
| `threshold` | float | No | Min similarity score |
| `filters` | object | No | Metadata filters |
| `rerank` | boolean | No | Use reranker (default: false) |

**Response:**
```json
{
  "results": [
    {
      "id": "550e8400-e29b-41d4-a716-446655440000",
      "memory": "Alice is a senior software engineer",
      "score": 0.92,
      "user_id": "alice",
      "metadata": {}
    }
  ]
}
```

---

### Reset Memory

Clear all memories.

```
POST /reset
```

**Response:**
```json
{
  "message": "Memory reset successfully"
}
```

---

## Error Responses

All errors follow this format:

```json
{
  "error": {
    "code": "ERROR_CODE",
    "message": "Human-readable message",
    "details": {}
  }
}
```

| Status | Code | Description |
|--------|------|-------------|
| 400 | `BAD_REQUEST` | Invalid input or not configured |
| 401 | `UNAUTHORIZED` | Invalid/missing API key |
| 404 | `NOT_FOUND` | Resource doesn't exist |
| 422 | `VALIDATION_ERROR` | Validation failed |
| 429 | `RATE_LIMIT` | Rate limit exceeded |
| 500 | `INTERNAL_ERROR` | Server error |

---

## Example Workflow

```bash
# 1. Start server
cargo run --bin rook-server

# 2. Configure
curl -X POST http://localhost:8080/configure \
  -H "Content-Type: application/json" \
  -d '{
    "llm": {"provider": "openai", "api_key": "'$OPENAI_API_KEY'"},
    "embedder": {"provider": "openai", "api_key": "'$OPENAI_API_KEY'"},
    "vector_store": {"provider": "qdrant", "url": "http://localhost:6333"}
  }'

# 3. Add memories
curl -X POST http://localhost:8080/memories \
  -H "Content-Type: application/json" \
  -d '{
    "messages": [{"role": "user", "content": "I love Rust programming"}],
    "user_id": "dev1"
  }'

# 4. Search
curl -X POST http://localhost:8080/search \
  -H "Content-Type: application/json" \
  -d '{"query": "programming languages", "user_id": "dev1"}'

# 5. List all
curl "http://localhost:8080/memories?user_id=dev1"
```
