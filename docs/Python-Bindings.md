# Python Bindings Guide

Using Rook from Python via the `rook-rs` package.

## Installation

### From Source

```bash
# Clone the repository
git clone https://github.com/heidornj/rook.git
cd rook/crates/rook-python

# Install maturin
pip install maturin

# Build and install
maturin develop --release
```

### From PyPI (when published)

```bash
pip install rook-rs
```

## Quick Start

```python
import asyncio
import os
from rook_rs import Memory, MemoryConfig

async def main():
    # Configure
    config = MemoryConfig(
        llm_provider="openai",
        llm_api_key=os.environ["OPENAI_API_KEY"],
        embedder_provider="openai",
        embedder_api_key=os.environ["OPENAI_API_KEY"],
        vector_store_provider="qdrant",
        vector_store_url="http://localhost:6333"
    )

    memory = Memory(config)

    # Add memories
    results = await memory.add(
        messages=[
            {"role": "user", "content": "My name is Alice and I love Python"}
        ],
        user_id="alice"
    )
    print(f"Added {len(results)} memories")

    # Search
    results = await memory.search(
        query="What programming language?",
        user_id="alice",
        limit=5
    )
    for r in results:
        print(f"- {r.memory} (score: {r.score:.2f})")

asyncio.run(main())
```

## Configuration

### MemoryConfig

```python
from rook_rs import MemoryConfig

config = MemoryConfig(
    # LLM settings
    llm_provider="openai",          # Required: openai, anthropic, ollama, etc.
    llm_model="gpt-4-turbo",        # Optional: defaults vary by provider
    llm_api_key="sk-...",           # Required for cloud providers
    llm_base_url=None,              # Optional: custom endpoint

    # Embedder settings
    embedder_provider="openai",      # Required
    embedder_model="text-embedding-3-small",
    embedder_api_key="sk-...",

    # Vector store settings
    vector_store_provider="qdrant",  # Required
    vector_store_url="http://localhost:6333",
    vector_store_api_key=None,
    collection_name="rook",
    embedding_dims=1536,

    # Graph store (optional)
    graph_store_provider="embedded",
    graph_store_url=None,

    # Reranker (optional)
    reranker_provider=None,
    reranker_model=None,
    reranker_api_key=None,
)
```

### Environment Variables

You can also use environment variables:

```python
import os

os.environ["OPENAI_API_KEY"] = "sk-..."
os.environ["ANTHROPIC_API_KEY"] = "sk-ant-..."

config = MemoryConfig(
    llm_provider="openai",
    llm_api_key=os.environ["OPENAI_API_KEY"],
    # ...
)
```

## Core Operations

### Adding Memories

```python
# From messages
results = await memory.add(
    messages=[
        {"role": "user", "content": "I work at Acme Corp as an engineer"},
        {"role": "assistant", "content": "That's great! What kind of engineering?"}
    ],
    user_id="alice",
    agent_id=None,        # Optional
    run_id=None,          # Optional session ID
    metadata={"source": "chat"},
    infer=True            # Extract facts (default: True)
)

# Each result has:
for r in results:
    print(r.id)        # UUID
    print(r.memory)    # Extracted fact
    print(r.event)     # "ADD", "UPDATE", "DELETE", "NONE"
```

### Searching Memories

```python
results = await memory.search(
    query="Where does Alice work?",
    user_id="alice",
    agent_id=None,
    run_id=None,
    limit=10,
    threshold=0.7,        # Minimum similarity score
    rerank=False          # Use reranker if configured
)

for r in results:
    print(f"ID: {r.id}")
    print(f"Memory: {r.memory}")
    print(f"Score: {r.score}")
    print(f"User: {r.user_id}")
    print(f"Metadata: {r.metadata}")
```

### Getting Memories

```python
# Get a specific memory
item = await memory.get(memory_id="550e8400-...")
print(item.memory)

# Get all memories for a user
items = await memory.get_all(user_id="alice", limit=100)
```

### Updating Memories

```python
updated = await memory.update(
    memory_id="550e8400-...",
    text="Alice is a senior software engineer"
)
print(f"Updated: {updated.memory}")
```

### Deleting Memories

```python
# Delete one
await memory.delete(memory_id="550e8400-...")

# Delete all for a user
await memory.delete_all(user_id="alice")

# Delete with filters
await memory.delete_all(
    user_id="alice",
    agent_id="bot1"
)
```

### Memory History

```python
history = await memory.history(memory_id="550e8400-...")
for version in history:
    print(f"v{version.version}: {version.content}")
    print(f"  Changed: {version.timestamp}")
```

## Advanced Features

### Hybrid Search

```python
from rook_rs import RetrievalMode

# Quick - vector only
results = await memory.hybrid_search(
    query="programming",
    user_id="alice",
    mode=RetrievalMode.QUICK,
    limit=10
)

# Standard - vector + spreading activation
results = await memory.hybrid_search(
    query="programming",
    user_id="alice",
    mode=RetrievalMode.STANDARD,
    limit=10
)

# Cognitive - with FSRS weighting
results = await memory.hybrid_search(
    query="programming",
    user_id="alice",
    mode=RetrievalMode.COGNITIVE,
    limit=10
)
```

### FSRS Operations

```python
from rook_rs import Grade

# Promote memory (was helpful)
await memory.promote(memory_id="...", grade=Grade.GOOD)

# Demote memory (was wrong)
await memory.demote(memory_id="...", grade=Grade.AGAIN)

# Available grades: AGAIN, HARD, GOOD, EASY
```

### Key Memories

```python
# Mark as key memory (always retrieved)
await memory.set_key_memory(memory_id="...", is_key=True)

# Key memories are automatically included in search results
```

### Smart Ingestion

```python
from rook_rs import IngestResult

result = await memory.smart_ingest(
    text="Alice now works at BigCorp",
    user_id="alice"
)

if result.is_created():
    print(f"New: {result.item.memory}")
elif result.is_updated():
    print(f"Updated: {result.item.memory}")
elif result.is_superseded():
    print(f"Replaced old memory")
elif result.is_skipped():
    print(f"Skipped: {result.reason}")
```

## Synchronous API

For non-async contexts:

```python
from rook_rs import Memory, MemoryConfig

config = MemoryConfig(...)
memory = Memory(config)

# Use sync methods
results = memory.add_sync(
    messages=[{"role": "user", "content": "Hello"}],
    user_id="alice"
)

results = memory.search_sync(
    query="greeting",
    user_id="alice"
)
```

## Error Handling

```python
from rook_rs import RookError

try:
    results = await memory.search(query="test", user_id="alice")
except RookError as e:
    print(f"Rook error: {e}")
except Exception as e:
    print(f"Other error: {e}")
```

## Type Hints

The package includes full type hints:

```python
from rook_rs import (
    Memory,
    MemoryConfig,
    MemoryItem,
    SearchResult,
    Grade,
    RetrievalMode,
    IngestResult,
)

def process_memories(results: list[SearchResult]) -> None:
    for r in results:
        print(r.memory)
```

## Integration Examples

### With LangChain

```python
from langchain.memory import BaseMemory
from rook_rs import Memory, MemoryConfig

class RookMemory(BaseMemory):
    def __init__(self, config: MemoryConfig, user_id: str):
        self.memory = Memory(config)
        self.user_id = user_id

    @property
    def memory_variables(self):
        return ["history"]

    def load_memory_variables(self, inputs):
        query = inputs.get("input", "")
        results = self.memory.search_sync(
            query=query,
            user_id=self.user_id,
            limit=5
        )
        history = "\n".join([r.memory for r in results])
        return {"history": history}

    def save_context(self, inputs, outputs):
        messages = [
            {"role": "user", "content": inputs["input"]},
            {"role": "assistant", "content": outputs["output"]}
        ]
        self.memory.add_sync(
            messages=messages,
            user_id=self.user_id
        )
```

### With FastAPI

```python
from fastapi import FastAPI, Depends
from rook_rs import Memory, MemoryConfig

app = FastAPI()

def get_memory():
    config = MemoryConfig(...)
    return Memory(config)

@app.post("/memories")
async def add_memory(
    content: str,
    user_id: str,
    memory: Memory = Depends(get_memory)
):
    results = await memory.add(
        messages=[{"role": "user", "content": content}],
        user_id=user_id
    )
    return {"added": len(results)}

@app.get("/search")
async def search(
    query: str,
    user_id: str,
    memory: Memory = Depends(get_memory)
):
    results = await memory.search(query=query, user_id=user_id)
    return {"results": [r.memory for r in results]}
```

## Performance Tips

1. **Reuse Memory instances** - Don't create new instances per request
2. **Use async methods** - Better throughput than sync methods
3. **Batch operations** - Add multiple messages at once
4. **Limit results** - Use appropriate limits for search
5. **Enable reranking selectively** - Only when precision matters
