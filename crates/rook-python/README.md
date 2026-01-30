# rook-rs Python Bindings

Python bindings for the Rook memory system - a cognitive memory layer for AI assistants.

## Features

- **Semantic Search**: Find relevant memories using natural language queries
- **Memory Decay**: Memories naturally decay over time using FSRS-6 algorithm
- **Spreading Activation**: Related memories strengthen each other
- **LLM-powered Extraction**: Automatically extract facts from conversations
- **Scoped Storage**: Organize memories by user_id and agent_id

## Installation

### From PyPI (when published)

```bash
pip install rook-rs
```

### From Source

```bash
cd crates/rook-python
pip install maturin
maturin develop  # Development build
maturin build    # Release wheel
```

## Requirements

- Python 3.8+
- Rust toolchain (for building from source)
- OpenAI API key (or compatible LLM/embedding service)
- Qdrant vector database (default backend)

## Environment Setup

```bash
export OPENAI_API_KEY=your-api-key
export QDRANT_URL=http://localhost:6333  # Optional, defaults to localhost
```

## Usage

### Basic Usage

```python
import rook_rs

# Create memory instance (uses environment variables for config)
memory = rook_rs.Memory()

# Add a memory
result = memory.add(
    content="The user prefers dark mode interfaces",
    user_id="user123"
)
print(f"Added {len(result.memories)} memories")

# Search memories
results = memory.search(
    query="user preferences",
    user_id="user123",
    limit=10
)

for r in results:
    print(f"{r.memory} (score: {r.score:.4f})")
```

### With Custom Configuration

```python
import rook_rs

# Custom prompts for fact extraction
memory = rook_rs.Memory({
    "custom_fact_extraction_prompt": "Extract key facts about the user: {content}",
    "custom_update_memory_prompt": "Decide how to update memory: {content}",
})
```

### Memory Operations

```python
# Get specific memory by ID
item = memory.get("memory_id")
if item:
    print(f"Memory: {item.memory}")
    print(f"Created: {item.created_at}")
    print(f"Category: {item.category}")
    print(f"Is Key: {item.is_key}")
    print(f"Metadata: {item.metadata}")

# Update memory content
updated = memory.update("memory_id", "Updated memory content")

# Delete specific memory
memory.delete("memory_id")

# Delete all memories for a user
memory.delete_all(user_id="user123")

# Get all memories for a user
items = memory.get_all(user_id="user123", limit=100)

# Reset everything (danger!)
memory.reset()
```

### Using Metadata

```python
# Add memory with custom metadata
result = memory.add(
    content="User completed onboarding",
    user_id="user123",
    metadata={
        "source": "onboarding_flow",
        "step": 5,
        "completed": True
    }
)

# Metadata is accessible on results
for mem in result.memories:
    print(mem.metadata)
```

### Scoped Memories

```python
# Memories can be scoped by user_id and/or agent_id
memory.add(
    content="User preference for agent A",
    user_id="user123",
    agent_id="agent_a"
)

# Search only within a specific scope
results = memory.search(
    query="preferences",
    user_id="user123",
    agent_id="agent_a"
)
```

## API Reference

### Memory Class

```python
class Memory:
    def __init__(self, config: dict = None): ...
    def add(self, content: str, user_id: str = None, agent_id: str = None,
            metadata: dict = None, infer: bool = True) -> AddResult: ...
    def search(self, query: str, user_id: str = None, agent_id: str = None,
               limit: int = 10, threshold: float = None) -> list[SearchResult]: ...
    def get(self, memory_id: str) -> MemoryItem | None: ...
    def delete(self, memory_id: str) -> None: ...
    def delete_all(self, user_id: str = None, agent_id: str = None) -> None: ...
    def get_all(self, user_id: str = None, agent_id: str = None,
                limit: int = None) -> list[MemoryItem]: ...
    def update(self, memory_id: str, content: str) -> MemoryItem: ...
    def reset(self) -> None: ...
```

### MemoryItem Class

```python
class MemoryItem:
    id: str              # Unique memory identifier
    memory: str          # Memory content
    hash: str | None     # MD5 hash of content
    score: float | None  # Search relevance score
    metadata: dict       # Custom metadata
    created_at: str | None
    updated_at: str | None
    category: str | None # Assigned category
    is_key: bool         # Whether this is a key memory
```

### SearchResult Class

```python
class SearchResult:
    id: str           # Memory identifier
    memory: str       # Memory content
    score: float      # Relevance score (0-1)
    metadata: dict    # Custom metadata
```

### AddResult Class

```python
class AddResult:
    memories: list[MemoryItem]  # Created/updated memories
```

## Building

### Development Build

```bash
cd crates/rook-python
pip install maturin
maturin develop
```

### Release Build

```bash
cd crates/rook-python
maturin build --release
pip install target/wheels/rook_rs-*.whl
```

### Running Tests

```bash
# After maturin develop
pytest tests/
```

## Architecture

The Python bindings use PyO3 to expose Rook's Rust core to Python:

- **Async Bridge**: Rust async operations are bridged to sync Python via `tokio::runtime::Runtime::block_on()`
- **GIL Release**: The Python GIL is released during Rust operations via `py.allow_threads()`
- **Type Conversion**: Python dicts are converted to Rust `HashMap<String, serde_json::Value>`
- **Memory Safety**: Rust's ownership system ensures memory safety at the FFI boundary

## License

Apache-2.0
