# MCP Server Guide

Integrating Rook with Claude Code via the Model Context Protocol (MCP).

## Overview

Rook provides an MCP server that allows Claude Code to store and retrieve long-term memories across conversations. This enables:

- Remembering user preferences across sessions
- Learning from past interactions
- Building a persistent knowledge base about the user

## Installation

### Prerequisites

- Rust 1.75+
- Claude Code installed

### Building the MCP Server

```bash
# Clone Rook
git clone https://github.com/heidornj/rook.git
cd rook

# Build the MCP server
cargo build --release --bin rook-mcp

# The binary is at: target/release/rook-mcp
```

## Configuration

### Claude Code Settings

Add Rook to your Claude Code MCP servers configuration:

**macOS:** `~/Library/Application Support/Claude/claude_desktop_config.json`
**Linux:** `~/.config/Claude/claude_desktop_config.json`
**Windows:** `%APPDATA%\Claude\claude_desktop_config.json`

```json
{
  "mcpServers": {
    "rook": {
      "command": "/path/to/rook-mcp",
      "env": {
        "OPENAI_API_KEY": "sk-...",
        "ROOK_VECTOR_STORE_URL": "http://localhost:6333"
      }
    }
  }
}
```

### Environment Variables

The MCP server accepts these environment variables:

| Variable | Required | Description |
|----------|----------|-------------|
| `OPENAI_API_KEY` | Yes* | OpenAI API key for LLM and embeddings |
| `ANTHROPIC_API_KEY` | Yes* | Anthropic API key (alternative to OpenAI) |
| `ROOK_LLM_PROVIDER` | No | LLM provider (default: openai) |
| `ROOK_LLM_MODEL` | No | LLM model (default: gpt-4-turbo) |
| `ROOK_EMBEDDER_PROVIDER` | No | Embedder provider (default: openai) |
| `ROOK_EMBEDDER_MODEL` | No | Embedding model (default: text-embedding-3-small) |
| `ROOK_VECTOR_STORE_PROVIDER` | No | Vector store (default: qdrant) |
| `ROOK_VECTOR_STORE_URL` | No | Vector store URL (default: http://localhost:6333) |
| `ROOK_COLLECTION_NAME` | No | Collection name (default: rook_mcp) |

*At least one LLM API key required

### Vector Store Setup

Start Qdrant (or your preferred vector store):

```bash
docker run -p 6333:6333 qdrant/qdrant
```

## Available Tools

The MCP server exposes these tools to Claude:

### `rook_add`

Store new memories from the conversation.

**Parameters:**
- `content` (string, required): The information to remember
- `user_id` (string, optional): User identifier

**Example use by Claude:**
```
I'll remember that you prefer dark mode for coding.
[Uses rook_add with content="User prefers dark mode for coding"]
```

### `rook_search`

Search for relevant memories.

**Parameters:**
- `query` (string, required): What to search for
- `user_id` (string, optional): User identifier
- `limit` (number, optional): Max results (default: 10)

**Example use by Claude:**
```
Let me check what I know about your coding preferences.
[Uses rook_search with query="coding preferences"]
```

### `rook_get`

Retrieve a specific memory by ID.

**Parameters:**
- `memory_id` (string, required): Memory UUID

### `rook_update`

Update an existing memory.

**Parameters:**
- `memory_id` (string, required): Memory UUID
- `content` (string, required): New content

### `rook_delete`

Delete a memory.

**Parameters:**
- `memory_id` (string, required): Memory UUID

### `rook_list`

List all memories for a user.

**Parameters:**
- `user_id` (string, optional): User identifier
- `limit` (number, optional): Max results

## Usage Examples

### Basic Memory Storage

When Claude learns something about you:

```
User: I'm a software engineer who specializes in Rust
Claude: I'll remember that! [stores memory]

[Later conversation]
User: What programming help can you give me?
Claude: [searches memories] Since you specialize in Rust, I can help with...
```

### Preferences

```
User: I prefer concise explanations without too much detail
Claude: Got it, I'll keep explanations brief. [stores preference]

[Later]
Claude: [retrieves preference] Here's a concise summary...
```

### Project Context

```
User: I'm working on a project called "Phoenix" - it's a web app in Rust
Claude: I'll remember about Phoenix. [stores context]

[Later]
User: How should I structure the database?
Claude: [searches for Phoenix] For your Phoenix web app, I'd suggest...
```

## Prompting Tips

Claude will use Rook more effectively with these prompting strategies:

### Explicit Storage
```
"Remember that I prefer TypeScript over JavaScript"
"Save this for later: my API key format is..."
"Keep in mind that I work at Acme Corp"
```

### Explicit Retrieval
```
"What do you remember about my preferences?"
"Recall what we discussed about the Phoenix project"
"What do you know about my work?"
```

### Automatic (Claude decides)
Claude will often store and retrieve memories automatically when:
- You share personal/preference information
- You provide context about projects
- You ask questions that might relate to past conversations

## Troubleshooting

### "MCP server not responding"

1. Check the binary path is correct in config
2. Verify the vector store is running:
   ```bash
   curl http://localhost:6333/health
   ```
3. Check logs:
   ```bash
   RUST_LOG=debug /path/to/rook-mcp
   ```

### "No memories found"

- Memories are scoped by `user_id` - ensure consistent user IDs
- Check the vector store has data:
  ```bash
  curl http://localhost:6333/collections/rook_mcp
  ```

### "API key errors"

Ensure environment variables are set in the MCP config:
```json
{
  "mcpServers": {
    "rook": {
      "command": "/path/to/rook-mcp",
      "env": {
        "OPENAI_API_KEY": "sk-actual-key-here"
      }
    }
  }
}
```

## Advanced Configuration

### Using Ollama (Local/Offline)

```json
{
  "mcpServers": {
    "rook": {
      "command": "/path/to/rook-mcp",
      "env": {
        "ROOK_LLM_PROVIDER": "ollama",
        "ROOK_LLM_MODEL": "llama2",
        "ROOK_EMBEDDER_PROVIDER": "ollama",
        "ROOK_EMBEDDER_MODEL": "nomic-embed-text",
        "OLLAMA_HOST": "http://localhost:11434"
      }
    }
  }
}
```

### Using PostgreSQL

```json
{
  "mcpServers": {
    "rook": {
      "command": "/path/to/rook-mcp",
      "env": {
        "OPENAI_API_KEY": "sk-...",
        "ROOK_VECTOR_STORE_PROVIDER": "pgvector",
        "ROOK_VECTOR_STORE_URL": "postgresql://user:pass@localhost/rook"
      }
    }
  }
}
```

## Privacy Considerations

- Memories are stored locally (in your vector store)
- No data is sent to Rook servers
- You control what Claude remembers
- Use `rook_delete` or `rook_list` + delete to remove memories
- Consider separate collections for different contexts
