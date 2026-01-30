# rook-embeddings

Embedding provider implementations for Rook.

## Supported Providers

- OpenAI (text-embedding-3-small/large, ada-002)
- Ollama (local embeddings)

## Usage

```rust
use rook_embeddings::openai::OpenAiEmbedder;

let embedder = OpenAiEmbedder::new("text-embedding-3-small", api_key)?;
```

See the [main repository](https://github.com/BangRocket/rook) for full documentation.

## License

Apache-2.0
