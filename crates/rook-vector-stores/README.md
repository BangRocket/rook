# rook-vector-stores

Vector store implementations for Rook.

## Supported Backends

- **Native Rust**: Qdrant, Redis, Elasticsearch, OpenSearch, MongoDB
- **SQL-based**: pgvector, MySQL, SQLite, sqlite-vec
- **REST API**: Pinecone, Weaviate, Chroma, Milvus, Upstash
- **Cloud**: Azure AI Search, Vertex AI, Supabase
- **Embedded**: LanceDB, DuckDB

## Usage

```rust
use rook_vector_stores::qdrant::QdrantVectorStore;

let store = QdrantVectorStore::new("http://localhost:6333", "collection", 1536).await?;
```

See the [main repository](https://github.com/BangRocket/rook) for full documentation.

## License

Apache-2.0
