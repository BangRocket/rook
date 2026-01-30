# rook-graph-stores

Graph store implementations for Rook.

## Supported Backends

- Embedded (SQLite + petgraph) - default
- Neo4j
- Memgraph
- Kuzu
- Neptune

## Usage

```rust
use rook_graph_stores::embedded::EmbeddedGraphStore;

let store = EmbeddedGraphStore::new("./graph.db")?;
```

See the [main repository](https://github.com/BangRocket/rook) for full documentation.

## License

Apache-2.0
