# rook-rerankers

Reranker implementations for Rook.

## Supported Providers

- Cohere
- LLM-based reranking

## Usage

```rust
use rook_rerankers::cohere::CohereReranker;

let reranker = CohereReranker::new(api_key)?;
```

See the [main repository](https://github.com/BangRocket/rook) for full documentation.

## License

Apache-2.0
