# rook-llm

LLM provider implementations for Rook.

## Supported Providers

- OpenAI (GPT-4, GPT-4 Turbo, GPT-3.5)
- Anthropic (Claude 3, Claude 2)
- Ollama (local models)

## Usage

```rust
use rook_llm::openai::OpenAiLlm;

let llm = OpenAiLlm::new("gpt-4-turbo", api_key)?;
```

See the [main repository](https://github.com/BangRocket/rook) for full documentation.

## License

Apache-2.0
