# rook-client

Client library for the Rook hosted API.

## Usage

```rust
use rook_client::RookClient;

let client = RookClient::new("https://api.rook.ai", api_key)?;
let results = client.search("query", user_id).await?;
```

See the [main repository](https://github.com/BangRocket/rook) for full documentation.

## License

Apache-2.0
