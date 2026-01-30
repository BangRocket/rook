# rook-server

REST API server for Rook.

## Running

```bash
# Build and run
cargo run --bin rook-server

# Or with Docker (coming soon)
docker run -p 8080:8080 ghcr.io/bangrocket/rook-server
```

## Environment Variables

| Variable | Default | Description |
|----------|---------|-------------|
| `ROOK_HOST` | `0.0.0.0` | Server host address |
| `ROOK_PORT` | `8080` | Server port |
| `ROOK_API_KEY` | - | API key for authentication |

See the [main repository](https://github.com/BangRocket/rook) for full documentation.

## License

Apache-2.0
