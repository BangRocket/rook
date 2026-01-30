# rook-mcp

Model Context Protocol (MCP) server for Rook memory system.

Enables Claude Code and other MCP-compatible clients to access Rook memory.

## Running

```bash
cargo run --bin rook-mcp
```

## Configuration

Add to your Claude Code MCP settings:

```json
{
  "mcpServers": {
    "rook": {
      "command": "rook-mcp"
    }
  }
}
```

See the [main repository](https://github.com/BangRocket/rook) for full documentation.

## License

Apache-2.0
