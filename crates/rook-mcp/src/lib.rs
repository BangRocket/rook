//! MCP server for Rook memory system.
//!
//! Provides MCP tools for memory operations, enabling Claude Code
//! and other MCP clients to use Rook as persistent memory.
//!
//! # Tools
//!
//! - `memory_add` - Add a new memory to the store
//! - `memory_search` - Search memories by semantic similarity
//! - `memory_get` - Get a specific memory by ID
//! - `memory_delete` - Delete a memory by ID
//!
//! # Configuration
//!
//! The server reads configuration from environment variables:
//!
//! - `ROOK_DATA_DIR` - Directory for data storage (default: ~/.rook)
//! - `OPENAI_API_KEY` - API key for embeddings and LLM
//!
//! # Usage with Claude Code
//!
//! Add to your `claude_desktop_config.json`:
//!
//! ```json
//! {
//!   "mcpServers": {
//!     "rook": {
//!       "command": "/path/to/rook-mcp"
//!     }
//!   }
//! }
//! ```

pub mod server;
pub mod tools;

pub use server::MemoryServer;
