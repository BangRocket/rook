//! Memory module - core memory implementation.

mod history;
mod json_parser;
mod main;
mod prompts;
mod session;
mod telemetry;

pub use history::{HistoryEvent, HistoryRecord, HistoryStore};
pub use json_parser::{extract_json, parse_facts, parse_memory_actions, remove_code_blocks};
pub use main::Memory;
pub use prompts::*;
pub use session::{build_filters_and_metadata, SessionScope};
pub use telemetry::{process_telemetry_filters, Telemetry};
