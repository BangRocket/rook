//! Python bindings for Rook memory system.
//!
//! This crate provides Python bindings via PyO3 for the Rook memory layer.
//! It exposes the core Memory API with Pythonic types and error handling.

use pyo3::prelude::*;

mod memory;
mod types;

use memory::Memory;
use types::{AddResult, MemoryItem, SearchResult};

/// Rook memory system for AI assistants.
///
/// This module provides Python bindings for Rook, a cognitive memory layer
/// that supports memory decay, strengthening with use, and intelligent
/// retrieval via spreading activation.
#[pymodule]
fn rook_rs(m: &Bound<'_, PyModule>) -> PyResult<()> {
    m.add_class::<Memory>()?;
    m.add_class::<MemoryItem>()?;
    m.add_class::<SearchResult>()?;
    m.add_class::<AddResult>()?;
    Ok(())
}
