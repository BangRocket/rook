# Rook

## What This Is

Rook is a cognitive science-based memory system for AI assistants, written in Rust. It combines proven patterns from the existing mem0 codebase with cognitive science foundations (FSRS-6 spaced repetition, spreading activation, prediction error gating) to create memory that "fades like yours does" — human-like memory with natural decay and surprise-based encoding rather than perfect recall.

## Core Value

Memory retrieval that mirrors human cognition: memories naturally decay, frequently-used memories strengthen, surprising information encodes more strongly, and related memories activate each other through spreading activation.

## Requirements

### Validated

<!-- Existing capabilities from mem0 codebase -->

- ✓ Plugin-based modular architecture with trait-driven abstractions — existing
- ✓ 8-crate workspace (core, llm, embeddings, vector-stores, graph-stores, rerankers, client, server) — existing
- ✓ LLM provider implementations (OpenAI, Anthropic, Ollama) — existing
- ✓ Embedding provider implementations (OpenAI, Ollama) — existing
- ✓ 25+ vector store backends (Qdrant, Pinecone, pgvector, Redis, etc.) — existing
- ✓ Graph store integration (Neo4j) — existing
- ✓ Reranker integration (Cohere) — existing
- ✓ REST API server with Axum — existing
- ✓ Entity scoping (user_id, agent_id, app_id, run_id) — existing
- ✓ History tracking with SQLite — existing
- ✓ Fact extraction via LLM prompts — existing

### Active

<!-- New capabilities to build -->

**Phase 0: Rename**
- [ ] Rename all crates from mem0-* to rook-* (mem0-core → rook-core, etc.)
- [ ] Rename all types from Mem0* to Rook* (Mem0Error → RookError, etc.)
- [ ] Update all environment variables from MEM0_* to ROOK_*
- [ ] Update all internal references, docs, comments
- [ ] Update Cargo.toml package names and dependencies

**Phase 1: Core Engine Enhancements**
- [ ] Memory struct with FSRS-6 fields (stability, difficulty, retrieval_strength, storage_strength)
- [ ] SQLite storage with sqlite-vss for vector similarity
- [ ] Dual-strength memory model (retrieval + storage strength)

**Phase 2: Minimal Graph + FSRS-6 Foundations**
- [ ] Node/edge tables in SQLite for spreading activation
- [ ] FSRS-6 retrievability calculation (power-law forgetting curve)
- [ ] Stability updates on access with grade system
- [ ] Basic spreading activation algorithm
- [ ] promote/demote operations for memory strength

**Phase 3: Smart Ingestion**
- [ ] Prediction error gating (Skip/Create/Update/Supersede decisions)
- [ ] Multi-layer contradiction detection (embedding → keyword → temporal → LLM)
- [ ] smart_ingest API
- [ ] Automatic strength signals (UsedInResponse, UserCorrection, etc.)

**Phase 4: Classification & Categories**
- [ ] Category system with configurable defaults
- [ ] Auto-classification via LLM
- [ ] Key memory tier (always-retrieved memories)
- [ ] Filter DSL with logical operators (And, Or, Not, comparisons)

**Phase 5: Full Graph Memory**
- [ ] Kuzu integration for graph storage
- [ ] LLM-based entity extraction pipeline
- [ ] Entity type classification (person, org, location, project, concept, event)
- [ ] Relationship extraction and typing
- [ ] Hierarchical categories as graph nodes
- [ ] ACT-R base-level activation

**Phase 6: Consolidation & Tagging**
- [ ] Synaptic tagging mechanism
- [ ] Behavioral tagging (novel event boost)
- [ ] consolidate() maintenance operation
- [ ] Multi-timescale memory phases (immediate → early → late)

**Phase 7: Advanced Retrieval**
- [ ] Tantivy full-text search integration
- [ ] Reranking pipeline
- [ ] FSRS-weighted retrieval
- [ ] Hybrid strategy configuration (Quick/Standard/Precise/Cognitive)
- [ ] Deduplication with configurable threshold

**Phase 8: Intentions & Events**
- [ ] Intention storage and trigger conditions
- [ ] Tiered intention checking (keyword bloom → semantic)
- [ ] Memory versioning (audit trail)
- [ ] Event system (created/updated/deleted/accessed)
- [ ] Webhook delivery with retry

**Phase 9: Multimodal**
- [ ] Document extraction (PDF, DOCX)
- [ ] Image extraction (OCR, vision LLM)
- [ ] Audio transcription
- [ ] Cross-modal retrieval

**Phase 10: Production Hardening**
- [ ] PostgreSQL + pgvector backend option
- [ ] Neo4j backend option (alternative to Kuzu)
- [ ] Connection pooling
- [ ] OpenTelemetry metrics
- [ ] Export/import utilities (JSON Lines, Parquet)
- [ ] Python bindings (PyO3)
- [ ] MCP server interface
- [ ] Migration utilities from mem0 data

### Out of Scope

- Real-time streaming memory updates — complexity vs value tradeoff for v1
- Multi-language support for prompts — English-first, internationalize later
- Custom training of embedding models — use existing providers
- Mobile SDKs — Rust library and REST API sufficient for v1

## Context

**Existing Codebase:**
The project builds on a working mem0-based implementation with:
- Trait-driven abstractions (Llm, Embedder, VectorStore, GraphStore, Reranker)
- 8 crates in a Cargo workspace
- Async/await throughout with Tokio runtime
- Factory pattern for provider instantiation
- REST API with Axum

**Cognitive Science Foundations:**
Based on 130+ years of memory research, implemented with concepts from Vestige:
- FSRS-6 (Free Spaced Repetition Scheduler) — models memory through stability, difficulty, and retrievability
- Spreading activation (Collins & Loftus, ACT-R) — related memories activate each other
- Prediction error gating — surprising information encoded more strongly
- Synaptic tagging — novel events boost consolidation of nearby memories
- Dual-strength model — separate retrieval and storage strength dimensions

**Database Compatibility:**
New schema design acceptable; migration path from mem0 data required (not 1:1 schema compatibility).

## Constraints

- **Tech stack**: Rust (Edition 2021), Tokio async runtime, existing trait architecture
- **Embedded-first**: Default to embedded databases (SQLite, sqlite-vss, Kuzu, Tantivy), scale to external when needed
- **Backwards migration**: Must provide migration utilities from existing mem0 databases
- **API compatibility**: REST API should remain compatible where possible to avoid breaking existing integrations

## Key Decisions

| Decision | Rationale | Outcome |
|----------|-----------|---------|
| Full rename mem0 → rook | New project identity, clean break while preserving architecture | — Pending |
| Embedded-first storage | Lower deployment complexity, scale when needed | — Pending |
| FSRS-6 over naive decay | Cognitive science foundation provides better memory dynamics | — Pending |
| Kuzu for graph (Phase 5) | Embedded graph DB, aligns with embedded-first philosophy | — Pending |
| Migration path over schema compat | Allows cleaner schema design for cognitive features | — Pending |

---
*Last updated: 2026-01-28 after initialization*
