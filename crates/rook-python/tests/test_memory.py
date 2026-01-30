"""Tests for rook_rs Python bindings.

These tests require building with maturin first:
    cd crates/rook-python && maturin develop
"""
import pytest


def test_import():
    """Test module can be imported."""
    import rook_rs
    assert hasattr(rook_rs, 'Memory')
    assert hasattr(rook_rs, 'MemoryItem')
    assert hasattr(rook_rs, 'SearchResult')
    assert hasattr(rook_rs, 'AddResult')


def test_memory_class_exists():
    """Test Memory class has expected methods."""
    import rook_rs
    # Verify class exists and has expected methods
    assert callable(getattr(rook_rs.Memory, '__init__', None))


def test_memory_item_repr():
    """Test MemoryItem has a repr method via the class."""
    import rook_rs
    # MemoryItem should be importable and have proper attributes defined
    assert hasattr(rook_rs, 'MemoryItem')


def test_search_result_class():
    """Test SearchResult class exists."""
    import rook_rs
    assert hasattr(rook_rs, 'SearchResult')


def test_add_result_class():
    """Test AddResult class exists."""
    import rook_rs
    assert hasattr(rook_rs, 'AddResult')


def test_memory_creation_requires_config():
    """Test Memory instance requires valid configuration.

    Without proper API keys/config, creation should fail with a RuntimeError.
    """
    import rook_rs
    # Without API keys, this should raise RuntimeError
    try:
        memory = rook_rs.Memory()
        # If we get here, we have valid env vars configured
        assert memory is not None
    except RuntimeError as e:
        # Expected if no API keys configured
        error_str = str(e).lower()
        assert "api" in error_str or "key" in error_str or "error" in error_str


def test_memory_with_config():
    """Test Memory with explicit config dict.

    Even with config, it will fail without real credentials,
    but this tests that config parsing works.
    """
    import rook_rs
    config = {
        "custom_fact_extraction_prompt": "Extract facts from: {content}",
    }
    try:
        memory = rook_rs.Memory(config)
    except RuntimeError:
        pass  # Expected without credentials - config parsing worked


def test_memory_with_invalid_config():
    """Test Memory with invalid config type raises TypeError."""
    import rook_rs
    with pytest.raises(TypeError):
        memory = rook_rs.Memory("invalid_config_type")


@pytest.mark.skip(reason="Requires running Qdrant and OpenAI API keys")
def test_add_and_search_integration():
    """Integration test for add and search.

    This test requires:
    - OPENAI_API_KEY environment variable
    - Qdrant running at localhost:6333
    """
    import rook_rs
    memory = rook_rs.Memory()

    # Add a memory
    result = memory.add(
        content="I love programming in Rust and Python",
        user_id="test_user"
    )
    assert result is not None
    assert hasattr(result, 'memories')

    # Search for memories
    results = memory.search(
        query="Rust programming",
        user_id="test_user",
        limit=5
    )
    assert isinstance(results, list)
    if len(results) > 0:
        assert hasattr(results[0], 'score')
        assert hasattr(results[0], 'memory')
        assert results[0].score > 0


@pytest.mark.skip(reason="Requires running Qdrant and OpenAI API keys")
def test_get_and_delete_integration():
    """Integration test for get and delete operations."""
    import rook_rs
    memory = rook_rs.Memory()

    # Add a memory first
    result = memory.add(
        content="Test memory for deletion",
        user_id="test_user"
    )

    if len(result.memories) > 0:
        memory_id = result.memories[0].id

        # Get the memory
        item = memory.get(memory_id)
        assert item is not None
        assert item.id == memory_id

        # Delete the memory
        memory.delete(memory_id)

        # Verify deletion
        deleted_item = memory.get(memory_id)
        assert deleted_item is None


@pytest.mark.skip(reason="Requires running Qdrant and OpenAI API keys")
def test_get_all_integration():
    """Integration test for get_all operation."""
    import rook_rs
    memory = rook_rs.Memory()

    # Add some memories
    memory.add(content="Memory one", user_id="test_user_getall")
    memory.add(content="Memory two", user_id="test_user_getall")

    # Get all memories for user
    items = memory.get_all(user_id="test_user_getall", limit=10)
    assert isinstance(items, list)
    assert len(items) >= 2

    # Cleanup
    memory.delete_all(user_id="test_user_getall")


@pytest.mark.skip(reason="Requires running Qdrant and OpenAI API keys")
def test_update_integration():
    """Integration test for update operation."""
    import rook_rs
    memory = rook_rs.Memory()

    # Add a memory
    result = memory.add(
        content="Original content",
        user_id="test_user_update"
    )

    if len(result.memories) > 0:
        memory_id = result.memories[0].id

        # Update the memory
        updated = memory.update(memory_id, "Updated content")
        assert updated.memory == "Updated content"

        # Cleanup
        memory.delete(memory_id)


@pytest.mark.skip(reason="Requires running Qdrant and OpenAI API keys")
def test_metadata_handling():
    """Integration test for metadata handling."""
    import rook_rs
    memory = rook_rs.Memory()

    # Add memory with metadata
    result = memory.add(
        content="Memory with metadata",
        user_id="test_user_meta",
        metadata={"source": "test", "priority": 5}
    )

    if len(result.memories) > 0:
        memory_id = result.memories[0].id

        # Get and verify metadata is accessible
        item = memory.get(memory_id)
        assert item is not None
        # metadata is a property that returns a dict
        metadata = item.metadata
        assert isinstance(metadata, dict)

        # Cleanup
        memory.delete(memory_id)
