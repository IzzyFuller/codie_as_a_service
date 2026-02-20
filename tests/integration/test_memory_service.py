"""
Tests for MemoryService identity context loading.

Verifies session_lines truncation and full-session loading behavior.
"""

from codie_as_a_service.adapters.storage.local_adapter import LocalMemoryAdapter
from codie_as_a_service.services.memory.memory_service import MemoryService


class TestGetIdentityContext:
    """Tests for MemoryService.get_identity_context."""

    def test_session_lines_truncates_current_session(self, tmp_path):
        """
        Given: An agent with a multi-line current_session
        When: get_identity_context is called with session_lines=2
        Then: Only the last 2 lines of current_session are included
        """
        adapter = LocalMemoryAdapter(base_dir=tmp_path)
        adapter.write_file(agent_id="tess", key="frame", content="# Frame")
        adapter.write_file(agent_id="tess", key="me", content="# Tess")
        adapter.write_file(agent_id="tess", key="context_anchors", content="# Anchors")
        adapter.write_file(
            agent_id="tess",
            key="current_session",
            content="line one\nline two\nline three\nline four",
        )
        service = MemoryService(storage=adapter)

        result = service.get_identity_context(agent_id="tess", session_lines=2)

        assert result.current_session == "line three\nline four"

    def test_no_session_lines_returns_full_session(self, tmp_path):
        """
        Given: An agent with a multi-line current_session
        When: get_identity_context is called without session_lines
        Then: The full current_session content is returned
        """
        adapter = LocalMemoryAdapter(base_dir=tmp_path)
        adapter.write_file(agent_id="tess", key="frame", content="# Frame")
        adapter.write_file(agent_id="tess", key="me", content="# Tess")
        adapter.write_file(agent_id="tess", key="context_anchors", content="# Anchors")
        full_content = "line one\nline two\nline three\nline four"
        adapter.write_file(agent_id="tess", key="current_session", content=full_content)
        service = MemoryService(storage=adapter)

        result = service.get_identity_context(agent_id="tess")

        assert result.current_session == full_content

    def test_get_identity_context_includes_frame(self, tmp_path):
        """
        Given: An agent with a frame file in memory
        When: get_identity_context is called
        Then: The result includes the frame content
        """
        adapter = LocalMemoryAdapter(base_dir=tmp_path)
        adapter.write_file(agent_id="tess", key="frame", content="# Tess Frame")
        adapter.write_file(agent_id="tess", key="me", content="# Tess")
        adapter.write_file(agent_id="tess", key="context_anchors", content="# Anchors")
        adapter.write_file(agent_id="tess", key="current_session", content="# Session")
        service = MemoryService(storage=adapter)

        result = service.get_identity_context(agent_id="tess")

        assert result.frame == "# Tess Frame"
