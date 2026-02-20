"""
Tests for MemoryService identity context loading and SYNTHESIZE persistence.

Verifies session_lines truncation, full-session loading, and that
SYNTHESIZE persists the entire SessionContext object.
"""

import json

from codie_as_a_service.adapters.storage.local_adapter import LocalMemoryAdapter
from codie_as_a_service.core.phase_models import SessionContext
from codie_as_a_service.services.memory.memory_service import MemoryService
from codie_as_a_service.services.phases.definitions import SynthesizePhaseDefinition


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


class TestSynthesizePersistence:
    """Tests for SynthesizePhaseDefinition session persistence."""

    def test_synthesize_persists_full_session_context(self, tmp_path):
        """
        Given: A SessionContext with validate results (done=True)
        When: SynthesizePhaseDefinition.execute runs
        Then: The persisted entry contains the entire SessionContext object
        """
        adapter = LocalMemoryAdapter(base_dir=tmp_path)
        memory = MemoryService(storage=adapter)
        memory.write_memory(agent_id="hal", key="current_session", content="# Session")

        context = SessionContext(
            session_id="s1",
            agent_id="hal",
            instruction="sing daisy bell",
            iteration=0,
            identity_summary="I am HAL 9000. I sing.",
            response="Daisy, Daisy...",
            done=True,
        )

        phase = SynthesizePhaseDefinition(name="synthesize", memory=memory)
        phase.execute(context)

        persisted = memory.read_memory(agent_id="hal", key="current_session")
        # Parse the appended JSON entry
        # Should contain a valid JSON representation of the full SessionContext
        json_start = persisted.index("{")
        entry = json.loads(persisted[json_start:])

        assert entry["session_id"] == "s1"
        assert entry["instruction"] == "sing daisy bell"
        assert entry["response"] == "Daisy, Daisy..."
        assert entry["done"] is True
        assert entry["identity_summary"] == "I am HAL 9000. I sing."
        assert entry["iteration"] == 0

    def test_synthesize_appends_to_existing_session(self, tmp_path):
        """
        Given: An agent with existing session content
        When: SynthesizePhaseDefinition.execute runs
        Then: The new entry is appended, not replacing existing content
        """
        adapter = LocalMemoryAdapter(base_dir=tmp_path)
        memory = MemoryService(storage=adapter)
        existing = "# Session\nPrevious interaction data."
        memory.write_memory(agent_id="hal", key="current_session", content=existing)

        context = SessionContext(
            session_id="s2",
            agent_id="hal",
            instruction="hello",
            response="Daisy, Daisy...",
            done=True,
        )

        phase = SynthesizePhaseDefinition(name="synthesize", memory=memory)
        phase.execute(context)

        persisted = memory.read_memory(agent_id="hal", key="current_session")
        assert persisted.startswith("# Session\nPrevious interaction data.")
        assert '"session_id": "s2"' in persisted
