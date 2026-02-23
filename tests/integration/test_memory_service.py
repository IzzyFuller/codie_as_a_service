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

    def test_get_identity_context_includes_frame_all_required_parts(self, tmp_path):
        """
        Given: An agent with a frame file in memory
        When: get_identity_context is called
        Then: The result includes the frame content
        """
        adapter = LocalMemoryAdapter(base_dir=tmp_path)
        expected_frame = "# Tess Frame"
        expected_me = "# Tess"
        expected_anchors = "# Tess Anchors"
        expected_notes = "# Tess Session"
        adapter.write_file(agent_id="tess", key="frame", content=expected_frame)
        adapter.write_file(agent_id="tess", key="me", content=expected_me)
        adapter.write_file(
            agent_id="tess", key="context_anchors", content=expected_anchors
        )
        adapter.write_file(
            agent_id="tess", key="current_session", content=expected_notes
        )
        service = MemoryService(storage=adapter)

        result = service.get_identity_context(agent_id="tess")

        assert result.frame == expected_frame
        assert result.me == expected_me
        assert result.context_anchors == expected_anchors
        assert result.current_session == expected_notes


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
            frame="Be HAL 9000.",
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
            frame="Be HAL 9000.",
            response="Daisy, Daisy...",
            done=True,
        )

        phase = SynthesizePhaseDefinition(name="synthesize", memory=memory)
        phase.execute(context)

        persisted = memory.read_memory(agent_id="hal", key="current_session")
        assert persisted.startswith("# Session\nPrevious interaction data.")
        assert '"session_id": "s2"' in persisted
