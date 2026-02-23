"""
Tests for ReActOrchestrator identity guard clauses.

Verifies that the orchestrator raises clear errors when required
identity files (me, frame) are missing for an agent.
"""

import pytest

from codie_as_a_service.adapters.storage.local_adapter import LocalMemoryAdapter
from codie_as_a_service.services.agent.react_orchestrator import ReActOrchestrator
from codie_as_a_service.services.memory.memory_service import MemoryService


class TestOrchestratorIdentityGuards:
    """Tests for identity validation before phase execution."""

    def test_raises_when_agent_has_no_me_file(self, tmp_path):
        """
        Given: An agent with a frame but no me file
        When: orchestrator.run is called
        Then: It raises ValueError about missing identity
        """
        adapter = LocalMemoryAdapter(base_dir=tmp_path)
        adapter.write_file(agent_id="ghost", key="frame", content="# Ghost Frame")
        memory = MemoryService(storage=adapter)
        orchestrator = ReActOrchestrator(memory=memory, phases=[])

        with pytest.raises(
            ValueError, match="No assistant identity configured for agent 'ghost'"
        ):
            orchestrator.run(session_id="s1", agent_id="ghost", instruction="hello")

    def test_raises_when_agent_has_no_frame_file(self, tmp_path):
        """
        Given: An agent with a me file but no frame
        When: orchestrator.run is called
        Then: It raises ValueError about missing frame
        """
        adapter = LocalMemoryAdapter(base_dir=tmp_path)
        adapter.write_file(agent_id="ghost", key="me", content="# Ghost Identity")
        memory = MemoryService(storage=adapter)
        orchestrator = ReActOrchestrator(memory=memory, phases=[])

        with pytest.raises(ValueError, match="No frame configured for agent 'ghost'"):
            orchestrator.run(session_id="s1", agent_id="ghost", instruction="hello")
