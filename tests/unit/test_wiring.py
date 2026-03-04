"""Unit tests for wiring module."""

from unittest.mock import Mock

import pytest

from codie_as_a_service.services.wiring import (
    build_orchestrator_phases,
    get_mcp_tool_definitions,
)


class TestBuildOrchestratorPhases:
    """Tests for build_orchestrator_phases function."""

    @pytest.fixture
    def mock_prompt_adapter(self):
        """Mock prompt adapter."""
        adapter = Mock()
        adapter.get_prompt.return_value = "test prompt"
        return adapter

    @pytest.fixture
    def mock_llm(self):
        """Mock LLM adapter."""
        return Mock()

    @pytest.fixture
    def mock_memory(self):
        """Mock memory service."""
        return Mock()

    def test_build_all_phases(self, mock_prompt_adapter, mock_llm, mock_memory):
        """
        Given: phase_names includes all available phases
        When: build_orchestrator_phases is called
        Then: All phases are built and returned in correct order
        """
        phases, post_phases = build_orchestrator_phases(
            phase_names=["hydrate", "process"],
            prompt_adapter=mock_prompt_adapter,
            tools=[],
            llm=mock_llm,
            memory=mock_memory,
        )

        # Should have 2 main phases
        assert len(phases) == 2
        assert phases[0].name == "hydrate"
        assert phases[1].name == "process"

        # Should have 1 post phase (synthesize always included)
        assert len(post_phases) == 1
        assert post_phases[0].name == "synthesize"

    def test_build_subset_of_phases(self, mock_prompt_adapter, mock_llm, mock_memory):
        """
        Given: phase_names includes only a subset of phases
        When: build_orchestrator_phases is called
        Then: Only requested phases are built
        """
        phases, post_phases = build_orchestrator_phases(
            phase_names=["process"],
            prompt_adapter=mock_prompt_adapter,
            tools=[],
            llm=mock_llm,
            memory=mock_memory,
        )

        # Should have only 1 main phase
        assert len(phases) == 1
        assert phases[0].name == "process"

        # Post phases still included
        assert len(post_phases) == 1
        assert post_phases[0].name == "synthesize"

    def test_build_no_phases(self, mock_prompt_adapter, mock_llm, mock_memory):
        """
        Given: phase_names is empty
        When: build_orchestrator_phases is called
        Then: No main phases are built, but post_phases still included
        """
        phases, post_phases = build_orchestrator_phases(
            phase_names=[],
            prompt_adapter=mock_prompt_adapter,
            tools=[],
            llm=mock_llm,
            memory=mock_memory,
        )

        # Should have no main phases
        assert len(phases) == 0

        # Post phases still included
        assert len(post_phases) == 1
        assert post_phases[0].name == "synthesize"

    def test_build_phases_ignores_order(
        self, mock_prompt_adapter, mock_llm, mock_memory
    ):
        """
        Given: phase_names in non-standard order
        When: build_orchestrator_phases is called
        Then: Phases are built in hardcoded order (hydrate, process)
        """
        phases, _ = build_orchestrator_phases(
            phase_names=["process", "hydrate"],
            prompt_adapter=mock_prompt_adapter,
            tools=[],
            llm=mock_llm,
            memory=mock_memory,
        )

        # Implementation uses hardcoded order, not phase_names order
        assert len(phases) == 2
        assert phases[0].name == "hydrate"
        assert phases[1].name == "process"


class TestToolDefinitions:
    """Tests for tool definition factory functions."""

    def test_get_mcp_tool_definitions(self):
        """
        Given: No parameters needed
        When: get_mcp_tool_definitions is called
        Then: Returns cognitive-memory MCP tool definitions
        """
        tools = get_mcp_tool_definitions()

        assert len(tools) == 4
        tool_names = [tool.name for tool in tools]
        assert "list_entities" in tool_names
        assert "read_entity" in tool_names
        assert "write_entity" in tool_names
        assert "add_session_note" in tool_names
