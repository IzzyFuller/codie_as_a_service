"""Unit tests for phase definitions."""

from unittest.mock import Mock

from codie_as_a_service.core.phase_models import (
    DefaultOutput,
    HydratedIdentity,
    SessionContext,
)
from codie_as_a_service.services.phases import TextLLMPhaseDefinition


class TestTextLLMPhaseDefinition:
    """Tests for TextLLMPhaseDefinition behavior."""

    def _make_context(self, iteration: int = 0) -> SessionContext:
        """Build a minimal SessionContext for phase testing."""
        return SessionContext(
            session_id="test-session",
            agent_id="test-agent",
            instruction="test instruction",
            frame="test frame",
            iteration=iteration,
            output_schema=DefaultOutput,
        )

    def test_skip_on_retry_skips_llm_call_on_subsequent_iterations(self):
        """
        Given: A phase with skip_on_retry=True
        When: execute() is called with iteration > 0
        Then: LLM is NOT called and context is unchanged
        """
        mock_llm = Mock()
        phase = TextLLMPhaseDefinition(
            name="hydrate",
            llm=mock_llm,
            system_prompt="test prompt",
            output_schema=HydratedIdentity,
            skip_on_retry=True,
        )

        context = self._make_context(iteration=1)
        original_summary = context.identity_summary

        phase.execute(context)

        mock_llm.call.assert_not_called()
        assert context.identity_summary == original_summary

    def test_skip_on_retry_runs_normally_on_first_iteration(self):
        """
        Given: A phase with skip_on_retry=True
        When: execute() is called with iteration=0
        Then: LLM IS called normally
        """
        mock_llm = Mock()
        mock_llm.call.return_value = "Identity summary from LLM."
        phase = TextLLMPhaseDefinition(
            name="hydrate",
            llm=mock_llm,
            system_prompt="test prompt",
            output_schema=HydratedIdentity,
            skip_on_retry=True,
        )

        context = self._make_context(iteration=0)

        phase.execute(context)

        mock_llm.call.assert_called_once()
        assert context.identity_summary == "Identity summary from LLM."

    def test_phase_without_skip_on_retry_runs_on_all_iterations(self):
        """
        Given: A phase with skip_on_retry=False (default)
        When: execute() is called with iteration > 0
        Then: LLM IS called
        """
        mock_llm = Mock()
        mock_llm.call.return_value = "Processed text."
        phase = TextLLMPhaseDefinition(
            name="process",
            llm=mock_llm,
            system_prompt="test prompt",
            output_schema=HydratedIdentity,
        )

        context = self._make_context(iteration=2)

        phase.execute(context)

        mock_llm.call.assert_called_once()
