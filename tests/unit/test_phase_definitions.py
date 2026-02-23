"""
Unit tests for phase definition message building.

Tests the shift to: identity as system prompt, instruction as user message.
Written BEFORE implementation (TDD).
"""

import pytest
from unittest.mock import MagicMock, call

from codie_as_a_service.core.models import Message
from codie_as_a_service.core.phase_models import (
    SessionContext,
    ExtendedInstruction,
    ProcessResult,
    ValidationResult,
    HydratedIdentity,
)
from codie_as_a_service.services.phases.definitions import (
    LLMPhaseDefinition,
    TextLLMPhaseDefinition,
)


# ============================================================================
# SessionContext: frame field is required
# ============================================================================


class TestSessionContextFrame:
    """frame is a required field on SessionContext — no default."""

    def test_frame_required_raises_without_it(self):
        """
        Given: SessionContext constructed without frame
        When: Pydantic validates
        Then: ValidationError — frame has no default
        """
        from pydantic import ValidationError

        with pytest.raises(ValidationError, match="frame"):
            SessionContext(
                session_id="s1",
                agent_id="a1",
                instruction="hello",
            )

    def test_frame_accepted_when_provided(self):
        """
        Given: SessionContext constructed with frame
        When: Pydantic validates
        Then: Object created with frame value
        """
        ctx = SessionContext(
            session_id="s1",
            agent_id="a1",
            instruction="hello",
            frame="Be direct. No hedging.",
        )
        assert ctx.frame == "Be direct. No hedging."


# ============================================================================
# _build_system_prompt: identity_summary + phase instructions
# ============================================================================


class TestBuildSystemPrompt:
    """system_prompt = identity_summary + \\n\\n + phase instructions."""

    def _make_context(self, **overrides) -> SessionContext:
        defaults = dict(
            session_id="s1",
            agent_id="a1",
            instruction="hello",
            frame="Be direct.",
            identity_summary="I am a helpful agent.",
        )
        defaults.update(overrides)
        return SessionContext(**defaults)

    def test_structured_phase_prepends_identity_to_prompt(self):
        """
        Given: LLMPhaseDefinition with system_prompt "Return JSON."
        When: _build_system_prompt called with context containing identity_summary
        Then: Returns "I am a helpful agent.\n\nReturn JSON."
        """
        llm = MagicMock()
        phase = LLMPhaseDefinition(
            name="test",
            llm=llm,
            system_prompt="Return JSON.",
            output_schema=ProcessResult,
        )
        ctx = self._make_context()
        result = phase._build_system_prompt(ctx)
        assert result == "I am a helpful agent.\n\nReturn JSON."

    def test_text_phase_prepends_identity_to_prompt(self):
        """
        Given: TextLLMPhaseDefinition with system_prompt "Summarize identity."
        When: _build_system_prompt called
        Then: Returns identity_summary + \\n\\n + phase prompt
        """
        llm = MagicMock()
        phase = TextLLMPhaseDefinition(
            name="test",
            llm=llm,
            system_prompt="Summarize identity.",
            output_schema=HydratedIdentity,
        )
        ctx = self._make_context()
        result = phase._build_system_prompt(ctx)
        assert result == "I am a helpful agent.\n\nSummarize identity."

    def test_empty_identity_returns_phase_prompt_only(self):
        """
        Given: Context with empty identity_summary
        When: _build_system_prompt called
        Then: Returns phase prompt without leading separator
        """
        llm = MagicMock()
        phase = LLMPhaseDefinition(
            name="test",
            llm=llm,
            system_prompt="Return JSON.",
            output_schema=ProcessResult,
        )
        ctx = self._make_context(identity_summary="")
        result = phase._build_system_prompt(ctx)
        assert result == "Return JSON."


# ============================================================================
# _build_user_message: instruction + non-empty context
# ============================================================================


class TestBuildUserMessage:
    """user message = instruction, plus response and history when non-empty."""

    def _make_context(self, **overrides) -> SessionContext:
        defaults = dict(
            session_id="s1",
            agent_id="a1",
            instruction="What is 2+2?",
            frame="Be direct.",
            identity_summary="I am a helpful agent.",
        )
        defaults.update(overrides)
        return SessionContext(**defaults)

    def test_instruction_only_when_response_and_history_empty(self):
        """
        Given: Context with empty response and empty conversation_history
        When: _build_user_message called
        Then: Returns instruction only
        """
        llm = MagicMock()
        phase = LLMPhaseDefinition(
            name="test",
            llm=llm,
            system_prompt="prompt",
            output_schema=ProcessResult,
        )
        ctx = self._make_context(response="", conversation_history=[])
        result = phase._build_user_message(ctx)
        assert result == "What is 2+2?"

    def test_includes_response_when_populated(self):
        """
        Given: Context with non-empty response
        When: _build_user_message called
        Then: Returns instruction + response section
        """
        llm = MagicMock()
        phase = LLMPhaseDefinition(
            name="test",
            llm=llm,
            system_prompt="prompt",
            output_schema=ProcessResult,
        )
        ctx = self._make_context(response="The answer is 4.")
        result = phase._build_user_message(ctx)
        assert "What is 2+2?" in result
        assert "The answer is 4." in result

    def test_includes_history_when_populated(self):
        """
        Given: Context with non-empty conversation_history
        When: _build_user_message called
        Then: Returns instruction + history section
        """
        llm = MagicMock()
        phase = LLMPhaseDefinition(
            name="test",
            llm=llm,
            system_prompt="prompt",
            output_schema=ProcessResult,
        )
        ctx = self._make_context(
            conversation_history=["HYDRATE: {}", "EXTEND: {}"],
        )
        result = phase._build_user_message(ctx)
        assert "What is 2+2?" in result
        assert "HYDRATE: {}" in result
        assert "EXTEND: {}" in result

    def test_includes_both_response_and_history_when_populated(self):
        """
        Given: Context with both response and history populated
        When: _build_user_message called
        Then: Returns all three sections
        """
        llm = MagicMock()
        phase = LLMPhaseDefinition(
            name="test",
            llm=llm,
            system_prompt="prompt",
            output_schema=ProcessResult,
        )
        ctx = self._make_context(
            response="The answer is 4.",
            conversation_history=["HYDRATE: prior output"],
        )
        result = phase._build_user_message(ctx)
        assert "What is 2+2?" in result
        assert "The answer is 4." in result
        assert "HYDRATE: prior output" in result

    def test_text_phase_builds_same_user_message(self):
        """
        Given: TextLLMPhaseDefinition
        When: _build_user_message called
        Then: Same behavior as structured phase
        """
        llm = MagicMock()
        phase = TextLLMPhaseDefinition(
            name="test",
            llm=llm,
            system_prompt="prompt",
            output_schema=HydratedIdentity,
        )
        ctx = self._make_context(response="", conversation_history=[])
        result = phase._build_user_message(ctx)
        assert result == "What is 2+2?"

    def test_text_phase_includes_response_and_history(self):
        """
        Given: TextLLMPhaseDefinition with response and history populated
        When: _build_user_message called
        Then: Includes all sections
        """
        llm = MagicMock()
        phase = TextLLMPhaseDefinition(
            name="test",
            llm=llm,
            system_prompt="prompt",
            output_schema=HydratedIdentity,
        )
        ctx = self._make_context(
            response="Prior output.",
            conversation_history=["HYDRATE: done"],
        )
        result = phase._build_user_message(ctx)
        assert "What is 2+2?" in result
        assert "Prior output." in result
        assert "HYDRATE: done" in result

    def test_text_phase_empty_identity_returns_phase_prompt_only(self):
        """
        Given: TextLLMPhaseDefinition with empty identity_summary
        When: _build_system_prompt called
        Then: Returns phase prompt without identity prefix
        """
        llm = MagicMock()
        phase = TextLLMPhaseDefinition(
            name="test",
            llm=llm,
            system_prompt="Summarize.",
            output_schema=HydratedIdentity,
        )
        ctx = self._make_context(identity_summary="")
        result = phase._build_system_prompt(ctx)
        assert result == "Summarize."


# ============================================================================
# execute() uses _build_system_prompt and _build_user_message
# ============================================================================


class TestExecuteUsesMessageBuilders:
    """execute() passes built messages to LLM, not raw model_dump_json."""

    def _make_context(self, **overrides) -> SessionContext:
        defaults = dict(
            session_id="s1",
            agent_id="a1",
            instruction="hello",
            frame="Be direct.",
            identity_summary="I am a helpful agent.",
        )
        defaults.update(overrides)
        return SessionContext(**defaults)

    def test_structured_phase_passes_built_messages_to_llm(self):
        """
        Given: LLMPhaseDefinition with identity in context
        When: execute() called
        Then: LLM receives system_prompt=identity+phase_prompt, user message=instruction
        """
        llm = MagicMock()
        llm.call.return_value = MagicMock(
            data={"output": "hi", "tools_used": [], "rationale": ""}
        )
        phase = LLMPhaseDefinition(
            name="process",
            llm=llm,
            system_prompt="Return JSON.",
            output_schema=ProcessResult,
        )
        ctx = self._make_context()
        phase.execute(ctx)

        llm.call.assert_called_once()
        call_kwargs = llm.call.call_args
        messages = call_kwargs.kwargs.get("messages") or call_kwargs[1].get("messages") if len(call_kwargs) > 1 else call_kwargs[0][0] if call_kwargs[0] else call_kwargs.kwargs["messages"]

        # System prompt should be identity + phase prompt
        system_prompt = call_kwargs.kwargs.get("system_prompt") or (call_kwargs[0][1] if len(call_kwargs[0]) > 1 else call_kwargs.kwargs["system_prompt"])
        assert system_prompt == "I am a helpful agent.\n\nReturn JSON."

        # User message should be instruction, NOT model_dump_json
        assert len(messages) == 1
        assert messages[0].role == "user"
        assert messages[0].content == "hello"
        # Must NOT contain full SessionContext JSON dump
        assert "session_id" not in messages[0].content
        assert "model_dump" not in messages[0].content

    def test_text_phase_passes_built_messages_to_llm(self):
        """
        Given: TextLLMPhaseDefinition with identity in context
        When: execute() called
        Then: LLM receives system_prompt=identity+phase_prompt, user message=instruction
        """
        llm = MagicMock()
        mock_response = MagicMock()
        mock_response.content = [MagicMock(text="I am the restored identity.")]
        llm.call.return_value = mock_response

        phase = TextLLMPhaseDefinition(
            name="hydrate",
            llm=llm,
            system_prompt="Summarize identity.",
            output_schema=HydratedIdentity,
        )
        ctx = self._make_context()
        phase.execute(ctx)

        llm.call.assert_called_once()
        call_kwargs = llm.call.call_args

        messages = call_kwargs.kwargs.get("messages") or call_kwargs[0][0]
        system_prompt = call_kwargs.kwargs.get("system_prompt") or call_kwargs[0][1]

        assert system_prompt == "I am a helpful agent.\n\nSummarize identity."
        assert len(messages) == 1
        assert messages[0].role == "user"
        assert "hello" in messages[0].content
        assert "session_id" not in messages[0].content
