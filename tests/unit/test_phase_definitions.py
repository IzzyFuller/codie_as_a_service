"""Unit tests for phase definitions."""

import pytest
from unittest.mock import MagicMock

from codie_as_a_service.core.phase_models import (
    SessionContext,
    ProcessResult,
    HydratedIdentity,
)
from codie_as_a_service.services.phases.definitions import (
    LLMPhaseDefinition,
    TextLLMPhaseDefinition,
)


class TestSessionContextFrame:
    """frame is a required field on SessionContext — no default."""

    def test_frame_required_raises_without_it(self):
        from pydantic import ValidationError

        with pytest.raises(ValidationError, match="frame"):
            SessionContext(
                session_id="s1",
                agent_id="a1",
                instruction="hello",
            )

    def test_frame_accepted_when_provided(self):
        ctx = SessionContext(
            session_id="s1",
            agent_id="a1",
            instruction="hello",
            frame="Be direct. No hedging.",
        )
        assert ctx.frame == "Be direct. No hedging."


class TestExecute:
    """execute() passes context fields to LLM, not raw model_dump_json."""

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

    def test_structured_phase_passes_context_to_llm(self):
        llm = MagicMock()
        llm.call.return_value = ProcessResult(output="hi", tools_used=[], rationale="")
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
        messages = call_kwargs.kwargs["messages"]
        system_prompt = call_kwargs.kwargs["system_prompt"]

        assert "I am a helpful agent." in system_prompt
        assert "Return JSON." in system_prompt
        assert len(messages) == 1
        assert messages[0].role == "user"
        assert "hello" in messages[0].content
        assert "session_id" not in messages[0].content

    def test_text_phase_passes_context_to_llm(self):
        llm = MagicMock()
        llm.call.return_value = "I am the restored identity."

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
        messages = call_kwargs.kwargs["messages"]
        system_prompt = call_kwargs.kwargs["system_prompt"]

        assert "I am a helpful agent." in system_prompt
        assert "Summarize identity." in system_prompt
        assert len(messages) == 1
        assert messages[0].role == "user"
        assert "hello" in messages[0].content
        assert "session_id" not in messages[0].content
