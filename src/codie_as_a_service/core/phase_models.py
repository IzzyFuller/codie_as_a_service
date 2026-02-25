"""Phase output models and orchestration config for the ReActOrchestrator."""

from __future__ import annotations

from typing import Any

from pydantic import BaseModel, Field

# =============================================================================
# Session Context (generic pipeline state)
# =============================================================================


class SessionContext(BaseModel):
    """Generic state that flows through the orchestration pipeline."""

    session_id: str
    agent_id: str
    instruction: str
    frame: str
    iteration: int = 0
    identity_summary: str = ""
    conversation_history: list[str] = []
    response: str = ""
    done: bool = False
    output_format_override: Any = Field(default=None, exclude=True)


# =============================================================================
# Phase Output Base
# =============================================================================


class DefaultOutput(BaseModel):
    """Default output shape returned to clients when no output_format is specified."""

    response: str = ""
    session_id: str = ""
    done: bool = False


# =============================================================================
# Phase Output Base
# =============================================================================


class PhaseOutputModel(BaseModel):
    """Base for all phase output models.

    Each phase output knows how to apply itself to the SessionContext.
    The orchestrator calls to_session_context() — it never needs to know
    which fields a phase updates.
    """

    def to_session_context(self, context: SessionContext) -> SessionContext:
        """Apply this phase's output to the session context.

        Subclasses must override to update the fields they own.
        """
        raise NotImplementedError


# =============================================================================
# Phase Output Models
# =============================================================================


class HydratedIdentity(PhaseOutputModel):
    """Output of HYDRATE phase: plain-text identity summary."""

    text_output: str

    def to_session_context(self, context: SessionContext) -> SessionContext:
        context.identity_summary = self.text_output
        context.conversation_history.append(f"HYDRATE: {self.model_dump_json()}")
        return context


class ExtendedContext(PhaseOutputModel):
    """Output of EXTEND phase: plain-text enriched context."""

    text_output: str

    def to_session_context(self, context: SessionContext) -> SessionContext:
        context.conversation_history.append(f"EXTEND: {self.model_dump_json()}")
        return context


class ProcessResult(PhaseOutputModel):
    """Output of PROCESS phase: main reasoning result."""

    output: str
    tools_used: list[str]
    rationale: str

    def to_session_context(self, context: SessionContext) -> SessionContext:
        context.response = self.output
        context.conversation_history.append(f"PROCESS: {self.model_dump_json()}")
        return context


class ValidationResult(PhaseOutputModel):
    """Output of VALIDATE phase: completion check."""

    done: bool
    rationale: str
    feedback: str

    def to_session_context(self, context: SessionContext) -> SessionContext:
        context.done = self.done
        context.conversation_history.append(f"VALIDATE: {self.model_dump_json()}")
        return context
