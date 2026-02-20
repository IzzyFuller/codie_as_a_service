"""Phase output models and orchestration config for the ReActOrchestrator."""

from __future__ import annotations

from pydantic import BaseModel

# =============================================================================
# Session Context (generic pipeline state)
# =============================================================================


class SessionContext(BaseModel):
    """Generic state that flows through the orchestration pipeline."""

    session_id: str
    agent_id: str
    instruction: str
    iteration: int = 0
    identity_summary: str = ""
    conversation_history: list[str] = []
    response: str = ""
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


class ExtendedInstruction(PhaseOutputModel):
    """Output of EXTEND phase: enriched instruction with tool selection."""

    instruction: str
    tool_manifest: list[str]
    rationale: str
    memory_references: list[str]

    def to_session_context(self, context: SessionContext) -> SessionContext:
        context.instruction = self.instruction
        return context


class ProcessResult(PhaseOutputModel):
    """Output of PROCESS phase: main reasoning result."""

    output: str
    tools_used: list[str]
    rationale: str

    def to_session_context(self, context: SessionContext) -> SessionContext:
        context.response = self.output
        return context


class ValidationResult(PhaseOutputModel):
    """Output of VALIDATE phase: completion check."""

    done: bool
    rationale: str
    feedback: str

    def to_session_context(self, context: SessionContext) -> SessionContext:
        context.done = self.done
        return context
