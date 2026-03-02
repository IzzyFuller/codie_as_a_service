"""Phase output models and orchestration config for the ReActOrchestrator."""

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
    output_schema: type[BaseModel] = Field(exclude=True)


# =============================================================================
# Default Output
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


class ProcessedText(PhaseOutputModel):
    """Output of PROCESS phase: plain-text reasoning result."""

    text_output: str

    def to_session_context(self, context: SessionContext) -> SessionContext:
        context.response = self.text_output
        context.conversation_history.append(f"PROCESS: {self.text_output}")
        return context
