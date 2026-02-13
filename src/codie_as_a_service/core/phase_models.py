"""Phase output models and orchestration config for the ReActOrchestrator."""

from pydantic import BaseModel

from codie_as_a_service.core.models import ToolDefinition


# =============================================================================
# Phase Output Models
# =============================================================================


class HydratedIdentity(BaseModel):
    """Output of HYDRATE phase: condensed identity context."""

    summary: str
    key_patterns: list[str]
    session_state: str


class ExtendedInstruction(BaseModel):
    """Output of EXTEND phase: enriched instruction with tool selection."""

    instruction: str
    tool_manifest: list[str]
    rationale: str
    memory_references: list[str]


class ProcessResult(BaseModel):
    """Output of PROCESS phase: main reasoning result."""

    output: str
    tools_used: list[str]
    trace: str


class ValidationResult(BaseModel):
    """Output of VALIDATE phase: completion check."""

    done: bool
    justification: str
    feedback: str


class SynthesisResult(BaseModel):
    """Output of SYNTHESIZE phase: memory writes + user-facing response."""

    response: str = ""
    writes: list[str]
    summary: str


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
# Orchestration Config
# =============================================================================


class PhaseDefinition(BaseModel):
    """Configuration for a single orchestration phase."""

    name: str
    system_prompt: str
    tools: list[ToolDefinition] = []
    output_schema: type[BaseModel]
    max_new_tokens: int | None = None
    sets_response_from: str | None = None
    sets_done_from: str | None = None
    sets_identity_from: str | None = None
