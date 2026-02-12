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
    """Output of SYNTHESIZE phase: memory writes."""

    writes: list[str]
    summary: str


# =============================================================================
# Orchestration Config
# =============================================================================


class PhaseDefinition(BaseModel):
    """Configuration for a single orchestration phase."""

    name: str
    system_prompt: str
    tools: list[ToolDefinition] = []
    output_schema: type[BaseModel]
    max_iterations: int = 1
    completes_request: bool = False
    max_new_tokens: int | None = None


# =============================================================================
# Orchestration State
# =============================================================================


class OrchestrationContext(BaseModel):
    """Mutable state passed between phases within an iteration."""

    agent_id: str
    instruction: str
    iteration: int = 0
    hydrate: HydratedIdentity | None = None
    extend: ExtendedInstruction | None = None
    process: ProcessResult | None = None
    validate: ValidationResult | None = None
    synthesize: SynthesisResult | None = None
