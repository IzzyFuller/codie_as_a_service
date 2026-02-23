"""Core domain models for deep agent service."""

from typing import Any, Literal, Union

from pydantic import BaseModel, Field


class IdentityContext(BaseModel):
    """
    Core identity files loaded at agent session start.

    These files establish who the agent is and their current working context.
    """

    frame: str | None = Field(
        default=None, description="Essential frame — existential grounding and voice"
    )
    current_session: str | None = Field(
        default=None, description="Session notes and recent work"
    )
    context_anchors: str | None = Field(
        default=None, description="Working memory pointers to entities"
    )
    me: str | None = Field(
        default=None, description="Core identity and behavioral patterns"
    )


# =============================================================================
# Message Models
# =============================================================================


class Message(BaseModel):
    """A message in the conversation history."""

    role: Literal["user", "assistant", "system"] = Field(
        ..., description="Role of the message sender"
    )
    content: str = Field(..., description="Message content")


class ToolResult(BaseModel):
    """Result from a tool execution, sent back to LLM."""

    tool_use_id: str = Field(..., description="ID of the tool call this responds to")
    content: str = Field(..., description="Result content from tool execution")


# =============================================================================
# Tool Models
# =============================================================================


class ToolDefinition(BaseModel):
    """Definition of a tool available to the agent."""

    name: str = Field(..., description="Tool name")
    description: str = Field(..., description="What the tool does")
    input_schema: dict[str, Any] = Field(
        ..., description="JSON schema for tool input parameters"
    )


# =============================================================================
# LLM Response Models
# =============================================================================


class ContentBlock(BaseModel):
    """Text content block in LLM response."""

    type: Literal["text"] = "text"
    text: str = Field(..., description="Text content")


class ToolUseBlock(BaseModel):
    """Tool use request block in LLM response."""

    type: Literal["tool_use"] = "tool_use"
    id: str = Field(..., description="Unique ID for this tool call")
    name: str = Field(..., description="Name of tool to call")
    input: dict[str, Any] = Field(..., description="Input arguments for the tool")


class LLMResponse(BaseModel):
    """Structured response from LLM."""

    stop_reason: Literal["end_turn", "tool_use", "max_tokens"] = Field(
        ..., description="Why the LLM stopped generating"
    )
    content: list[Union[ContentBlock, ToolUseBlock]] = Field(
        default_factory=list, description="Content blocks in the response"
    )
    data: dict[str, Any] | None = Field(
        default=None, description="Parsed structured output when output_format was used"
    )


# =============================================================================
# Pub/Sub Message Models (E2E Communication)
# =============================================================================


class RunAgentRequest(BaseModel):
    """Request to run the agent, delivered via Pub/Sub."""

    agent_id: str = Field(
        ..., description="Agent identifier for agent memory isolation"
    )
    session_id: str = Field(
        ..., description="Session identifier for conversation continuity"
    )
    message: str = Field(..., description="User's message to the agent")
    output_format: dict[str, Any] | None = Field(
        default=None, description="Optional JSON Schema for structured output"
    )


class AgentResponse(BaseModel):
    """Response from agent processing, published back to Pub/Sub."""

    agent_id: str = Field(..., description="Agent identifier for this response")
    session_id: str = Field(
        ..., description="Session identifier for conversation continuity"
    )
    response_data: dict[str, Any] = Field(
        ..., description="Structured response data (always populated)"
    )
    status: Literal["success", "error", "max_iterations"] = Field(
        ..., description="Processing status"
    )
