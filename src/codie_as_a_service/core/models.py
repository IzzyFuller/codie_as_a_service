"""Core domain models for deep agent service."""

from typing import Any, Literal

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
