"""Pub/Sub message models for agent communication."""

from typing import Any, Literal

from pydantic import BaseModel, Field


class RunAgentRequest(BaseModel):
    """Request to run the agent, delivered via Pub/Sub."""

    agent_id: str = Field(
        ..., description="Agent identifier for agent memory isolation"
    )
    session_id: str | None = Field(
        default=None, description="Session identifier for conversation continuity"
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
