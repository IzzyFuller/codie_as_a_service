"""Abstract protocols (interfaces) for dependency injection."""

from __future__ import annotations

from typing import TYPE_CHECKING, Any, Protocol, runtime_checkable

from codie_as_a_service.core.models import (
    LLMResponse,
    Message,
    ToolDefinition,
)

if TYPE_CHECKING:
    from codie_as_a_service.core.phase_models import SessionContext


@runtime_checkable
class MemoryProtocol(Protocol):
    """Abstract interface for memory storage operations."""

    def read_file(self, agent_id: str, key: str) -> str | None:
        """
        Read a memory file for an agent.

        Args:
            agent_id: Agent identifier
            key: Memory key (e.g., 'current_session', 'context_anchors')

        Returns:
            File content as string, or None if file doesn't exist
        """
        ...

    def write_file(self, agent_id: str, key: str, content: str) -> None:
        """
        Write a memory file for an agent.

        Args:
            agent_id: Agent identifier
            key: Memory key
            content: Content to write
        """
        ...

    def list_files(self, agent_id: str) -> list[str]:
        """
        List all memory files for an agent.

        Args:
            agent_id: Agent identifier

        Returns:
            List of memory keys
        """
        ...


@runtime_checkable
class LLMProtocol(Protocol):
    """Abstract interface for LLM operations."""

    def call(
        self,
        messages: list["Message"],
        system_prompt: str,
        tools: list["ToolDefinition"] | None = None,
        output_format: dict[str, Any] | None = None,
        max_new_tokens: int | None = None,
    ) -> "LLMResponse":
        """
        Call the LLM with messages and optional tools.

        Args:
            messages: Conversation history
            system_prompt: System prompt for the agent
            tools: Optional list of tool definitions
            output_format: Optional JSON schema for structured output
            max_new_tokens: Optional max tokens to generate (adapter-specific)

        Returns:
            Structured LLM response
        """
        ...


@runtime_checkable
class PromptProtocol(Protocol):
    """Abstract interface for prompt management."""

    def get_prompt(self, name: str, **variables: str) -> str:
        """
        Get a prompt by name with variable substitution.

        Args:
            name: Prompt identifier
            **variables: Template variables to inject

        Returns:
            Compiled prompt string
        """
        ...


@runtime_checkable
class AuthProtocol(Protocol):
    """Abstract interface for authentication."""

    def verify(self, credentials: str) -> bool:
        """
        Verify credentials.

        Args:
            credentials: The credentials to verify (e.g., API key)

        Returns:
            True if valid, False otherwise
        """
        ...


@runtime_checkable
class Phase(Protocol):
    """Abstract interface for an orchestration phase.

    Any class with an execute(context) method satisfies this protocol.
    LLM phases call an adapter; deterministic phases (e.g. SYNTHESIZE)
    do their work directly.
    """

    name: str

    def execute(self, context: SessionContext) -> None:
        """Execute this phase, mutating the session context in place."""
        ...
