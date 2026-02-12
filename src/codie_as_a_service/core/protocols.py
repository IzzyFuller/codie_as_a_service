"""Abstract protocols (interfaces) for dependency injection."""

from typing import Any, Protocol, runtime_checkable

from codie_as_a_service.core.models import (
    LLMResponse,
    Message,
    ToolDefinition,
)


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
class ToolExecutor(Protocol):
    """Abstract interface for tool execution."""

    def execute(self, agent_id: str, tool_name: str, tool_input: dict[str, Any]) -> str:
        """
        Execute a tool and return the result as a string.

        Args:
            agent_id: Agent identifier for scoped operations
            tool_name: Name of the tool to execute
            tool_input: Input arguments for the tool

        Returns:
            Tool result as string
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
