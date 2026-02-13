"""Composite tool executor that routes tool calls to the appropriate executor."""

from typing import Any

from codie_as_a_service.core.protocols import ToolExecutor


class CompositeToolExecutor:
    """
    Routes tool calls to the appropriate executor by tool name.

    Each tool name maps to exactly one executor. Unknown tools raise ValueError.
    """

    def __init__(self, executors: dict[str, ToolExecutor]) -> None:
        """
        Initialize composite executor.

        Args:
            executors: Mapping of tool_name -> executor instance
        """
        self._executors = executors

    def execute(self, agent_id: str, tool_name: str, tool_input: dict[str, Any]) -> str:
        """Route tool call to the registered executor. Raises ValueError for unknown tools."""
        executor = self._executors.get(tool_name)
        if not executor:
            raise ValueError(f"No executor for tool: {tool_name}")
        return executor.execute(agent_id, tool_name, tool_input)
