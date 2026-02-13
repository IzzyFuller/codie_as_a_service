"""MCP tool executor implementing ToolExecutor protocol."""

from typing import Any

from codie_as_a_service.adapters.mcp.mcp_client import MCPStdioClient


class MCPToolExecutor:
    """
    Executes MCP-based tools via stdio client.

    Routes tool calls to an MCP server. agent_id is not forwarded
    to the MCP server — the server handles its own scoping via
    its COGNITIVE_MEMORY_PATH environment variable.
    """

    def __init__(self, mcp_client: MCPStdioClient) -> None:
        self._mcp = mcp_client

    def execute(self, agent_id: str, tool_name: str, tool_input: dict[str, Any]) -> str:
        """Execute a tool via MCP server. agent_id is ignored (MCP scopes internally)."""
        return self._mcp.call_tool(tool_name, tool_input)
