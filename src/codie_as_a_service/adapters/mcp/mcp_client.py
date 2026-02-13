"""MCP stdio client wrapper for communicating with MCP servers."""

import asyncio

from mcp import ClientSession, StdioServerParameters, types
from mcp.client.stdio import stdio_client


class MCPStdioClient:
    """
    Synchronous wrapper around the MCP Python SDK's async stdio client.

    Spawns an MCP server process via stdio and calls tools on it.
    Each call_tool invocation creates a fresh connection. This is simple
    and correct; optimize to persistent connection if latency matters.
    """

    def __init__(
        self,
        command: str,
        args: list[str],
        env: dict[str, str] | None = None,
    ):
        """
        Initialize MCP stdio client.

        Args:
            command: Command to spawn the MCP server (e.g., "node")
            args: Arguments for the command (e.g., ["path/to/server.js"])
            env: Environment variables for the server process
        """
        self._server_params = StdioServerParameters(
            command=command,
            args=args,
            env=env,
        )

    async def _call_tool_async(self, tool_name: str, arguments: dict) -> str:
        """
        Call a tool on the MCP server asynchronously.

        Spawns the server, initializes session, calls tool, extracts text.
        """
        async with stdio_client(self._server_params) as (read, write):
            async with ClientSession(read, write) as session:
                await session.initialize()
                result = await session.call_tool(tool_name, arguments=arguments)

                # Extract text content from result
                texts = []
                for content in result.content:
                    if isinstance(content, types.TextContent):
                        texts.append(content.text)
                return "\n".join(texts)

    def call_tool(self, tool_name: str, arguments: dict) -> str:
        """
        Call a tool on the MCP server synchronously.

        Bridges sync→async using asyncio.run().

        Args:
            tool_name: Name of the MCP tool to call
            arguments: Arguments for the tool

        Returns:
            Tool result as string
        """
        return asyncio.run(self._call_tool_async(tool_name, arguments))
