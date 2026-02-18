"""
Tests for MCP client components.

Tests MCPStdioClient sync wrapper around MCP SDK.
MCPStdioClient is mocked at the boundary - we don't spawn real MCP servers in tests.
"""

from contextlib import asynccontextmanager
from unittest.mock import AsyncMock, MagicMock, patch

from mcp import types

from codie_as_a_service.adapters.mcp.mcp_client import MCPStdioClient


class TestMCPStdioClient:
    """Tests for MCPStdioClient sync wrapper around MCP SDK."""

    def test_call_tool_returns_text_result(self):
        """
        Given: MCPStdioClient with mocked async internals
        When: Calling a tool that returns text content
        Then: Returns the text as a string
        """
        client = MCPStdioClient(command="node", args=["server.js"])

        # Mock the internal async method
        with patch.object(client, "_call_tool_async") as mock_async:
            mock_async.return_value = "entity content here"
            result = client.call_tool("read_entity", {"path": "me"})

        assert result == "entity content here"

    def test_call_tool_passes_arguments(self):
        """
        Given: MCPStdioClient
        When: Calling a tool with specific arguments
        Then: Arguments are forwarded correctly
        """
        client = MCPStdioClient(command="node", args=["server.js"])

        with patch.object(client, "_call_tool_async") as mock_async:
            mock_async.return_value = "ok"
            client.call_tool("write_entity", {"path": "me", "content": "# Identity"})

        mock_async.assert_called_once_with(
            "write_entity", {"path": "me", "content": "# Identity"}
        )

    def test_stores_server_params(self):
        """
        Given: MCPStdioClient initialized with command, args, env
        Then: Server parameters are stored correctly
        """
        client = MCPStdioClient(
            command="node",
            args=["server.js"],
            env={"COGNITIVE_MEMORY_PATH": "/tmp/memory"},
        )

        assert client._server_params.command == "node"
        assert client._server_params.args == ["server.js"]
        assert client._server_params.env["COGNITIVE_MEMORY_PATH"] == "/tmp/memory"

    def test_call_tool_async_extracts_text_content(self):
        """
        Given: MCP server returns TextContent in call_tool result
        When: call_tool bridges sync→async
        Then: Extracts text from TextContent blocks and joins with newlines
        """
        # Build mock MCP result with TextContent
        mock_result = MagicMock()
        mock_result.content = [
            types.TextContent(type="text", text="first line"),
            types.TextContent(type="text", text="second line"),
        ]

        # Mock session
        mock_session = AsyncMock()
        mock_session.initialize = AsyncMock()
        mock_session.call_tool = AsyncMock(return_value=mock_result)

        # Mock ClientSession as async context manager
        @asynccontextmanager
        async def mock_client_session(read, write):
            yield mock_session

        # Mock stdio_client as async context manager
        @asynccontextmanager
        async def mock_stdio(params):
            yield ("mock_read", "mock_write")

        client = MCPStdioClient(command="node", args=["server.js"])

        with (
            patch(
                "codie_as_a_service.adapters.mcp.mcp_client.stdio_client",
                side_effect=mock_stdio,
            ),
            patch(
                "codie_as_a_service.adapters.mcp.mcp_client.ClientSession",
                side_effect=mock_client_session,
            ),
        ):
            result = client.call_tool("read_entity", {"path": "me"})

        assert result == "first line\nsecond line"
        mock_session.initialize.assert_called_once()
        mock_session.call_tool.assert_called_once_with(
            "read_entity", arguments={"path": "me"}
        )
