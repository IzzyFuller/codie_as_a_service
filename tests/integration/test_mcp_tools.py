"""
Tests for MCP tool execution components.

Tests MCPStdioClient, MCPToolExecutor, and CompositeToolExecutor.
MCPStdioClient is mocked at the boundary - we don't spawn real MCP servers in tests.
"""

from contextlib import asynccontextmanager
from unittest.mock import AsyncMock, MagicMock, patch

import pytest

from mcp import types

from codie_as_a_service.adapters.mcp.mcp_client import MCPStdioClient
from codie_as_a_service.services.tools.mcp_tool_executor import MCPToolExecutor
from codie_as_a_service.services.tools.composite_tool_executor import (
    CompositeToolExecutor,
)


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


class TestMCPToolExecutor:
    """Tests for MCPToolExecutor implementing ToolExecutor protocol."""

    def test_execute_delegates_to_mcp_client(self):
        """
        Given: MCPToolExecutor with mocked MCP client
        When: Executing a tool
        Then: Delegates to client.call_tool with correct args
        """
        mock_client = MagicMock(spec=MCPStdioClient)
        mock_client.call_tool.return_value = "entity data"
        executor = MCPToolExecutor(mcp_client=mock_client)

        result = executor.execute(
            agent_id="tess",
            tool_name="read_entity",
            tool_input={"path": "me"},
        )

        assert result == "entity data"
        mock_client.call_tool.assert_called_once_with("read_entity", {"path": "me"})

    def test_execute_passes_all_arguments(self):
        """
        Given: MCPToolExecutor with mocked MCP client
        When: Executing write_entity tool
        Then: All arguments passed through
        """
        mock_client = MagicMock(spec=MCPStdioClient)
        mock_client.call_tool.return_value = "written"
        executor = MCPToolExecutor(mcp_client=mock_client)

        result = executor.execute(
            agent_id="tess",
            tool_name="write_entity",
            tool_input={"path": "me", "content": "# New content"},
        )

        assert result == "written"
        mock_client.call_tool.assert_called_once_with(
            "write_entity", {"path": "me", "content": "# New content"}
        )

    def test_agent_id_not_passed_to_mcp(self):
        """
        Given: MCPToolExecutor
        When: Executing any tool
        Then: agent_id is NOT passed to MCP (MCP handles its own scoping)
        """
        mock_client = MagicMock(spec=MCPStdioClient)
        mock_client.call_tool.return_value = "ok"
        executor = MCPToolExecutor(mcp_client=mock_client)

        executor.execute(agent_id="tess", tool_name="list_entities", tool_input={})

        # agent_id should NOT appear in the MCP call
        mock_client.call_tool.assert_called_once_with("list_entities", {})


class TestCompositeToolExecutor:
    """Tests for CompositeToolExecutor routing."""

    def test_routes_to_correct_executor_by_tool_name(self):
        """
        Given: CompositeToolExecutor with multiple executors registered
        When: Executing different tools
        Then: Each tool routes to its registered executor
        """
        memory_exec = MagicMock()
        memory_exec.execute.return_value = "memory result"
        mcp_exec = MagicMock()
        mcp_exec.execute.return_value = "mcp result"

        composite = CompositeToolExecutor(
            executors={
                "read_memory": memory_exec,
                "write_memory": memory_exec,
                "read_entity": mcp_exec,
                "write_entity": mcp_exec,
            }
        )

        result1 = composite.execute("tess", "read_memory", {"key": "me"})
        result2 = composite.execute("tess", "read_entity", {"path": "me"})

        assert result1 == "memory result"
        assert result2 == "mcp result"
        memory_exec.execute.assert_called_once_with(
            "tess", "read_memory", {"key": "me"}
        )
        mcp_exec.execute.assert_called_once_with("tess", "read_entity", {"path": "me"})

    def test_raises_value_error_for_unknown_tool(self):
        """
        Given: CompositeToolExecutor with registered tools
        When: Executing an unregistered tool
        Then: Raises ValueError
        """
        memory_exec = MagicMock()
        composite = CompositeToolExecutor(
            executors={
                "read_memory": memory_exec,
            }
        )

        with pytest.raises(ValueError, match="No executor for tool: unknown_tool"):
            composite.execute("tess", "unknown_tool", {})

    def test_empty_executors_raises_for_any_tool(self):
        """
        Given: CompositeToolExecutor with no registered executors
        When: Executing any tool
        Then: Raises ValueError
        """
        composite = CompositeToolExecutor(executors={})

        with pytest.raises(ValueError, match="No executor for tool"):
            composite.execute("tess", "read_memory", {})
