"""ReAct Agent - Reason, Act, Observe loop."""

import logging

from codie_as_a_service.core.models import (
    ContentBlock,
    LLMResponse,
    Message,
    ToolDefinition,
    ToolUseBlock,
)
from codie_as_a_service.core.protocols import (
    LLMProtocol,
    PromptProtocol,
    ToolExecutor,
)
from codie_as_a_service.services.memory.memory_service import MemoryService

logger = logging.getLogger(__name__)

_TOOL_LOOP_SAFETY_LIMIT = 100


class ReActAgent:
    """
    ReAct agent implementing Reason -> Act -> Observe loop.

    Uses LLM to reason about user requests, execute tools, and generate responses.
    Also serves as reusable mini-loop engine for the orchestrator via run_tool_loop().
    """

    def __init__(
        self,
        llm: LLMProtocol,
        prompts: PromptProtocol,
        memory: MemoryService,
        prompt_names: list[str],
        tool_executor: ToolExecutor,
        tools: list[ToolDefinition],
        session_lines: int | None = 50,
    ):
        """
        Initialize ReAct agent.

        Args:
            llm: LLM adapter for reasoning
            prompts: Prompt adapter for system prompts
            memory: Memory service for agent data
            prompt_names: List of prompt names to fetch and combine for system prompt
            tool_executor: Executor for handling tool calls
            tools: Tool definitions available to the agent
            session_lines: Number of recent session lines to include (None for all)
        """
        self._llm = llm
        self._prompts = prompts
        self._memory = memory
        self._prompt_names = prompt_names
        self._tool_executor = tool_executor
        self._tools = tools
        self._session_lines = session_lines

    def run_tool_loop(
        self,
        system_prompt: str,
        messages: list[Message],
        tools: list[ToolDefinition],
        tool_executor: ToolExecutor,
        agent_id: str,
        max_new_tokens: int | None = None,
    ) -> list[Message]:
        """
        Execute tool calls until the model stops requesting tools.

        Runs the LLM with tools until it returns end_turn. Each tool_use
        response is executed and results fed back. Returns the enriched
        messages list for a subsequent schema-constrained call.

        Args:
            system_prompt: System prompt for this loop
            messages: Initial messages (at minimum one user message)
            tools: Tool definitions available in this loop
            tool_executor: Executor for handling tool calls
            agent_id: Agent identifier for scoped tool operations
            max_new_tokens: Max tokens per LLM call (passed through to adapter)

        Returns:
            Messages enriched with tool call/result exchanges
        """
        for _ in range(_TOOL_LOOP_SAFETY_LIMIT):
            response = self._llm.call(
                messages=messages,
                system_prompt=system_prompt,
                tools=tools,
                max_new_tokens=max_new_tokens,
            )

            if response.stop_reason != "tool_use":
                break

            tool_results = self._execute_tools(agent_id, response, tool_executor)

            messages.append(
                Message(
                    role="assistant",
                    content=self._format_assistant_message(response),
                )
            )
            messages.append(
                Message(
                    role="user",
                    content=self._format_tool_results(tool_results),
                )
            )

        return messages

    def _execute_tools(
        self, agent_id: str, response: LLMResponse, tool_executor: ToolExecutor
    ) -> list[dict[str, str]]:
        """Execute tool calls from LLM response."""
        results = []

        for block in response.content:
            if isinstance(block, ToolUseBlock):
                result = tool_executor.execute(
                    agent_id=agent_id, tool_name=block.name, tool_input=block.input
                )
                results.append({"tool_use_id": block.id, "content": result})

        return results

    def _format_assistant_message(self, response: LLMResponse) -> str:
        """Format LLM response as assistant message content."""
        parts = []
        for block in response.content:
            if isinstance(block, ContentBlock):
                parts.append(block.text)
            elif isinstance(block, ToolUseBlock):
                parts.append(f"[Calling tool: {block.name}]")
        return " ".join(parts)

    def _format_tool_results(self, results: list[dict[str, str]]) -> str:
        """Format tool results as message content."""
        parts = []
        for result in results:
            parts.append(f"Tool result: {result['content']}")
        return "\n".join(parts)
