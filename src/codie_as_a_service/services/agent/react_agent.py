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
        max_iterations: int = 10,
        session_lines: int | None = 50,
    ):
        """
        Initialize ReAct agent.

        Args:
            llm: LLM adapter for reasoning
            prompts: Prompt adapter for system prompts
            memory: Memory service for user data
            prompt_names: List of prompt names to fetch and combine for system prompt
            tool_executor: Executor for handling tool calls
            tools: Tool definitions available to the agent
            max_iterations: Maximum reasoning iterations before stopping
            session_lines: Number of recent session lines to include (None for all)
        """
        self._llm = llm
        self._prompts = prompts
        self._memory = memory
        self._prompt_names = prompt_names
        self._tool_executor = tool_executor
        self._tools = tools
        self._max_iterations = max_iterations
        self._session_lines = session_lines

    def run_tool_loop(
        self,
        system_prompt: str,
        messages: list[Message],
        tools: list[ToolDefinition],
        tool_executor: ToolExecutor,
        user_id: str,
        max_iterations: int | None = None,
    ) -> str:
        """
        Execute a tool-calling loop: call LLM, execute tools, repeat until end_turn.

        This is the reusable mini-loop engine used by the orchestrator for
        tool-using phases (EXTEND, PROCESS, SYNTHESIZE).

        Args:
            system_prompt: System prompt for this loop
            messages: Initial messages (at minimum one user message)
            tools: Tool definitions available in this loop
            tool_executor: Executor for handling tool calls
            user_id: User identifier for scoped tool operations
            max_iterations: Override max iterations (defaults to self._max_iterations)

        Returns:
            Collected text from LLM responses
        """
        effective_max = (
            max_iterations if max_iterations is not None else self._max_iterations
        )
        return self._tool_loop(
            system_prompt, messages, tools, tool_executor, user_id, effective_max
        )

    def _tool_loop(
        self,
        system_prompt: str,
        messages: list[Message],
        tools: list[ToolDefinition],
        tool_executor: ToolExecutor,
        user_id: str,
        max_iterations: int,
    ) -> str:
        """
        Core tool-calling loop used by run_tool_loop.

        Args:
            system_prompt: System prompt for the LLM
            messages: Conversation messages
            tools: Tool definitions for the LLM
            tool_executor: Executor for tool calls
            user_id: User identifier
            max_iterations: Maximum loop iterations

        Returns:
            Collected text from LLM responses
        """
        collected_text: list[str] = []

        for _ in range(max_iterations):
            # Reason: Call LLM
            response = self._llm.call(
                messages=messages,
                system_prompt=system_prompt,
                tools=tools,
            )

            # Collect any text content
            for block in response.content:
                if isinstance(block, ContentBlock):
                    collected_text.append(block.text)

            # Check if done
            if response.stop_reason == "end_turn":
                break

            # Act: Execute tools if requested
            if response.stop_reason == "tool_use":
                tool_results = self._execute_tools(user_id, response, tool_executor)

                # Add assistant message with tool calls
                assistant_content = self._format_assistant_message(response)
                messages.append(Message(role="assistant", content=assistant_content))

                # Add tool results as user message (Anthropic convention)
                results_content = self._format_tool_results(tool_results)
                messages.append(Message(role="user", content=results_content))

        # Return collected text or default response
        return (
            " ".join(collected_text)
            if collected_text
            else "I couldn't complete the request."
        )

    def _execute_tools(
        self, user_id: str, response: LLMResponse, tool_executor: ToolExecutor
    ) -> list[dict[str, str]]:
        """Execute tool calls from LLM response."""
        results = []

        for block in response.content:
            if isinstance(block, ToolUseBlock):
                result = tool_executor.execute(
                    user_id=user_id, tool_name=block.name, tool_input=block.input
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
