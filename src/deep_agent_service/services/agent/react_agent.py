"""ReAct Agent - Reason, Act, Observe loop."""

import json
from typing import Any

from deep_agent_service.core.models import (
    ContentBlock,
    DEFAULT_OUTPUT_FORMAT,
    LLMResponse,
    Message,
    ToolDefinition,
    ToolUseBlock,
)
from deep_agent_service.core.protocols import (
    LLMProtocol,
    PromptProtocol,
)
from deep_agent_service.services.memory.memory_service import MemoryService


class ReActAgent:
    """
    ReAct agent implementing Reason -> Act -> Observe loop.

    Uses LLM to reason about user requests, execute tools, and generate responses.
    """

    def __init__(
        self,
        llm: LLMProtocol,
        prompts: PromptProtocol,
        memory: MemoryService,
        prompt_names: list[str],
        max_iterations: int = 10,
    ):
        """
        Initialize ReAct agent.

        Args:
            llm: LLM adapter for reasoning
            prompts: Prompt adapter for system prompts
            memory: Memory service for user data
            prompt_names: List of prompt names to fetch and combine for system prompt
            max_iterations: Maximum reasoning iterations before stopping
        """
        self._llm = llm
        self._prompts = prompts
        self._memory = memory
        self._prompt_names = prompt_names
        self._max_iterations = max_iterations

    def process(
        self, user_id: str, message: str, output_format: dict[str, Any] | None = None
    ) -> dict[str, Any]:
        """
        Process a user message through two-phase approach: ReAct loop, then structure output.

        Args:
            user_id: User identifier
            message: User's message
            output_format: Optional JSON Schema for structured output (uses default if None)

        Returns:
            Structured data dict (always)
        """
        # Phase 1: ReAct loop (always runs with tools)
        text_result = self._react_loop(user_id, message)

        # Phase 2: Structure output (always runs, uses default if no format specified)
        effective_format = output_format or DEFAULT_OUTPUT_FORMAT
        return self._structure_output(text_result, effective_format)

    def _react_loop(self, user_id: str, message: str) -> str:
        """
        Execute ReAct loop with tools, return text result.

        Args:
            user_id: User identifier
            message: User's message

        Returns:
            Text result from ReAct loop
        """
        # Load identity context for system prompt
        identity = self._memory.get_identity_context(user_id=user_id)

        # Validate assistant identity exists (me.md equivalent)
        if not identity.me:
            raise ValueError(f"No assistant identity configured for user '{user_id}'")

        # Build system prompt with identity by combining all configured prompts
        prompt_parts = []
        for prompt_name in self._prompt_names:
            prompt_part = self._prompts.get_prompt(
                prompt_name,
                me=identity.me,
                context_anchors=identity.context_anchors,
                current_session=identity.current_session,
            )
            prompt_parts.append(prompt_part)

        system_prompt = "\n\n".join(prompt_parts)

        # Initialize conversation with user message
        messages: list[Message] = [Message(role="user", content=message)]

        # Get tool definitions
        tools = self._get_tool_definitions()

        # ReAct loop
        collected_text: list[str] = []

        for _ in range(self._max_iterations):
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
                tool_results = self._execute_tools(user_id, response)

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

    def _structure_output(
        self, text_result: str, output_format: dict[str, Any]
    ) -> dict[str, Any]:
        """
        Transform text result into structured format via LLM call.

        Args:
            text_result: Text result from ReAct loop
            output_format: JSON Schema for structured output

        Returns:
            Structured data dict
        """
        messages = [
            Message(
                role="user",
                content=f"Extract and format the following information according to the schema:\n\n{text_result}",
            )
        ]

        response = self._llm.call(
            messages=messages,
            system_prompt="You are a data extraction assistant. Return the information in the exact format requested.",
            tools=None,
            output_format=output_format,
        )

        for block in response.content:
            if isinstance(block, ContentBlock):
                return json.loads(block.text)

        raise ValueError("No content in structured output response")

    def _get_tool_definitions(self) -> list[ToolDefinition]:
        """Get available tool definitions."""
        return [
            ToolDefinition(
                name="read_memory",
                description="Read user memory by key (e.g., 'current_session', 'context_anchors')",
                input_schema={
                    "type": "object",
                    "properties": {
                        "key": {
                            "type": "string",
                            "description": "Memory key to read",
                        }
                    },
                    "required": ["key"],
                },
            ),
            ToolDefinition(
                name="write_memory",
                description="Write content to user memory",
                input_schema={
                    "type": "object",
                    "properties": {
                        "key": {
                            "type": "string",
                            "description": "Memory key to write",
                        },
                        "content": {
                            "type": "string",
                            "description": "Content to write",
                        },
                    },
                    "required": ["key", "content"],
                },
            ),
        ]

    def _execute_tools(
        self, user_id: str, response: LLMResponse
    ) -> list[dict[str, str]]:
        """Execute tool calls from LLM response."""
        results = []

        for block in response.content:
            if isinstance(block, ToolUseBlock):
                result = self._execute_single_tool(user_id, block)
                results.append({"tool_use_id": block.id, "content": result})

        return results

    def _execute_single_tool(self, user_id: str, tool: ToolUseBlock) -> str:
        """Execute a single tool call."""
        if tool.name == "read_memory":
            key = tool.input.get("key", "")
            content = self._memory.read_memory(user_id=user_id, key=key)
            return content if content else f"No memory found for key: {key}"

        elif tool.name == "write_memory":
            key = tool.input.get("key", "")
            content = tool.input.get("content", "")
            self._memory.write_memory(user_id=user_id, key=key, content=content)
            return f"Successfully wrote to {key}"

        # Tools are hardcoded in _get_tool_definitions() - LLM can only request those.
        # If we reach here, the LLM hallucinated a tool name (shouldn't happen).
        raise ValueError(f"Unknown tool requested: {tool.name}")  # pragma: no cover

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
