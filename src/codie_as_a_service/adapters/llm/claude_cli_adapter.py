"""
Adapter for Claude Code CLI as LLM backend.

Uses `claude -p` (print mode) with JSON output format.
Disables built-in tools and passes custom tool definitions via system prompt.
"""

import json
import re
import subprocess
from typing import Any

from codie_as_a_service.core.models import (
    ContentBlock,
    LLMResponse,
    Message,
    ToolDefinition,
    ToolUseBlock,
)


class ClaudeCliAdapter:
    """
    Adapter for Claude Code CLI using print mode.

    Uses subprocess to call `claude -p --output-format json --tools ""`.
    Tool definitions are passed via system prompt; tool calls are parsed
    from JSON patterns in the response text.
    """

    def call(
        self,
        messages: list[Message],
        system_prompt: str,
        tools: list[ToolDefinition] | None = None,
        output_format: dict[str, Any] | None = None,
    ) -> LLMResponse:
        """
        Call Claude CLI with messages and optional tools.

        Args:
            messages: Conversation history in domain format
            system_prompt: System prompt for the agent
            tools: Optional tool definitions (appended to system prompt)
            output_format: Optional JSON Schema for structured output (not used)

        Returns:
            Structured LLMResponse
        """
        # Build full system prompt with tool definitions
        full_system_prompt = self._build_system_prompt(system_prompt, tools)

        # Build conversation as single prompt
        prompt = self._build_prompt(messages)

        # Call Claude CLI
        result = self._run_claude(prompt, full_system_prompt)

        # Parse response
        return self._parse_response(result)

    def _build_system_prompt(
        self, base_prompt: str, tools: list[ToolDefinition] | None
    ) -> str:
        """Build system prompt with tool definitions appended."""
        if not tools:
            return base_prompt

        tool_descriptions = []
        for tool in tools:
            params = self._format_parameters(tool.input_schema)
            tool_descriptions.append(f"- {tool.name}({params}): {tool.description}")

        tools_section = """

## Available Tools

You have these tools available:
{tool_list}

When you need to use a tool, include this JSON in your response:
{{"tool_use": {{"name": "tool_name", "arguments": {{...}}}}}}

After using a tool, wait for the result before continuing.
When you have a final answer and don't need any tools, respond with plain text.
""".format(tool_list="\n".join(tool_descriptions))

        return base_prompt + tools_section

    def _format_parameters(self, schema: dict[str, Any]) -> str:
        """Format input schema as parameter string."""
        properties = schema.get("properties", {})
        required = set(schema.get("required", []))

        params = []
        for name, prop in properties.items():
            param_type = prop.get("type", "any")
            if name in required:
                params.append(f"{name}: {param_type}")
            else:  # pragma: no cover
                params.append(f"{name}?: {param_type}")

        return ", ".join(params)

    def _build_prompt(self, messages: list[Message]) -> str:
        """Build prompt string from messages."""
        parts = []
        for msg in messages:
            if msg.role == "user":
                parts.append(f"User: {msg.content}")
            elif msg.role == "assistant":
                parts.append(f"Assistant: {msg.content}")
        return "\n\n".join(parts)

    def _run_claude(self, prompt: str, system_prompt: str) -> str:  # pragma: no cover
        """
        Run Claude CLI and return result text.

        This method is mocked in tests - actual subprocess call not exercised.

        Args:
            prompt: The user prompt
            system_prompt: System prompt to use

        Returns:
            The result text from Claude's response
        """
        cmd = [
            "claude",
            "-p",
            "--output-format",
            "json",
            "--tools",
            "",
            "--system-prompt",
            system_prompt,
            prompt,
        ]

        result = subprocess.run(
            cmd,
            capture_output=True,
            text=True,
            timeout=120,
        )

        if result.returncode != 0:
            raise RuntimeError(
                f"Claude CLI failed (exit {result.returncode}):\n"
                f"stderr: {result.stderr[:500]}\n"
                f"stdout: {result.stdout[:500]}"
            )

        # Parse JSON response
        try:
            response = json.loads(result.stdout)
        except json.JSONDecodeError as e:
            raise RuntimeError(
                f"Claude CLI returned invalid JSON: {e}\n"
                f"stdout: {result.stdout[:500]!r}\n"
                f"stderr: {result.stderr[:500]!r}"
            )

        if response.get("is_error"):
            raise RuntimeError(
                f"Claude CLI error: {response.get('result', 'Unknown error')}"
            )

        return response.get("result", "")

    def _parse_response(self, text: str) -> LLMResponse:
        """
        Parse response text into LLMResponse.

        Extracts {"tool_use": ...} patterns and remaining text content.
        """
        content_blocks: list[ContentBlock | ToolUseBlock] = []

        # Parse tool calls
        tool_calls, remaining_text = self._parse_tool_calls(text)

        # Add tool call blocks
        for i, tool_call in enumerate(tool_calls):
            content_blocks.append(
                ToolUseBlock(
                    id=f"call_{i}",
                    name=tool_call["name"],
                    input=tool_call.get("arguments", {}),
                )
            )

        # Add text content if any remains
        if remaining_text.strip():
            content_blocks.append(ContentBlock(text=remaining_text.strip()))

        # Determine stop reason
        stop_reason = "tool_use" if tool_calls else "end_turn"

        return LLMResponse(stop_reason=stop_reason, content=content_blocks)

    def _parse_tool_calls(self, text: str) -> tuple[list[dict[str, Any]], str]:
        """
        Parse {"tool_use": ...} patterns from response text.

        Returns:
            Tuple of (list of parsed tool calls, remaining text without tool calls)
        """
        tool_calls = []
        remaining_text = text

        # Find all {"tool_use": {...}} patterns
        # Match complete JSON objects starting with {"tool_use":
        pattern = r'\{"tool_use":\s*(\{[^{}]*(?:\{[^{}]*\}[^{}]*)*\})\}'

        for match in re.finditer(pattern, text):
            inner_json = match.group(1)
            tool_call = json.loads(inner_json)
            tool_calls.append(tool_call)
            # Remove the matched pattern from remaining text
            remaining_text = remaining_text.replace(match.group(0), "", 1)

        return tool_calls, remaining_text
