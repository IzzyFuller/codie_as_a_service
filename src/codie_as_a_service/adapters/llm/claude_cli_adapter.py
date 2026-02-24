"""
Adapter for Claude Code CLI as LLM backend.

Uses `claude -p` (print mode) with JSON output format.
Tools are handled natively by Claude Code — no text injection needed.
"""

import json
import logging
import os
import subprocess
from typing import Any

from codie_as_a_service.core.models import (
    ContentBlock,
    LLMResponse,
    Message,
    ToolDefinition,
)


logger = logging.getLogger(__name__)


class ClaudeCliAdapter:
    """
    Adapter for Claude Code CLI using print mode.

    Uses subprocess to call `claude -p --output-format json`.
    Tools are exposed natively via Claude Code's tool system.
    Tool execution happens inside Claude Code — the adapter returns
    the final result with all tool calls already resolved.
    """

    def call(
        self,
        messages: list[Message],
        system_prompt: str,
        tools: list[ToolDefinition] = [],
        output_format: dict[str, Any] | None = None,
        max_new_tokens: int | None = None,
    ) -> LLMResponse:
        """
        Call Claude CLI with messages, tools, and optional structured output.

        Claude Code handles tool execution natively. One call per phase —
        tools are resolved internally, result is the final output.

        Args:
            messages: Conversation history in domain format
            system_prompt: System prompt for the agent
            tools: Optional tool definitions (handled natively by Claude Code)
            output_format: Optional JSON Schema for structured output (--json-schema)
            max_new_tokens: Ignored (Claude CLI manages token limits internally)

        Returns:
            Structured LLMResponse with final result (tools already resolved)
        """
        prompt = self._build_prompt(messages)
        logger.info(
            "Calling Claude CLI (prompt length: %d, system length: %d)",
            len(prompt),
            len(system_prompt),
        )
        logger.debug("Prompt: %.300s", prompt)

        # Extract JSON schema for structured output
        json_schema = None
        if output_format and output_format.get("type") == "json_schema":
            json_schema = output_format.get("schema")

        # Single call — Claude Code handles tools natively
        result = self._run_claude(
            prompt, system_prompt, json_schema=json_schema, tools=tools
        )
        logger.info("Claude CLI returned %d chars", len(result))
        logger.debug("Result: %.500s", result)

        # No tool call parsing needed — Claude Code resolved them internally
        data = json.loads(result) if output_format else None
        return LLMResponse(
            stop_reason="end_turn",
            content=[ContentBlock(text=result)],
            data=data,
        )

    def _build_prompt(self, messages: list[Message]) -> str:
        """Build prompt string from messages."""
        parts = []
        for msg in messages:
            prefix = "User" if msg.role == "user" else "Assistant"
            parts.append(f"{prefix}: {msg.content}")
        return "\n\n".join(parts)

    def _run_claude(  # pragma: no cover
        self,
        prompt: str,
        system_prompt: str,
        json_schema: dict[str, Any] | None = None,
        tools: list[ToolDefinition] = [],
    ) -> str:
        """
        Run Claude CLI and return result text.

        This method is mocked in tests - actual subprocess call not exercised.

        Args:
            prompt: The user prompt
            system_prompt: System prompt to use
            json_schema: Optional JSON schema to force structured output
            tools: Tool definitions — always passed via --allowedTools

        Returns:
            The result text from Claude's response
        """
        cmd = [
            "claude",
            "-p",
            "--output-format",
            "json",
            "--system-prompt",
            system_prompt,
        ]

        if json_schema is not None:
            cmd.extend(["--json-schema", json.dumps(json_schema)])

        cmd.extend(["--allowedTools", ",".join(t.name for t in tools)])

        result = subprocess.run(
            cmd,
            input=prompt,
            capture_output=True,
            text=True,
            timeout=int(os.environ.get("LLM_TIMEOUT", "300")),
        )

        logger.debug("Claude CLI exit code: %d", result.returncode)
        if result.stderr:
            logger.debug("Claude CLI stderr: %.500s", result.stderr)
        logger.debug("Claude CLI stdout: %.500s", result.stdout)

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

        logger.debug("Raw Claude JSON keys: %s", list(response.keys()))

        # --json-schema puts structured data in a separate field (as a dict)
        if json_schema is not None and "structured_output" in response:
            output = json.dumps(response["structured_output"])
            logger.debug("Structured output: %r", output[:500])
            return output

        output = response.get("result", "")
        logger.debug("Raw result value: %r", output[:500] if output else output)
        return output
