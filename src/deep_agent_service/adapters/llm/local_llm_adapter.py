"""
Adapter for local LLMs served via OpenAI-compatible APIs.

This adapter is designed for local language models (e.g., SmolLM3-3B via mlx_lm.server)
that expose an OpenAI-compatible HTTP interface but have different behavior from
cloud-hosted models:

- Text responses may be plain text, not JSON (even when asked for JSON)
- Responses may be wrapped in markdown code blocks
- Tool call arguments are always JSON strings (OpenAI API spec)

The adapter handles these local model quirks, normalizing responses to the
standard format expected by the service layer.
"""

import json
import logging
import re
from typing import Any

from openai import OpenAI
from openai.types.chat import ChatCompletion

from deep_agent_service.core.models import (
    ContentBlock,
    LLMResponse,
    Message,
    ToolDefinition,
    ToolUseBlock,
)

logger = logging.getLogger(__name__)


class LocalLLMAdapter:
    """
    Adapter for local LLMs served via OpenAI-compatible APIs.

    Designed for models like SmolLM3-3B running on mlx_lm.server.
    Handles local model quirks:
    - Normalizes plain text responses to JSON when structured output requested
    - Extracts JSON from markdown code blocks
    - Parses tool call arguments (always JSON strings per OpenAI API spec)
    """

    def __init__(self, base_url: str, model: str, timeout: float = 120.0):
        """
        Initialize adapter for local LLM.

        Args:
            base_url: Base URL for the local LLM server (e.g., http://localhost:8080/v1)
            model: Model identifier as recognized by the local server
            timeout: Request timeout in seconds (default 120s - local models can be slow)
        """
        self._client = OpenAI(base_url=base_url, api_key="not-needed", timeout=timeout)
        self._model = model

    def call(
        self,
        messages: list[Message],
        system_prompt: str,
        tools: list[ToolDefinition] | None = None,
        output_format: dict[str, Any] | None = None,
    ) -> LLMResponse:
        """
        Call local LLM via OpenAI-compatible API.

        Args:
            messages: Conversation history in domain format
            system_prompt: System prompt for the agent
            tools: Optional tool definitions
            output_format: Optional JSON Schema for structured output.
                          Local models don't support native structured output,
                          so this adapter normalizes text responses to valid JSON.

        Returns:
            Structured LLMResponse with normalized content
        """
        api_messages = self._prepare_messages(messages, system_prompt)

        kwargs: dict[str, Any] = {
            "model": self._model,
            "max_tokens": 4096,
            "messages": api_messages,
        }

        if tools:
            kwargs["tools"] = self._convert_tools_to_openai_format(tools)

        response = self._client.chat.completions.create(**kwargs)

        return self._convert_response_to_domain_format(response, output_format)

    def _prepare_messages(
        self, messages: list[Message], system_prompt: str
    ) -> list[dict[str, Any]]:
        """Prepare messages including system prompt for OpenAI format."""
        api_messages: list[dict[str, Any]] = [
            {"role": "system", "content": system_prompt}
        ]

        for msg in messages:
            api_messages.append({"role": msg.role, "content": msg.content})

        return api_messages

    def _convert_tools_to_openai_format(
        self, tools: list[ToolDefinition]
    ) -> list[dict[str, Any]]:
        """Convert domain ToolDefinitions to OpenAI format."""
        return [
            {
                "type": "function",
                "function": {
                    "name": tool.name,
                    "description": tool.description,
                    "parameters": tool.input_schema,
                },
            }
            for tool in tools
        ]

    def _convert_response_to_domain_format(
        self,
        response: ChatCompletion,
        output_format: dict[str, Any] | None = None,
    ) -> LLMResponse:
        """
        Convert local LLM response to domain LLMResponse.

        Handles local model quirks:
        - Normalizes text to JSON when output_format specified
        - Parses tool call arguments (always JSON strings)
        """
        content_blocks: list[ContentBlock | ToolUseBlock] = []

        choice = response.choices[0]

        if choice.message.content:
            text = choice.message.content
            # Local models don't support native structured output - normalize text to JSON
            if output_format:
                text = self._normalize_to_json(text)
            content_blocks.append(ContentBlock(text=text))

        # Tool calls: OpenAI API always returns arguments as JSON string
        if choice.message.tool_calls:
            for tool_call in choice.message.tool_calls:
                if tool_call.type == "function":
                    raw_args = tool_call.function.arguments
                    args = json.loads(raw_args) if raw_args else {}
                    content_blocks.append(
                        ToolUseBlock(
                            id=tool_call.id,
                            name=tool_call.function.name,
                            input=args,
                        )
                    )

        stop_reason = self._map_finish_reason_to_stop_reason(choice.finish_reason)

        return LLMResponse(
            stop_reason=stop_reason,
            content=content_blocks,
        )

    def _normalize_to_json(self, text: str) -> str:
        """
        Normalize local LLM text response to valid JSON string.

        Local models often return plain text or markdown-wrapped JSON
        even when asked for JSON. This method handles:
        1. Already valid JSON -> return as-is
        2. JSON in markdown code blocks -> extract it
        3. Plain text -> wrap in {"response": "..."}

        Args:
            text: Raw text from local LLM response

        Returns:
            Valid JSON string
        """
        text = text.strip()

        # Try 1: Already valid JSON
        try:
            json.loads(text)
            logger.debug("_normalize_to_json: text is already valid JSON")
            return text
        except json.JSONDecodeError:
            pass

        # Try 2: Extract JSON from markdown code blocks (common local model behavior)
        if "```" in text:
            match = re.search(r"```(?:json)?\s*(\{.*?\})\s*```", text, re.DOTALL)
            if match:
                extracted = match.group(1)
                try:
                    json.loads(extracted)
                    logger.debug("_normalize_to_json: extracted JSON from markdown")
                    return extracted
                except json.JSONDecodeError:
                    pass

        # Fallback: Wrap raw text in standard response format
        logger.debug("_normalize_to_json: wrapping raw text in JSON structure")
        return json.dumps({"response": text})

    def _map_finish_reason_to_stop_reason(self, finish_reason: str) -> str:
        """Map OpenAI finish_reason to domain stop_reason."""
        mapping = {
            "stop": "end_turn",
            "tool_calls": "tool_use",
            "length": "max_tokens",
        }
        return mapping.get(finish_reason, "end_turn")
