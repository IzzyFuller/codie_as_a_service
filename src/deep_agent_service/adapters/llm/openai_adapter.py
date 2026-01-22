"""OpenAI SDK adapter implementing LLMProtocol."""

import json
import logging
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


class OpenAILLMAdapter:
    """Adapter for OpenAI API and OpenAI-compatible endpoints (e.g., mlx_lm.server)."""

    def __init__(self, base_url: str, model: str, timeout: float = 120.0):
        """
        Initialize adapter with base URL and model name.

        Args:
            base_url: Base URL for the OpenAI-compatible API endpoint
            model: Model identifier to use
            timeout: Request timeout in seconds (default 120s for local LLMs)
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
        Call OpenAI-compatible API with messages and optional tools.

        Args:
            messages: Conversation history in domain format
            system_prompt: System prompt for the agent
            tools: Optional tool definitions
            output_format: Optional JSON Schema for structured output (not used with OpenAI)

        Returns:
            Structured LLMResponse
        """
        # Prepare messages with system prompt
        api_messages = self._prepare_messages(messages, system_prompt)

        # Build API call kwargs
        kwargs: dict[str, Any] = {
            "model": self._model,
            "max_tokens": 4096,
            "messages": api_messages,
        }

        # Add tools if provided
        if tools:
            kwargs["tools"] = self._convert_tools_to_openai_format(tools)

        # Call the API
        response = self._client.chat.completions.create(**kwargs)

        # Convert response to domain format
        return self._convert_response_to_domain_format(response)

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
        self, response: ChatCompletion
    ) -> LLMResponse:
        """Convert OpenAI response to domain LLMResponse."""
        content_blocks: list[ContentBlock | ToolUseBlock] = []

        # Extract content from the first choice
        choice = response.choices[0]

        if choice.message.content:
            content_blocks.append(ContentBlock(text=choice.message.content))

        # Extract tool calls if any
        if choice.message.tool_calls:
            for tool_call in choice.message.tool_calls:
                if tool_call.type == "function":
                    # OpenAI returns arguments as JSON string, parse it
                    args = tool_call.function.arguments
                    logger.debug(
                        f"Tool call {tool_call.function.name}: raw args={args!r}"
                    )
                    if isinstance(args, str):
                        try:
                            args = json.loads(args) if args else {}
                        except json.JSONDecodeError as e:
                            logger.error(
                                f"Failed to parse tool args: {e}, raw={args!r}"
                            )
                            args = {}
                    content_blocks.append(
                        ToolUseBlock(
                            id=tool_call.id,
                            name=tool_call.function.name,
                            input=args,
                        )
                    )

        # Map OpenAI finish_reason to domain stop_reason
        stop_reason = self._map_finish_reason_to_stop_reason(choice.finish_reason)

        return LLMResponse(
            stop_reason=stop_reason,
            content=content_blocks,
        )

    def _map_finish_reason_to_stop_reason(self, finish_reason: str) -> str:
        """Map OpenAI finish_reason to domain stop_reason."""
        mapping = {
            "stop": "end_turn",
            "tool_calls": "tool_use",
            "length": "max_tokens",
        }
        return mapping.get(finish_reason, "end_turn")
