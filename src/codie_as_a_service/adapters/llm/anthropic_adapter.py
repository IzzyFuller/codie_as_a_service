"""
Adapter for Anthropic Messages API as LLM backend.

Uses the anthropic Python SDK for direct API calls.
Follows the same mockable-boundary pattern as ClaudeCliAdapter
and LocalLLMAdapter.
"""

import logging
from typing import Any

import anthropic
from pydantic import BaseModel

from codie_as_a_service.core.models import (
    Message,
    ToolDefinition,
)


logger = logging.getLogger(__name__)


class AnthropicAPIAdapter:
    """
    Adapter for Anthropic Messages API.

    Uses the anthropic Python SDK for inference.
    _create_message() is the mockable boundary (like _run_claude
    and _generate in other adapters).  Returns plain text.
    """

    DEFAULT_MAX_TOKENS = 4096

    def __init__(self, api_key: str, model: str = "claude-sonnet-4-20250514"):
        """
        Initialize adapter with Anthropic API credentials.

        Args:
            api_key: Anthropic API key
            model: Model identifier (default: claude-sonnet-4-20250514)
        """
        self._client = anthropic.Anthropic(api_key=api_key)
        self._model = model

    def call(
        self,
        messages: list[Message],
        system_prompt: str,
        tools: list[ToolDefinition] | None = None,
        output_model: type[BaseModel] | None = None,
        max_new_tokens: int | None = None,
    ) -> BaseModel | str:
        """
        Call Anthropic Messages API with messages and optional structured output.

        Args:
            messages: Conversation history in domain format
            system_prompt: System prompt for the agent
            tools: Optional tool definitions (converted to Anthropic format)
            output_model: Optional Pydantic model for structured output
            max_new_tokens: Max tokens to generate (default: 4096)

        Returns:
            Validated Pydantic model when output_model is given, plain text otherwise
        """
        api_messages = self._prepare_messages(messages)
        api_tools = self._convert_tools(tools) if tools else None
        effective_max_tokens = max_new_tokens or self.DEFAULT_MAX_TOKENS

        logger.info(
            "Calling Anthropic API (model=%s, messages=%d, max_tokens=%d)",
            self._model,
            len(api_messages),
            effective_max_tokens,
        )

        text = self._create_message(
            messages=api_messages,
            system_prompt=system_prompt,
            max_tokens=effective_max_tokens,
            tools=api_tools,
        )

        logger.info("Anthropic API returned %d chars", len(text))
        logger.debug("Result: %.500s", text)

        # Return validated model or plain text
        if output_model is not None:
            return output_model.model_validate_json(text)
        return text

    def _create_message(  # pragma: no cover
        self,
        messages: list[dict[str, str]],
        system_prompt: str,
        max_tokens: int,
        tools: list[dict[str, Any]] | None = None,
    ) -> str:
        """
        Call Anthropic Messages API and return text response.

        This method is mocked in tests -- actual API call not exercised.
        Returns plain text (same contract as _run_claude and _generate
        in the other adapters).

        Args:
            messages: Messages in Anthropic API format
            system_prompt: System prompt
            max_tokens: Maximum tokens to generate
            tools: Optional tool definitions in Anthropic format

        Returns:
            Response text extracted from Anthropic Message content blocks
        """
        kwargs: dict[str, Any] = {
            "model": self._model,
            "max_tokens": max_tokens,
            "system": system_prompt,
            "messages": messages,
        }
        if tools:
            kwargs["tools"] = tools

        response = self._client.messages.create(**kwargs)

        # Extract text from response content blocks
        parts = []
        for block in response.content:
            if block.type == "text":
                parts.append(block.text)
            elif block.type == "tool_use":
                logger.warning(
                    "Tool use block in response (tool=%s) -- skipping. "
                    "Tool execution is not handled by this adapter.",
                    block.name,
                )
        return "".join(parts)

    def _prepare_messages(self, messages: list[Message]) -> list[dict[str, str]]:
        """Convert domain messages to Anthropic API format.

        Anthropic API requires role to be 'user' or 'assistant'.
        System messages are folded into the system prompt parameter.
        """
        api_messages = []
        for msg in messages:
            if msg.role == "system":
                # System messages are handled via the system parameter
                continue
            api_messages.append({"role": msg.role, "content": msg.content})
        return api_messages

    def _convert_tools(self, tools: list[ToolDefinition]) -> list[dict[str, Any]]:
        """Convert domain ToolDefinitions to Anthropic API format."""
        return [
            {
                "name": tool.name,
                "description": tool.description,
                "input_schema": tool.input_schema,
            }
            for tool in tools
        ]
