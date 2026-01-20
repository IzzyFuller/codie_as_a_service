"""Anthropic SDK adapter implementing LLMProtocol."""

from typing import Any

from anthropic import Anthropic
from anthropic.types import Message as AnthropicMessage

from deep_agent_service.core.models import (
    ContentBlock,
    LLMResponse,
    Message,
    ToolDefinition,
    ToolUseBlock,
)


class AnthropicLLMAdapter:
    """Adapter for Anthropic Claude API."""

    def __init__(self, client: Anthropic, model: str):
        """
        Initialize adapter with Anthropic client.

        Args:
            client: Anthropic SDK client instance
            model: Model identifier to use
        """
        self._client = client
        self._model = model

    def call(
        self,
        messages: list[Message],
        system_prompt: str,
        tools: list[ToolDefinition] | None = None,
        output_format: dict[str, Any] | None = None,
    ) -> LLMResponse:
        """
        Call Claude API with messages and optional tools.

        Args:
            messages: Conversation history in domain format
            system_prompt: System prompt for the agent
            tools: Optional tool definitions
            output_format: Optional JSON Schema for structured output

        Returns:
            Structured LLMResponse
        """
        # Convert messages to Anthropic format
        anthropic_messages = self._convert_messages_to_anthropic_format(messages)

        # Build API call kwargs
        kwargs: dict[str, Any] = {
            "model": self._model,
            "max_tokens": 4096,
            "system": system_prompt,
            "messages": anthropic_messages,
        }

        # Add tools if provided
        if tools:
            kwargs["tools"] = self._convert_tools_to_anthropic_format(tools)

        # Add structured output if provided
        if output_format:
            kwargs["betas"] = ["structured-outputs-2025-11-13"]
            kwargs["output_format"] = output_format

        # Call the API
        response = self._client.messages.create(**kwargs)

        # Convert response to domain format
        return self._convert_response_to_domain_format(response)

    def _convert_messages_to_anthropic_format(
        self, messages: list[Message]
    ) -> list[dict[str, Any]]:
        """Convert domain Messages to Anthropic format."""
        return [{"role": msg.role, "content": msg.content} for msg in messages]

    def _convert_tools_to_anthropic_format(
        self, tools: list[ToolDefinition]
    ) -> list[dict[str, Any]]:
        """Convert domain ToolDefinitions to Anthropic format."""
        return [
            {
                "name": tool.name,
                "description": tool.description,
                "input_schema": tool.input_schema,
            }
            for tool in tools
        ]

    def _convert_response_to_domain_format(
        self, response: AnthropicMessage
    ) -> LLMResponse:
        """Convert Anthropic response to domain LLMResponse."""
        content_blocks: list[ContentBlock | ToolUseBlock] = []

        for block in response.content:
            if block.type == "text":
                content_blocks.append(ContentBlock(text=block.text))
            elif block.type == "tool_use":
                content_blocks.append(
                    ToolUseBlock(
                        id=block.id,
                        name=block.name,
                        input=block.input,
                    )
                )

        return LLMResponse(
            stop_reason=response.stop_reason,
            content=content_blocks,
        )
