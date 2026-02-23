"""
Adapter for local LLMs using Apple MLX.

This adapter uses mlx-lm for efficient inference on Apple Silicon.
SmolLM3's native tool calling via xml_tools parameter is preserved.
Tool execution happens internally — the adapter returns the final result.
"""

import json
import logging
from typing import Any

from mlx_lm import generate, load
from mlx_lm.sample_utils import make_sampler

from codie_as_a_service.core.models import (
    ContentBlock,
    LLMResponse,
    Message,
    ToolDefinition,
)


logger = logging.getLogger(__name__)


class LocalLLMAdapter:
    """
    Adapter for SmolLM3 using Apple MLX.

    Uses mlx-lm for 4-bit quantized inference on Apple Silicon.
    Uses apply_chat_template with xml_tools for native tool calling.
    Tool execution happens internally — returns final result with
    all tool calls already resolved.
    """

    def __init__(self, model_name: str, device: str = "mps"):
        """
        Initialize adapter with model from Hugging Face / MLX Community.

        Args:
            model_name: Model identifier (e.g., "mlx-community/SmolLM3-3B-Base-4bit")
            device: Kept for backward compatibility (ignored - MLX handles device placement)
        """
        logger.info("Loading model %s via MLX", model_name)
        self._model, self._tokenizer = load(model_name)
        logger.info("Model loaded successfully")

    DEFAULT_MAX_NEW_TOKENS = 2048

    def call(
        self,
        messages: list[Message],
        system_prompt: str,
        tools: list[ToolDefinition] | None = None,
        output_format: dict[str, Any] | None = None,
        max_new_tokens: int | None = None,
    ) -> LLMResponse:
        """
        Call local LLM using MLX.

        Tool execution happens internally via _generate. One call per phase —
        tools are resolved internally, result is the final output.

        Args:
            messages: Conversation history in domain format
            system_prompt: System prompt for the agent
            tools: Optional tool definitions (passed to xml_tools)
            output_format: Optional JSON Schema for structured output
            max_new_tokens: Max tokens to generate (default: 2048)

        Returns:
            Structured LLMResponse with final result (tools already resolved)
        """
        # Convert to chat format
        chat_messages = self._prepare_messages(messages, system_prompt)

        # Convert tools to SmolLM3 format
        smol_tools = self._convert_tools(tools) if tools else None

        # Apply chat template - tokenize=False gives us a string prompt
        prompt = self._tokenizer.apply_chat_template(
            chat_messages,
            xml_tools=smol_tools,
            enable_thinking=False,
            add_generation_prompt=True,
            tokenize=False,
        )

        effective_max_tokens = max_new_tokens or self.DEFAULT_MAX_NEW_TOKENS
        logger.info("Generating (max_new_tokens=%d)", effective_max_tokens)

        json_schema = None
        if output_format and output_format.get("type") == "json_schema":
            json_schema = output_format["schema"]

        # Single call — _generate handles tool loop internally in production
        text = self._generate(prompt, effective_max_tokens, json_schema=json_schema)

        # Return final result as text content (tools already resolved)
        data = json.loads(text) if output_format else None
        return LLMResponse(
            stop_reason="end_turn",
            content=[ContentBlock(text=text)],
            data=data,
        )

    def _generate(  # pragma: no cover
        self, prompt: str, max_tokens: int, *, json_schema: dict | None = None
    ) -> str:
        """Generate text from prompt using MLX.

        When json_schema is provided, uses Outlines to constrain generation
        to valid JSON matching the schema.
        """
        sampler = make_sampler(temp=0.0)
        if json_schema is not None:
            from outlines import from_mlxlm
            from outlines import json_schema as outlines_json_schema

            outlines_model = from_mlxlm(self._model, self._tokenizer)
            return outlines_model(
                prompt,
                output_type=outlines_json_schema(json_schema),
                max_tokens=max_tokens,
                sampler=sampler,
            )
        return generate(
            self._model,
            self._tokenizer,
            prompt=prompt,
            max_tokens=max_tokens,
            sampler=sampler,
        )

    def _prepare_messages(
        self, messages: list[Message], system_prompt: str
    ) -> list[dict[str, str]]:
        """Convert domain messages to chat format."""
        chat_messages = [{"role": "system", "content": system_prompt}]
        for msg in messages:
            chat_messages.append({"role": msg.role, "content": msg.content})
        return chat_messages

    def _convert_tools(self, tools: list[ToolDefinition]) -> list[dict[str, Any]]:
        """Convert domain ToolDefinitions to SmolLM3 format."""
        return [
            {
                "name": tool.name,
                "description": tool.description,
                "parameters": tool.input_schema,
            }
            for tool in tools
        ]
