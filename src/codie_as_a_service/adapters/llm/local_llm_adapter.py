"""
Adapter for local LLMs using Hugging Face Transformers.

This adapter uses SmolLM3's native Transformers API with direct tool calling
support via xml_tools parameter. No OpenAI compatibility layer needed.
"""

import json
import re
from typing import Any

from transformers import AutoModelForCausalLM, AutoTokenizer

from codie_as_a_service.core.models import (
    ContentBlock,
    LLMResponse,
    Message,
    ToolDefinition,
    ToolUseBlock,
)


class LocalLLMAdapter:
    """
    Adapter for SmolLM3 using native Transformers API.

    Uses apply_chat_template with xml_tools for native tool calling.
    Parses <tool_call> XML tags from model output.
    """

    def __init__(self, model_name: str, device: str = "mps"):
        """
        Initialize adapter with model from Hugging Face.

        Args:
            model_name: Hugging Face model identifier (e.g., "HuggingFaceTB/SmolLM3-3B")
            device: Device to run on ("mps" for Apple Silicon, "cuda", or "cpu")
        """
        self._tokenizer = AutoTokenizer.from_pretrained(model_name)
        self._model = AutoModelForCausalLM.from_pretrained(model_name).to(device)
        self._device = device

    def call(
        self,
        messages: list[Message],
        system_prompt: str,
        tools: list[ToolDefinition] | None = None,
        output_format: dict[str, Any] | None = None,
    ) -> LLMResponse:
        """
        Call local LLM using Transformers API.

        Args:
            messages: Conversation history in domain format
            system_prompt: System prompt for the agent
            tools: Optional tool definitions (passed to xml_tools)
            output_format: Optional JSON Schema for structured output

        Returns:
            Structured LLMResponse
        """
        # Convert to chat format
        chat_messages = self._prepare_messages(messages, system_prompt)

        # Convert tools to SmolLM3 format
        smol_tools = self._convert_tools(tools) if tools else None

        # Apply chat template with tools - get both input_ids and attention_mask
        tokenized = self._tokenizer.apply_chat_template(
            chat_messages,
            xml_tools=smol_tools,
            enable_thinking=False,
            add_generation_prompt=True,
            tokenize=True,
            return_tensors="pt",
            return_dict=True,
        )
        input_ids = tokenized["input_ids"].to(self._device)
        attention_mask = tokenized["attention_mask"].to(self._device)

        # Generate with explicit attention mask
        outputs = self._model.generate(
            input_ids,
            attention_mask=attention_mask,
            max_new_tokens=4096,
            temperature=0.6,
            top_p=0.95,
            pad_token_id=self._tokenizer.eos_token_id,
        )

        # Decode only the new tokens (not the input)
        new_tokens = outputs[0][input_ids.shape[-1] :]
        text = self._tokenizer.decode(new_tokens, skip_special_tokens=True)

        return self._parse_response(text)

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

    def _parse_response(self, text: str) -> LLMResponse:
        """
        Parse model output into LLMResponse.

        Extracts <tool_call> tags and remaining text content.
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
        Parse <tool_call> tags from SmolLM3 response.

        Returns:
            Tuple of (list of parsed tool calls, remaining text without tool calls)
        """
        tool_calls = []
        remaining_text = text

        # Find all <tool_call>...</tool_call> patterns
        pattern = r"<tool_call>\s*(\{.*?\})\s*</tool_call>"
        matches = re.findall(pattern, text, re.DOTALL)

        for match in matches:
            tool_call = json.loads(match)
            tool_calls.append(tool_call)
            remaining_text = re.sub(
                r"<tool_call>\s*" + re.escape(match) + r"\s*</tool_call>",
                "",
                remaining_text,
                count=1,
            )

        return tool_calls, remaining_text
