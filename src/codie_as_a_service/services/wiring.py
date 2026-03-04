"""Shared wiring for orchestrator phases and tool definitions.

Both main_http and main_pubsub use these to build the orchestrator.
"""

from codie_as_a_service.core.models import ToolDefinition
from codie_as_a_service.core.phase_models import HydratedIdentity, ProcessedText
from codie_as_a_service.core.protocols import LLMProtocol, Phase, PromptProtocol
from codie_as_a_service.services.memory.memory_service import MemoryService
from codie_as_a_service.services.phases import (
    SynthesizePhaseDefinition,
    TextLLMPhaseDefinition,
)


def get_memory_tool_definitions() -> list[ToolDefinition]:
    """Get the standard memory tool definitions."""
    return [
        ToolDefinition(
            name="read_memory",
            description="Read agent memory by key (e.g., 'current_session', 'context_anchors')",
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
            description="Write content to agent memory",
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
        ToolDefinition(
            name="list_memory_keys",
            description="List all memory keys for the agent",
            input_schema={
                "type": "object",
                "properties": {},
            },
        ),
    ]


def get_mcp_tool_definitions() -> list[ToolDefinition]:
    """Get tool definitions for cognitive-memory MCP tools."""
    return [
        ToolDefinition(
            name="list_entities",
            description="List entities in long-term memory, optionally filtered by prefix",
            input_schema={
                "type": "object",
                "properties": {
                    "filter_prefix": {
                        "type": "string",
                        "description": "Optional prefix filter (e.g., 'people/', 'projects/')",
                        "default": "",
                    }
                },
            },
        ),
        ToolDefinition(
            name="read_entity",
            description="Read entity from long-term memory by path",
            input_schema={
                "type": "object",
                "properties": {
                    "entity_path": {
                        "type": "string",
                        "description": "Full path to entity (e.g., 'people/john-doe', 'projects/mcp-servers')",
                    }
                },
                "required": ["entity_path"],
            },
        ),
        ToolDefinition(
            name="write_entity",
            description="Write entity to long-term memory",
            input_schema={
                "type": "object",
                "properties": {
                    "entity_path": {
                        "type": "string",
                        "description": "Full path to entity (e.g., 'people/john-doe', 'concepts/learning')",
                    },
                    "content": {
                        "type": "string",
                        "description": "Content to write to entity",
                    },
                },
                "required": ["entity_path", "content"],
            },
        ),
        ToolDefinition(
            name="add_session_note",
            description="Add contextual note to current session",
            input_schema={
                "type": "object",
                "properties": {
                    "note_type": {
                        "type": "string",
                        "enum": ["context", "insight", "decision"],
                        "description": "Type of session note to add",
                    },
                    "content": {
                        "type": "string",
                        "description": "Note content to append to current session",
                    },
                },
                "required": ["note_type", "content"],
            },
        ),
    ]


def build_orchestrator_phases(
    phase_names: list[str],
    prompt_adapter: PromptProtocol,
    tools: list[ToolDefinition],
    llm: LLMProtocol,
    memory: MemoryService,
) -> tuple[list[Phase], list[Phase]]:
    """Build orchestrator phases filtered by phase_names list.

    Pipeline: HYDRATE → PROCESS (text) → SYNTHESIZE (post-phase)

    Args:
        phase_names: List of phase names to include (e.g., ["hydrate", "process"])
        prompt_adapter: Prompt adapter for fetching system prompts
        tools: Tool definitions to provide to phases
        llm: LLM adapter for phase execution
        memory: Memory service for persistence

    Returns:
        Tuple of (loop_phases, post_phases). Loop phases run each iteration;
        post_phases run once after the loop completes (done=true or max iterations).
    """
    phases: list[Phase] = []

    if "hydrate" in phase_names:
        phases.append(
            TextLLMPhaseDefinition(
                name="hydrate",
                llm=llm,
                system_prompt=prompt_adapter.get_prompt("orchestrator_hydrate"),
                output_schema=HydratedIdentity,
                skip_on_retry=True,
            )
        )

    if "process" in phase_names:
        phases.append(
            TextLLMPhaseDefinition(
                name="process",
                llm=llm,
                system_prompt=prompt_adapter.get_prompt("orchestrator_process"),
                output_schema=ProcessedText,
                tools=tools,
                max_new_tokens=32768,
            )
        )

    post_phases: list[Phase] = [
        SynthesizePhaseDefinition(
            name="synthesize",
            memory=memory,
        ),
    ]
    return phases, post_phases
