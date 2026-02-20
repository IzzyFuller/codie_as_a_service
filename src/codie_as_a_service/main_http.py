"""HTTP entry point for deep agent service."""

import json
import logging
import os
from typing import Any, Generator

import uvicorn
from dotenv import load_dotenv
from fastapi import Depends, FastAPI, Header, HTTPException
from fastapi.responses import StreamingResponse
from pydantic import BaseModel

from codie_as_a_service.adapters.auth.api_key_adapter import APIKeyAuthAdapter
from codie_as_a_service.adapters.llm.claude_cli_adapter import ClaudeCliAdapter
from codie_as_a_service.adapters.llm.local_llm_adapter import LocalLLMAdapter
from codie_as_a_service.adapters.prompts.file_adapter import FilePromptAdapter
from codie_as_a_service.adapters.storage.local_adapter import LocalMemoryAdapter
from codie_as_a_service.core.protocols import (
    AuthProtocol,
    LLMProtocol,
    MemoryProtocol,
    Phase,
    PromptProtocol,
)
from codie_as_a_service.core.models import ToolDefinition
from codie_as_a_service.core.phase_models import (
    ExtendedInstruction,
    HydratedIdentity,
    ProcessResult,
    ValidationResult,
)
from codie_as_a_service.services.agent.react_orchestrator import ReActOrchestrator
from codie_as_a_service.services.phases import (
    LLMPhaseDefinition,
    SynthesizePhaseDefinition,
    TextLLMPhaseDefinition,
)
from codie_as_a_service.services.memory.memory_service import MemoryService

load_dotenv()
logging.basicConfig(
    level=os.environ.get("LOG_LEVEL", "DEBUG").upper(),
    format="%(asctime)s %(levelname)s %(name)s: %(message)s",
)
logger = logging.getLogger(__name__)


class ChatRequest(BaseModel):
    """Request body for /chat endpoint."""

    agent_id: str
    session_id: str
    message: str
    output_format: dict[str, Any] | None = None


def create_app(
    memory_service: MemoryService,
    llm_adapter: LLMProtocol,
    prompt_adapter: PromptProtocol,
    prompt_names: list[str],
    auth: AuthProtocol,
    tools: list[ToolDefinition] | None = None,
) -> FastAPI:
    """
    Create FastAPI app with chat endpoint.

    Args:
        memory_service: Service for reading/writing agent memory
        llm_adapter: LLM adapter (handles tool execution internally)
        prompt_adapter: File-based prompt adapter
        prompt_names: List of prompt names to fetch and combine for system prompt
        auth: Authentication adapter for verifying requests
        tools: Optional custom tool definitions (default: memory tools only)

    Returns:
        Configured FastAPI application
    """
    app = FastAPI(title="Deep Agent Service")

    if tools is None:
        tools = _get_memory_tool_definitions()

    # Build orchestrator — each phase owns its execution
    phases = _build_orchestrator_phases(
        prompt_adapter, tools, llm_adapter, memory_service
    )
    orchestrator = ReActOrchestrator(
        memory=memory_service,
        phases=phases,
    )

    def verify_api_key(x_api_key: str | None = Header(None)) -> None:
        """FastAPI dependency to verify API key."""
        if x_api_key is None or not auth.verify(x_api_key):
            raise HTTPException(status_code=401, detail="Invalid or missing API key")

    def generate_sse_events(
        agent_id: str,
        session_id: str,
        message: str,
        output_format: dict[str, Any] | None = None,
    ) -> Generator[str, None, None]:
        """Generate SSE events for chat response."""
        try:
            # Process through orchestrator
            session_context = orchestrator.run(
                session_id=session_id,
                agent_id=agent_id,
                instruction=message,
                output_format=output_format,
            )

            # Emit structured response
            response_payload = {
                "output": session_context.response,
                "session_id": session_context.session_id,
                "done": session_context.done,
            }
            yield f"event: response\ndata: {json.dumps(response_payload)}\n\n"

            # Emit done event
            yield f"event: done\ndata: {json.dumps({'usage': {'input_tokens': 0, 'output_tokens': 0}})}\n\n"

        except Exception as e:
            # Emit error event
            yield f"event: error\ndata: {json.dumps({'message': str(e)})}\n\n"

    @app.get("/health")
    async def health() -> dict[str, str]:
        """Health check endpoint."""
        return {"status": "ok"}

    @app.post("/chat", dependencies=[Depends(verify_api_key)])
    async def chat(request: ChatRequest) -> StreamingResponse:
        """
        Process chat message and return streaming response.

        Streams SSE events:
        - event: response, data: {...} (structured response, always)
        - event: done, data: {"usage": {...}}
        - event: error, data: {"message": "..."}
        """
        return StreamingResponse(
            generate_sse_events(
                request.agent_id,
                request.session_id,
                request.message,
                request.output_format,
            ),
            media_type="text/event-stream",
        )

    return app


def _get_memory_tool_definitions() -> list[ToolDefinition]:
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


def _get_mcp_tool_definitions() -> list[ToolDefinition]:
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


def _build_orchestrator_phases(
    prompt_adapter: PromptProtocol,
    tools: list[ToolDefinition],
    llm: LLMProtocol,
    memory: MemoryService,
) -> list[Phase]:
    """Build the standard orchestrator phase definitions."""
    return [
        TextLLMPhaseDefinition(
            name="hydrate",
            llm=llm,
            system_prompt=prompt_adapter.get_prompt("orchestrator_hydrate"),
            output_schema=HydratedIdentity,
            skip_on_retry=True,
        ),
        LLMPhaseDefinition(
            name="extend",
            llm=llm,
            system_prompt=prompt_adapter.get_prompt("orchestrator_extend"),
            output_schema=ExtendedInstruction,
            tools=tools,
            skip_on_retry=True,
        ),
        LLMPhaseDefinition(
            name="process",
            llm=llm,
            system_prompt=prompt_adapter.get_prompt("orchestrator_process"),
            tools=tools,
            output_schema=ProcessResult,
        ),
        SynthesizePhaseDefinition(
            name="synthesize",
            memory=memory,
        ),
        LLMPhaseDefinition(
            name="validate",
            llm=llm,
            system_prompt=prompt_adapter.get_prompt("orchestrator_validate"),
            output_schema=ValidationResult,
        ),
    ]


def main() -> None:
    """Start the HTTP server."""
    # Configuration from environment
    host = os.environ.get("HTTP_HOST")
    if not host:
        raise ValueError("HTTP_HOST environment variable is required")

    port_str = os.environ.get("HTTP_PORT")
    if not port_str:
        raise ValueError("HTTP_PORT environment variable is required")
    port = int(port_str)

    llm_adapter_type = os.environ.get("LLM_ADAPTER", "claude_cli")

    prompts_dir = os.environ.get("PROMPTS_DIR")
    if not prompts_dir:
        raise ValueError("PROMPTS_DIR environment variable is required")

    prompt_names_str = os.environ.get("PROMPT_NAMES")
    if not prompt_names_str:
        raise ValueError("PROMPT_NAMES environment variable is required")
    prompt_names = [name.strip() for name in prompt_names_str.split(",")]

    api_key = os.environ.get("API_KEY")
    if not api_key:
        raise ValueError("API_KEY environment variable is required")

    # Path template for agent directory resolution (default: "agents/{agent_id}")
    # Set to empty string for flat directory (base_dir IS agent dir)
    storage_path_template = os.environ.get("STORAGE_PATH_TEMPLATE", "agents/{agent_id}")

    # Initialize storage adapter
    storage_dir = os.environ.get("STORAGE_DIR")
    if not storage_dir:
        raise ValueError("STORAGE_DIR environment variable is required")
    storage_adapter: MemoryProtocol = LocalMemoryAdapter(
        base_dir=storage_dir, agent_path_template=storage_path_template
    )

    # Initialize LLM adapter based on type
    if llm_adapter_type == "claude_cli":
        llm_adapter: LLMProtocol = ClaudeCliAdapter()
    elif llm_adapter_type == "local":
        model_name = os.environ.get("MODEL_NAME")
        if not model_name:
            raise ValueError("MODEL_NAME required when LLM_ADAPTER=local")
        device = os.environ.get("DEVICE", "mps")
        llm_adapter = LocalLLMAdapter(model_name=model_name, device=device)
    else:
        raise ValueError(f"Unknown LLM_ADAPTER: {llm_adapter_type}")

    prompt_adapter = FilePromptAdapter(prompts_dir=prompts_dir)

    # Build memory service
    memory_service = MemoryService(storage=storage_adapter)

    # Initialize auth adapter
    auth_adapter = APIKeyAuthAdapter(valid_key=api_key)

    # Determine tool definitions
    # When MCP is configured, cognitive-memory tools replace memory tools
    # Tool execution is handled by the adapter (Claude Code natively, local via internal loop)
    mcp_server_path = os.environ.get("MCP_SERVER_PATH")
    mcp_memory_path = os.environ.get("MCP_MEMORY_PATH")

    if mcp_server_path and mcp_memory_path:
        logger.info(
            "MCP tools enabled: %s (memory: %s)", mcp_server_path, mcp_memory_path
        )
        tools = _get_mcp_tool_definitions()
    else:
        tools = _get_memory_tool_definitions()

    # Create and run app
    app = create_app(
        memory_service=memory_service,
        llm_adapter=llm_adapter,
        prompt_adapter=prompt_adapter,
        prompt_names=prompt_names,
        auth=auth_adapter,
        tools=tools,
    )

    uvicorn.run(app, host=host, port=port)


if __name__ == "__main__":
    main()
