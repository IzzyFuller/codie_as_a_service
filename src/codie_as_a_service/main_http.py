"""HTTP entry point for deep agent service."""

import json
import logging
import os
import uuid
from typing import Any, Generator

import uvicorn
from dotenv import load_dotenv
from fastapi import FastAPI
from fastapi.responses import StreamingResponse
from pydantic import BaseModel

from codie_as_a_service.adapters.llm.claude_cli_adapter import ClaudeCliAdapter
from codie_as_a_service.adapters.llm.local_llm_adapter import LocalLLMAdapter
from codie_as_a_service.adapters.prompts.file_adapter import FilePromptAdapter
from codie_as_a_service.adapters.storage.local_adapter import LocalMemoryAdapter
from codie_as_a_service.core.models import ToolDefinition
from codie_as_a_service.core.protocols import (
    LLMProtocol,
    MemoryProtocol,
    PromptProtocol,
)
from codie_as_a_service.services.agent.react_orchestrator import ReActOrchestrator
from codie_as_a_service.services.memory.memory_service import MemoryService
from codie_as_a_service.services.schema_utils import json_schema_to_model
from codie_as_a_service.services.wiring import (
    build_orchestrator_phases,
    get_mcp_tool_definitions,
    get_memory_tool_definitions,
)

load_dotenv()
logging.basicConfig(
    level=os.environ.get("LOG_LEVEL", "DEBUG").upper(),
    format="%(asctime)s %(levelname)s %(name)s: %(message)s",
)
logger = logging.getLogger(__name__)


class ChatRequest(BaseModel):
    """Request body for /chat endpoint."""

    agent_id: str
    session_id: str | None = None
    message: str
    output_format: dict[str, Any] | None = None


def create_app(
    memory_service: MemoryService,
    llm_adapter: LLMProtocol,
    prompt_adapter: PromptProtocol,
    prompt_names: list[str],
    tools: list[ToolDefinition] | None = None,
    orchestrator: ReActOrchestrator | None = None,
) -> FastAPI:
    """
    Create FastAPI app with chat endpoint.

    Args:
        memory_service: Service for reading/writing agent memory
        llm_adapter: LLM adapter (handles tool execution internally)
        prompt_adapter: File-based prompt adapter
        prompt_names: List of prompt names to fetch and combine for system prompt
        tools: Optional custom tool definitions (default: memory tools only)
        orchestrator: Optional pre-built orchestrator (tests inject custom pipelines)

    Returns:
        Configured FastAPI application
    """
    app = FastAPI(title="Deep Agent Service")

    if orchestrator is None:
        if tools is None:
            tools = get_memory_tool_definitions()

        # Build orchestrator — each phase owns its execution
        phases, post_phases = build_orchestrator_phases(
            phase_names=["hydrate", "process", "format"],
            prompt_adapter=prompt_adapter,
            tools=tools,
            llm=llm_adapter,
            memory=memory_service,
        )
        orchestrator = ReActOrchestrator(
            memory=memory_service,
            phases=phases,
            post_phases=post_phases,
        )

    def generate_sse_events(
        agent_id: str,
        session_id: str,
        message: str,
        output_format: dict[str, Any] | None = None,
    ) -> Generator[str, None, None]:
        """Generate SSE events for chat response."""
        try:
            # Process through orchestrator
            output_model = (
                json_schema_to_model(output_format) if output_format else None
            )
            result = orchestrator.run(
                session_id=session_id,
                agent_id=agent_id,
                instruction=message,
                output_format=output_model,
            )

            # Emit structured response — always model_dump()
            yield f"event: response\ndata: {json.dumps(result.model_dump())}\n\n"

            # Emit done event
            yield f"event: done\ndata: {json.dumps({'usage': {'input_tokens': 0, 'output_tokens': 0}})}\n\n"

        except Exception as e:
            # Emit error event
            yield f"event: error\ndata: {json.dumps({'message': str(e)})}\n\n"

    @app.get("/health")
    async def health() -> dict[str, str]:
        """Health check endpoint."""
        return {"status": "ok"}

    @app.post("/chat")
    async def chat(request: ChatRequest) -> StreamingResponse:
        """
        Process chat message and return streaming response.

        Streams SSE events:
        - event: response, data: {...} (structured response, always)
        - event: done, data: {"usage": {...}}
        - event: error, data: {"message": "..."}
        """
        session_id = request.session_id or str(uuid.uuid4())
        return StreamingResponse(
            generate_sse_events(
                request.agent_id,
                session_id,
                request.message,
                request.output_format,
            ),
            media_type="text/event-stream",
        )

    return app


def main() -> None:
    """Start the HTTP server."""
    # Configuration from environment
    host = os.environ.get("HTTP_HOST")

    port = int(os.environ.get("HTTP_PORT"))

    llm_adapter_type = os.environ.get("LLM_ADAPTER", "claude_cli")

    prompts_dir = os.environ.get("PROMPTS_DIR")

    prompt_names = [name.strip() for name in os.environ["PROMPT_NAMES"].split(",")]

    # Initialize storage adapter
    storage_adapter: MemoryProtocol = LocalMemoryAdapter(
        base_dir=os.environ.get("STORAGE_DIR"),
        agent_path_template=os.environ.get(
            "STORAGE_PATH_TEMPLATE", "agents/{agent_id}"
        ),
    )

    # Initialize LLM adapter based on type
    if llm_adapter_type == "claude_cli":
        llm_adapter: LLMProtocol = ClaudeCliAdapter()
    else:
        model_name = os.environ.get("MODEL_NAME")
        device = os.environ.get("DEVICE", "mps")
        llm_adapter = LocalLLMAdapter(model_name=model_name, device=device)

    prompt_adapter = FilePromptAdapter(prompts_dir=prompts_dir)

    # Build memory service
    memory_service = MemoryService(storage=storage_adapter)

    # Determine tool definitions
    # When MCP is configured, cognitive-memory tools replace memory tools
    # Tool execution is handled by the adapter (Claude Code natively, local via internal loop)
    mcp_server_path = os.environ.get("MCP_SERVER_PATH")
    mcp_memory_path = os.environ.get("MCP_MEMORY_PATH")

    if mcp_server_path and mcp_memory_path:
        logger.info(
            "MCP tools enabled: %s (memory: %s)", mcp_server_path, mcp_memory_path
        )
        tools = get_mcp_tool_definitions()
    else:
        tools = get_memory_tool_definitions()

    # Create and run app
    app = create_app(
        memory_service=memory_service,
        llm_adapter=llm_adapter,
        prompt_adapter=prompt_adapter,
        prompt_names=prompt_names,
        tools=tools,
    )

    uvicorn.run(app, host=host, port=port)


if __name__ == "__main__":
    main()
