"""HTTP entry point for deep agent service."""

import json
import logging
import os
from typing import Any, Generator

import uvicorn
from dotenv import load_dotenv
from fastapi import Depends, FastAPI, Header, HTTPException
from fastapi.responses import StreamingResponse
from google.cloud import storage
from pydantic import BaseModel

from codie_as_a_service.adapters.auth.api_key_adapter import APIKeyAuthAdapter
from codie_as_a_service.adapters.llm.claude_cli_adapter import ClaudeCliAdapter
from codie_as_a_service.adapters.llm.local_llm_adapter import LocalLLMAdapter
from codie_as_a_service.adapters.prompts.file_adapter import FilePromptAdapter
from codie_as_a_service.adapters.storage.gcs_adapter import GCSMemoryAdapter
from codie_as_a_service.adapters.storage.local_adapter import LocalMemoryAdapter
from codie_as_a_service.core.protocols import (
    AuthProtocol,
    LLMProtocol,
    MemoryProtocol,
    PromptProtocol,
)
from codie_as_a_service.core.models import ToolDefinition
from codie_as_a_service.core.phase_models import (
    ExtendedInstruction,
    HydratedIdentity,
    PhaseDefinition,
    ProcessResult,
    SynthesisResult,
    ValidationResult,
)
from codie_as_a_service.services.agent.react_agent import ReActAgent
from codie_as_a_service.services.agent.react_orchestrator import ReActOrchestrator
from codie_as_a_service.services.memory.memory_service import MemoryService
from codie_as_a_service.services.tools.memory_tool_executor import MemoryToolExecutor

load_dotenv()
logging.basicConfig(
    level=os.environ.get("LOG_LEVEL", "INFO").upper(),
    format="%(asctime)s %(levelname)s %(name)s: %(message)s",
)
logger = logging.getLogger(__name__)


class ChatRequest(BaseModel):
    """Request body for /chat endpoint."""

    user_id: str
    session_id: str
    message: str
    output_format: dict[str, Any] | None = None


def create_app(
    memory_service: MemoryService,
    llm_adapter: LLMProtocol,
    prompt_adapter: PromptProtocol,
    prompt_names: list[str],
    auth: AuthProtocol,
) -> FastAPI:
    """
    Create FastAPI app with chat endpoint.

    Args:
        memory_service: Service for reading/writing user memory
        llm_adapter: OpenAI-compatible LLM adapter
        prompt_adapter: File-based prompt adapter
        prompt_names: List of prompt names to fetch and combine for system prompt
        auth: Authentication adapter for verifying requests

    Returns:
        Configured FastAPI application
    """
    app = FastAPI(title="Deep Agent Service")

    # Build tool executor and tool definitions
    tool_executor = MemoryToolExecutor(memory=memory_service)
    tools = _get_memory_tool_definitions()

    # Initialize agent with adapters
    agent = ReActAgent(
        llm=llm_adapter,
        prompts=prompt_adapter,
        memory=memory_service,
        prompt_names=prompt_names,
        tool_executor=tool_executor,
        tools=tools,
    )

    # Build orchestrator phases and orchestrator
    phases = _build_orchestrator_phases(prompt_adapter, tools)
    format_phase = PhaseDefinition(
        name="format",
        system_prompt=(
            "You are a JSON formatter. Return ONLY valid JSON, no other text. "
            'Example: {"response": "Hello!"}'
        ),
        output_schema=ProcessResult,  # Not used for FORMAT
    )
    orchestrator = ReActOrchestrator(
        react_agent=agent,
        llm=llm_adapter,
        memory=memory_service,
        phases=phases,
        format_phase=format_phase,
    )

    def verify_api_key(x_api_key: str | None = Header(None)) -> None:
        """FastAPI dependency to verify API key."""
        if x_api_key is None or not auth.verify(x_api_key):
            raise HTTPException(status_code=401, detail="Invalid or missing API key")

    def generate_sse_events(
        user_id: str,
        session_id: str,
        message: str,
        output_format: dict[str, Any] | None = None,
    ) -> Generator[str, None, None]:
        """Generate SSE events for chat response."""
        try:
            # Process through orchestrator (always returns dict)
            response = orchestrator.run(
                user_id=user_id,
                instruction=message,
                tool_executor=tool_executor,
                output_format=output_format,
            )

            # Emit structured response
            yield f"event: response\ndata: {json.dumps(response)}\n\n"

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
                request.user_id,
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
            description="Read user memory by key (e.g., 'current_session', 'context_anchors')",
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
            description="Write content to user memory",
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
            description="List all memory keys for the user",
            input_schema={
                "type": "object",
                "properties": {},
            },
        ),
    ]


def _build_orchestrator_phases(
    prompt_adapter: PromptProtocol, tools: list[ToolDefinition]
) -> list[PhaseDefinition]:
    """Build the standard orchestrator phase definitions."""
    return [
        PhaseDefinition(
            name="hydrate",
            system_prompt=prompt_adapter.get_prompt("orchestrator_hydrate"),
            tools=[],
            output_schema=HydratedIdentity,
            max_iterations=1,
        ),
        PhaseDefinition(
            name="extend",
            system_prompt=prompt_adapter.get_prompt("orchestrator_extend"),
            tools=tools,
            output_schema=ExtendedInstruction,
            max_iterations=5,
        ),
        PhaseDefinition(
            name="process",
            system_prompt=prompt_adapter.get_prompt("orchestrator_process"),
            tools=tools,
            output_schema=ProcessResult,
            max_iterations=10,
        ),
        PhaseDefinition(
            name="validate",
            system_prompt=prompt_adapter.get_prompt("orchestrator_validate"),
            tools=[],
            output_schema=ValidationResult,
            max_iterations=1,
            completes_request=True,
        ),
        PhaseDefinition(
            name="synthesize",
            system_prompt=prompt_adapter.get_prompt("orchestrator_synthesize"),
            tools=tools,
            output_schema=SynthesisResult,
            max_iterations=5,
        ),
    ]


def main() -> None:
    """Start the HTTP server."""
    # Configuration from environment
    storage_adapter_type = os.environ.get("STORAGE_ADAPTER", "gcs")

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

    # Initialize storage adapter based on type
    if storage_adapter_type == "gcs":
        gcs_bucket_name = os.environ.get("GCS_BUCKET_NAME")
        if not gcs_bucket_name:
            raise ValueError("GCS_BUCKET_NAME required when STORAGE_ADAPTER=gcs")
        gcs_client = storage.Client()
        bucket = gcs_client.bucket(gcs_bucket_name)
        storage_adapter: MemoryProtocol = GCSMemoryAdapter(bucket=bucket)
    elif storage_adapter_type == "local":
        storage_dir = os.environ.get("STORAGE_DIR")
        if not storage_dir:
            raise ValueError("STORAGE_DIR required when STORAGE_ADAPTER=local")
        storage_adapter = LocalMemoryAdapter(base_dir=storage_dir)
    else:
        raise ValueError(f"Unknown STORAGE_ADAPTER: {storage_adapter_type}")

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

    # Create and run app
    app = create_app(
        memory_service=memory_service,
        llm_adapter=llm_adapter,
        prompt_adapter=prompt_adapter,
        prompt_names=prompt_names,
        auth=auth_adapter,
    )

    uvicorn.run(app, host=host, port=port)


if __name__ == "__main__":
    main()
