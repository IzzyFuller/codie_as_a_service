"""HTTP entry point for deep agent service."""

import json
import os
from typing import Any, Generator

import uvicorn
from fastapi import Depends, FastAPI, Header, HTTPException
from fastapi.responses import StreamingResponse
from google.cloud import storage
from pydantic import BaseModel

from codie_as_a_service.adapters.auth.api_key_adapter import APIKeyAuthAdapter
from codie_as_a_service.adapters.llm.claude_cli_adapter import ClaudeCliAdapter
from codie_as_a_service.adapters.llm.local_llm_adapter import LocalLLMAdapter
from codie_as_a_service.adapters.prompts.file_adapter import FilePromptAdapter
from codie_as_a_service.adapters.storage.gcs_adapter import GCSMemoryAdapter
from codie_as_a_service.core.protocols import AuthProtocol, LLMProtocol, PromptProtocol
from codie_as_a_service.services.agent.react_agent import ReActAgent
from codie_as_a_service.services.memory.memory_service import MemoryService


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

    # Initialize agent with adapters
    agent = ReActAgent(
        llm=llm_adapter,
        prompts=prompt_adapter,
        memory=memory_service,
        prompt_names=prompt_names,
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
            # Process through agent (always returns dict)
            response = agent.process(
                user_id=user_id, message=message, output_format=output_format
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


def main() -> None:
    """Start the HTTP server."""
    # Configuration from environment
    gcs_bucket_name = os.environ.get("GCS_BUCKET_NAME")
    if not gcs_bucket_name:
        raise ValueError("GCS_BUCKET_NAME environment variable is required")

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

    # Initialize clients
    gcs_client = storage.Client()
    bucket = gcs_client.bucket(gcs_bucket_name)

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
    memory_service = MemoryService(storage=GCSMemoryAdapter(bucket=bucket))

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
