"""HTTP entry point for deep agent service."""

import json
import os
from typing import Any, Generator

import uvicorn
from fastapi import FastAPI
from fastapi.responses import StreamingResponse
from google.cloud import storage
from pydantic import BaseModel

from deep_agent_service.adapters.llm.openai_adapter import OpenAILLMAdapter
from deep_agent_service.adapters.prompts.file_adapter import FilePromptAdapter
from deep_agent_service.adapters.storage.gcs_adapter import GCSMemoryAdapter
from deep_agent_service.services.agent.react_agent import ReActAgent
from deep_agent_service.services.memory.memory_service import MemoryService


class ChatRequest(BaseModel):
    """Request body for /chat endpoint."""

    user_id: str
    session_id: str
    message: str
    output_format: dict[str, Any] | None = None


def create_app(
    memory_service: MemoryService,
    llm_adapter: OpenAILLMAdapter,
    prompt_adapter: FilePromptAdapter,
    prompt_names: list[str],
) -> FastAPI:
    """
    Create FastAPI app with chat endpoint.

    Args:
        memory_service: Service for reading/writing user memory
        llm_adapter: OpenAI-compatible LLM adapter
        prompt_adapter: File-based prompt adapter
        prompt_names: List of prompt names to fetch and combine for system prompt

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

    @app.post("/chat")
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

    local_llm_url = os.environ.get("LOCAL_LLM_URL")
    if not local_llm_url:
        raise ValueError("LOCAL_LLM_URL environment variable is required")

    local_model_name = os.environ.get("LOCAL_MODEL_NAME")
    if not local_model_name:
        raise ValueError("LOCAL_MODEL_NAME environment variable is required")

    prompts_dir = os.environ.get("PROMPTS_DIR")
    if not prompts_dir:
        raise ValueError("PROMPTS_DIR environment variable is required")

    prompt_names_str = os.environ.get("PROMPT_NAMES")
    if not prompt_names_str:
        raise ValueError("PROMPT_NAMES environment variable is required")
    prompt_names = [name.strip() for name in prompt_names_str.split(",")]

    # Initialize clients
    gcs_client = storage.Client()
    bucket = gcs_client.bucket(gcs_bucket_name)

    # Initialize adapters
    llm_adapter = OpenAILLMAdapter(base_url=local_llm_url, model=local_model_name)
    prompt_adapter = FilePromptAdapter(prompts_dir=prompts_dir)

    # Build memory service
    memory_service = MemoryService(storage=GCSMemoryAdapter(bucket=bucket))

    # Create and run app
    app = create_app(
        memory_service=memory_service,
        llm_adapter=llm_adapter,
        prompt_adapter=prompt_adapter,
        prompt_names=prompt_names,
    )

    uvicorn.run(app, host=host, port=port)


if __name__ == "__main__":
    main()
