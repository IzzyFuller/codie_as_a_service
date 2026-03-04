"""Lucea Health — Triage Demo entry point.

Wires together the CaaS pipeline, NiceGUI app, and transcript playback.
Run with: python -m codie_as_a_service.demo.lucea
"""

from __future__ import annotations

import logging
import os
from pathlib import Path

from dotenv import load_dotenv
from nicegui import ui

from codie_as_a_service.adapters.llm.anthropic_adapter import AnthropicAPIAdapter
from codie_as_a_service.adapters.llm.claude_cli_adapter import ClaudeCliAdapter
from codie_as_a_service.adapters.prompts.file_adapter import FilePromptAdapter
from codie_as_a_service.adapters.storage.local_adapter import LocalMemoryAdapter
from codie_as_a_service.api.client import CaaSClient
from codie_as_a_service.core.protocols import LLMProtocol
from codie_as_a_service.demo.lucea.app import create_lucea_app
from codie_as_a_service.main_http import create_app
from codie_as_a_service.services.memory.memory_service import MemoryService

load_dotenv()
logging.basicConfig(
    level=os.environ.get("LOG_LEVEL", "INFO").upper(),
    format="%(asctime)s %(levelname)s %(name)s: %(message)s",
)
logger = logging.getLogger(__name__)

# Paths
_PROJECT_ROOT = Path(__file__).resolve().parents[4]  # up from demo/lucea/ to repo root
_DATA_DIR = _PROJECT_ROOT / "data"
_PROMPTS_DIR = _PROJECT_ROOT / "prompts"


def _build_llm_adapter() -> LLMProtocol:
    """Select LLM adapter based on environment configuration."""
    api_key = os.environ.get("ANTHROPIC_API_KEY")
    if api_key:
        model = os.environ.get("ANTHROPIC_MODEL", "claude-sonnet-4-20250514")
        logger.info("Using AnthropicAPIAdapter (model=%s)", model)
        return AnthropicAPIAdapter(api_key=api_key, model=model)

    logger.info("Using ClaudeCliAdapter")
    return ClaudeCliAdapter()


def _build_caas_client() -> CaaSClient:
    """Build a CaaSClient backed by an in-process FastAPI app."""
    storage = LocalMemoryAdapter(
        base_dir=str(_DATA_DIR),
        agent_path_template="agents/{agent_id}",
    )
    memory_service = MemoryService(storage=storage)
    prompt_adapter = FilePromptAdapter(prompts_dir=str(_PROMPTS_DIR))
    llm = _build_llm_adapter()

    # create_app builds the orchestrator internally
    app = create_app(
        memory_service=memory_service,
        llm_adapter=llm,
        prompt_adapter=prompt_adapter,
        prompt_names=["codie_as_a_service_system"],
    )

    return CaaSClient(app=app)


def main() -> None:
    """Launch the Lucea triage demo."""
    client = _build_caas_client()
    logger.info("CaaS client ready — live chat enabled")

    @ui.page("/")
    def index():
        create_lucea_app(caas_client=client)

    ui.run(
        title="Lucea Health — Triage Assistant",
        port=int(os.environ.get("LUCEA_PORT", "8090")),
        reload=False,
    )


if __name__ == "__main__":
    main()
