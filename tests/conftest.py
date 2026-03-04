"""Pytest configuration and shared fixtures for ATDD."""

import os
import shutil
import socket
import subprocess
import tempfile
import time
import uuid
from pathlib import Path
from unittest.mock import MagicMock, patch

import pika
import pytest
from google.cloud import firestore
from codie_as_a_service.api.client import CaaSClient
from synapse.adapters.rabbitmq import RabbitMQPublisher, RabbitMQSubscriber

from codie_as_a_service.adapters.llm.anthropic_adapter import AnthropicAPIAdapter
from codie_as_a_service.adapters.llm.claude_cli_adapter import ClaudeCliAdapter
from codie_as_a_service.adapters.llm.local_llm_adapter import LocalLLMAdapter
from codie_as_a_service.adapters.prompts.file_adapter import FilePromptAdapter
from codie_as_a_service.adapters.storage.local_adapter import LocalMemoryAdapter
from codie_as_a_service.main_http import create_app as create_http_app
from codie_as_a_service.main_pubsub import create_app
from codie_as_a_service.services.agent.react_orchestrator import ReActOrchestrator
from codie_as_a_service.services.memory.memory_service import MemoryService
from codie_as_a_service.services.wiring import (
    build_orchestrator_phases,
    get_memory_tool_definitions,
)

# Test configuration
PROJECT_ID = "test-project"
RABBITMQ_PORT = 5672
FIRESTORE_EMULATOR_PORT = 8086
# Domain-level messaging constants (adapter translates to implementation)
REQUEST_SUBSCRIPTION = "agent.requests"
RESPONSE_SUBSCRIPTION = "agent.responses"
RESPONSE_TOPIC = "agent.responses"

# Default identity memory — minimum viable agent
DEFAULT_IDENTITY = {
    "frame": "# Frame",
    "me": "# Identity",
    "context_anchors": "# Anchors",
    "current_session": "# Session",
}


# ============================================================================
# Helpers (not fixtures — plain functions tests can call)
# ============================================================================


def setup_agent_memory(
    memory_service: MemoryService,
    memory: dict[str, str] | None = None,
) -> tuple[str, str]:
    """Create a test agent with identity files in memory.

    Args:
        memory_service: Real MemoryService with LocalMemoryAdapter
        memory: Optional dict of memory key -> content.
               Defaults to minimal identity if not provided.

    Returns:
        Tuple of (agent_id, session_id)
    """
    agent_id = str(uuid.uuid4())
    session_id = str(uuid.uuid4())

    for key, content in (memory or DEFAULT_IDENTITY).items():
        memory_service.write_memory(agent_id=agent_id, key=key, content=content)

    return agent_id, session_id


def get_llm_mock(llm_adapter) -> MagicMock:
    """Return the mocked boundary method for any LLM adapter.

    All three boundary methods return plain text, so tests can
    uniformly set ``mock.return_value = "text"``.
    """
    if isinstance(llm_adapter, ClaudeCliAdapter):
        return llm_adapter._run_claude
    if isinstance(llm_adapter, AnthropicAPIAdapter):
        return llm_adapter._create_message
    return llm_adapter._generate


# ============================================================================
# Google Cloud Emulator Fixtures (Real Infrastructure for Tests)
# ============================================================================


def _port_is_reachable(host: str, port: int) -> bool:
    """Check if a TCP port is accepting connections."""
    try:
        with socket.create_connection((host, port), timeout=2):
            return True
    except OSError:
        return False


@pytest.fixture(scope="session")
def rabbitmq_broker():
    """Start RabbitMQ Docker container, or reuse existing service on port."""
    started_container = False

    if not _port_is_reachable("localhost", RABBITMQ_PORT):
        subprocess.run(["docker", "rm", "-f", "rabbitmq-test"], capture_output=True)
        subprocess.run(
            [
                "docker",
                "run",
                "-d",
                "--name",
                "rabbitmq-test",
                "-p",
                f"{RABBITMQ_PORT}:5672",
                "rabbitmq:3-management",
            ],
            check=True,
            capture_output=True,
        )
        time.sleep(10)
        started_container = True

    yield

    if started_container:
        subprocess.run(["docker", "stop", "rabbitmq-test"], capture_output=True)
        subprocess.run(["docker", "rm", "rabbitmq-test"], capture_output=True)


@pytest.fixture(scope="session")
def firestore_emulator():
    """Start Firestore emulator, or reuse existing service on port."""
    started_container = False

    if not _port_is_reachable("localhost", FIRESTORE_EMULATOR_PORT):
        subprocess.run(
            ["docker", "rm", "-f", "firestore-emulator-test"], capture_output=True
        )
        subprocess.run(
            [
                "docker",
                "run",
                "-d",
                "--name",
                "firestore-emulator-test",
                "-p",
                f"{FIRESTORE_EMULATOR_PORT}:8080",
                "google/cloud-sdk:emulators",
                "/bin/bash",
                "-c",
                f"gcloud beta emulators firestore start --project={PROJECT_ID} --host-port=0.0.0.0:8080",
            ],
            check=True,
            capture_output=True,
        )
        time.sleep(5)
        started_container = True

    os.environ["FIRESTORE_EMULATOR_HOST"] = f"localhost:{FIRESTORE_EMULATOR_PORT}"

    yield

    if started_container:
        subprocess.run(
            ["docker", "stop", "firestore-emulator-test"], capture_output=True
        )
        subprocess.run(["docker", "rm", "firestore-emulator-test"], capture_output=True)
    os.environ.pop("FIRESTORE_EMULATOR_HOST", None)


@pytest.fixture(scope="session")
def rabbitmq_connection(rabbitmq_broker) -> pika.BlockingConnection:
    """Provide RabbitMQ connection."""
    return pika.BlockingConnection(
        pika.ConnectionParameters(host="localhost", port=RABBITMQ_PORT)
    )


@pytest.fixture(scope="session")
def rabbitmq_publisher(rabbitmq_connection) -> RabbitMQPublisher:
    """Provide RabbitMQ publisher."""
    return RabbitMQPublisher(rabbitmq_connection)


@pytest.fixture(scope="session")
def rabbitmq_subscriber(rabbitmq_connection) -> RabbitMQSubscriber:
    """Provide RabbitMQ subscriber."""
    return RabbitMQSubscriber(rabbitmq_connection)


@pytest.fixture(scope="session")
def firestore_client(firestore_emulator) -> firestore.Client:
    """Provide Firestore client connected to emulator."""
    return firestore.Client(project=PROJECT_ID)


@pytest.fixture(scope="session")
def rabbitmq_infrastructure(rabbitmq_connection):
    """Create messaging infrastructure for tests."""
    channel = rabbitmq_connection.channel()

    # Declare request subscription endpoint
    channel.queue_declare(queue=REQUEST_SUBSCRIPTION, durable=True)

    # Declare response topic endpoint
    channel.exchange_declare(
        exchange=RESPONSE_TOPIC, exchange_type="fanout", durable=True
    )
    channel.queue_declare(queue=RESPONSE_SUBSCRIPTION, durable=True)
    channel.queue_bind(queue=RESPONSE_SUBSCRIPTION, exchange=RESPONSE_TOPIC)

    # Purge stale messages from previous test runs
    channel.queue_purge(queue=REQUEST_SUBSCRIPTION)
    channel.queue_purge(queue=RESPONSE_SUBSCRIPTION)


# ============================================================================
# Storage Adapter Fixtures
# ============================================================================


@pytest.fixture(scope="function")
def storage_adapter(tmp_path):
    """Provide LocalMemoryAdapter for test isolation.

    Function-scoped to ensure each test gets a fresh adapter.
    """
    return LocalMemoryAdapter(base_dir=str(tmp_path))


@pytest.fixture(scope="function")
def memory_service(storage_adapter):
    """Create memory service with local storage adapter.

    Function-scoped - each test gets fresh MemoryService instance.
    """
    return MemoryService(storage=storage_adapter)


@pytest.fixture(scope="session")
def pubsub_memory_service():
    """Session-scoped memory service for pubsub tests.

    Pubsub tests exercise the messaging layer, not storage adapters.
    """
    temp_dir = tempfile.mkdtemp(prefix="pubsub_test_memory_")
    return MemoryService(storage=LocalMemoryAdapter(base_dir=temp_dir))


# ============================================================================
# LLM Adapter Parameterization
# ============================================================================


def create_local_llm_adapter():
    """Create LocalLLMAdapter with mocked MLX model loading and generation."""
    mock_tokenizer = MagicMock()
    mock_model = MagicMock()

    with (
        patch("codie_as_a_service.adapters.llm.local_llm_adapter.load") as mock_load,
        patch("codie_as_a_service.adapters.llm.local_llm_adapter.from_mlxlm"),
    ):
        mock_load.return_value = (mock_model, mock_tokenizer)
        adapter = LocalLLMAdapter(model_name="test-model")

    # Mock _generate directly - bypasses mlx_lm.generate
    adapter._generate = MagicMock(return_value="I'm ready to help you.")

    return adapter


def create_claude_cli_adapter():
    """Create ClaudeCliAdapter with mocked _run_claude method."""
    adapter = ClaudeCliAdapter()
    # Mock _run_claude - will be configured per-test via get_llm_mock()
    adapter._run_claude = MagicMock(return_value="I'm ready to help you.")
    return adapter


def create_anthropic_adapter():
    """Create AnthropicAPIAdapter with mocked _create_message method."""
    with patch("codie_as_a_service.adapters.llm.anthropic_adapter.anthropic"):
        adapter = AnthropicAPIAdapter(api_key="test-key")

    adapter._create_message = MagicMock(return_value="I'm ready to help you.")
    return adapter


@pytest.fixture(
    scope="function",
    params=[
        pytest.param("local", id="local"),
        pytest.param("claude_cli", id="claude_cli"),
        pytest.param("anthropic", id="anthropic"),
    ],
)
def llm_adapter_type(request):
    """Parameterized LLM adapter type."""
    return request.param


@pytest.fixture(scope="function")
def llm_adapter(llm_adapter_type):
    """Create LLM adapter based on parameterized type.

    Function-scoped to ensure mock state isolation between tests.
    """
    if llm_adapter_type == "local":
        return create_local_llm_adapter()
    elif llm_adapter_type == "claude_cli":
        return create_claude_cli_adapter()
    elif llm_adapter_type == "anthropic":
        return create_anthropic_adapter()
    else:
        raise ValueError(f"Unknown LLM adapter type: {llm_adapter_type}")


@pytest.fixture(scope="session")
def pubsub_llm_adapter():
    """Session-scoped LLM adapter for pubsub tests.

    Separate from function-scoped llm_adapter to maintain session state.
    Uses LocalLLMAdapter - pubsub tests exercise messaging, not LLM adapters.
    """
    return create_local_llm_adapter()


@pytest.fixture(scope="session")
def agent_app(
    pubsub_memory_service,
    pubsub_llm_adapter,
    file_prompt_adapter,
    rabbitmq_connection,
    rabbitmq_publisher,
    rabbitmq_subscriber,
    rabbitmq_infrastructure,
):
    """Start the agent app for pubsub E2E tests.

    Session-scoped to maintain RabbitMQ connection state.
    Uses dedicated pubsub_memory_service (not parameterized).
    Purges request queue to avoid stale messages from previous test sessions.
    """
    # Purge stale requests from previous sessions (rabbitmq_infrastructure
    # purges during setup, but stale messages can accumulate between sessions)
    purge_channel = rabbitmq_connection.channel()
    purge_channel.queue_purge(queue=REQUEST_SUBSCRIPTION)
    purge_channel.close()

    tools = get_memory_tool_definitions()
    phases, post_phases = build_orchestrator_phases(
        phase_names=["hydrate", "process"],
        prompt_adapter=file_prompt_adapter,
        tools=tools,
        llm=pubsub_llm_adapter,
        memory=pubsub_memory_service,
    )
    orchestrator = ReActOrchestrator(
        memory=pubsub_memory_service,
        phases=phases,
        post_phases=post_phases,
    )

    # Adapters implement synapse protocols
    app = create_app(
        memory_service=pubsub_memory_service,
        llm_adapter=pubsub_llm_adapter,
        prompt_adapter=file_prompt_adapter,
        prompt_names=["codie_as_a_service_system"],
        publisher=rabbitmq_publisher,
        subscriber=rabbitmq_subscriber,
        request_subscription_path=REQUEST_SUBSCRIPTION,
        response_topic_path=RESPONSE_TOPIC,
        orchestrator=orchestrator,
    )
    app.start()
    yield app
    app.stop()


# ============================================================================
# Pubsub Test Client
# ============================================================================


@pytest.fixture(scope="session")
def caas_pubsub_client(rabbitmq_connection, rabbitmq_infrastructure):
    """Provide a CaaSClient wired to RabbitMQ for pubsub E2E tests."""
    client_connection = pika.BlockingConnection(
        pika.ConnectionParameters(host="localhost", port=RABBITMQ_PORT)
    )
    return CaaSClient(
        connection=client_connection,
        request_topic=REQUEST_SUBSCRIPTION,
        response_topic=RESPONSE_TOPIC,
    )


@pytest.fixture(scope="session")
def file_prompt_adapter():
    """Create FilePromptAdapter with test prompts in temporary directory."""
    # Create temporary prompts directory
    temp_dir = tempfile.mkdtemp(prefix="test_prompts_")
    prompts_path = Path(temp_dir)

    test_prompts = {
        "codie_as_a_service_system.txt": (
            "You are a helpful AI assistant with access to agent memory. "
            "Identity: {me}. Context: {context_anchors}. Session: {current_session}. "
            "You can read and write to the user's memory using the provided tools."
        ),
        "orchestrator_hydrate.txt": (
            "You are an identity hydration agent. Summarize the identity context."
        ),
        "orchestrator_process.txt": (
            "You are a processing agent. Execute the instruction using available tools. "
            "Return your response as plain text."
        ),
    }

    # Write test prompt files
    for filename, content in test_prompts.items():
        prompt_file = prompts_path / filename
        prompt_file.write_text(content, encoding="utf-8")

    # Create and return FilePromptAdapter
    adapter = FilePromptAdapter(prompts_dir=prompts_path)
    yield adapter

    # Cleanup: remove temp directory after tests
    shutil.rmtree(temp_dir, ignore_errors=True)


# ============================================================================
# Orchestrator & HTTP Fixtures
# ============================================================================


@pytest.fixture(scope="function")
def orchestrator(memory_service, llm_adapter, file_prompt_adapter):
    """Build a process-only orchestrator for HTTP E2E tests.

    Uses only PROCESS phase (1 LLM call) + SYNTHESIZE post-phase.
    Most tests only need to verify the HTTP flow, response shape,
    and persistence — not multi-phase pipeline behavior.
    """
    tools = get_memory_tool_definitions()
    phases, post_phases = build_orchestrator_phases(
        phase_names=["process"],
        prompt_adapter=file_prompt_adapter,
        tools=tools,
        llm=llm_adapter,
        memory=memory_service,
    )
    return ReActOrchestrator(
        memory=memory_service,
        phases=phases,
        post_phases=post_phases,
    )


@pytest.fixture(scope="function")
def http_app(
    orchestrator,
    memory_service,
    llm_adapter,
    file_prompt_adapter,
):
    """Create FastAPI HTTP app with injected orchestrator.

    Function-scoped to ensure test isolation with parameterized storage.
    """
    return create_http_app(
        memory_service=memory_service,
        llm_adapter=llm_adapter,
        prompt_adapter=file_prompt_adapter,
        prompt_names=["codie_as_a_service_system"],
        orchestrator=orchestrator,
    )


@pytest.fixture(scope="function")
def caas_client(http_app):
    """Provide a CaaSClient wired to the test FastAPI app for E2E tests."""
    return CaaSClient(app=http_app)


# ============================================================================
# Test Data Fixtures
# ============================================================================


@pytest.fixture
def test_agent_id() -> str:
    """Provide consistent test user ID."""
    return "test_user_123"


@pytest.fixture
def test_session_id() -> str:
    """Provide consistent test session ID."""
    return "test_session_456"


@pytest.fixture
def sample_memory_content() -> str:
    """Provide sample memory content for testing."""
    return """# Current Session Memory

## Tasks Completed
- Initial setup
- Memory structure created

## Current Focus
- Testing memory isolation
"""
