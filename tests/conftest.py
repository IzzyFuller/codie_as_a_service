"""Pytest configuration and shared fixtures for ATDD."""

import json
import os
import subprocess
import tempfile
import time
import uuid
from dataclasses import dataclass
from pathlib import Path
from unittest.mock import MagicMock, patch

import pika
import pytest
from google.cloud import firestore
from starlette.testclient import TestClient as StarletteTestClient
from synapse.adapters.rabbitmq import RabbitMQPublisher, RabbitMQSubscriber

from codie_as_a_service.adapters.llm.claude_cli_adapter import ClaudeCliAdapter
from codie_as_a_service.adapters.llm.local_llm_adapter import LocalLLMAdapter
from codie_as_a_service.adapters.messaging.models import AgentResponse, RunAgentRequest
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

# ============================================================================
# Domain-Level Test Response Specs (Adapter-Agnostic)
# ============================================================================


@dataclass
class LLMResponseSpec:
    """
    Domain-level LLM response specification for tests.

    Tests describe responses in domain terms. The TestApp converts
    these to adapter-specific formats internally.
    """

    content: str | None = None


# Test configuration
PROJECT_ID = "test-project"
RABBITMQ_PORT = 5672
FIRESTORE_EMULATOR_PORT = 8086
# Domain-level messaging constants (adapter translates to implementation)
REQUEST_SUBSCRIPTION = "agent.requests"
RESPONSE_SUBSCRIPTION = "agent.responses"
RESPONSE_TOPIC = "agent.responses"


# ============================================================================
# Google Cloud Emulator Fixtures (Real Infrastructure for Tests)
# ============================================================================


def _port_is_reachable(host: str, port: int) -> bool:
    """Check if a TCP port is accepting connections."""
    import socket

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
    # Default must be valid DefaultOutput JSON so FORMAT phase can parse it
    adapter._generate = MagicMock(
        return_value=json.dumps(
            {"response": "I'm ready to help you.", "session_id": "", "done": True}
        )
    )

    return adapter


def create_claude_cli_adapter():
    """Create ClaudeCliAdapter with mocked _run_claude method."""
    adapter = ClaudeCliAdapter()
    # Mock _run_claude - will be configured by TestApp.stub_llm_responses
    # Default must be valid DefaultOutput JSON so FORMAT phase can parse it
    adapter._run_claude = MagicMock(
        return_value=json.dumps(
            {"response": "I'm ready to help you.", "session_id": "", "done": True}
        )
    )
    return adapter


@pytest.fixture(
    scope="function",
    params=[
        pytest.param("local", id="local"),
        pytest.param("claude_cli", id="claude_cli"),
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
    rabbitmq_publisher,
    rabbitmq_subscriber,
    rabbitmq_infrastructure,
):
    """Start the agent app for pubsub E2E tests.

    Session-scoped to maintain RabbitMQ connection state.
    Uses dedicated pubsub_memory_service (not parameterized).
    """
    tools = get_memory_tool_definitions()
    phases, post_phases = build_orchestrator_phases(
        phase_names=["hydrate", "process", "format"],
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
# Test Client Fixture
# ============================================================================


class TestClient:
    """Client for E2E tests - simulates external client interacting with the system."""

    def __init__(
        self,
        connection: pika.BlockingConnection,
        publisher: RabbitMQPublisher,
        subscriber: RabbitMQSubscriber,
        request_subscription: str,
        response_topic: str,
    ):
        self._connection = connection
        self._publisher = publisher
        self._subscriber = subscriber
        self._request_subscription = request_subscription
        self._response_topic = response_topic

    def send_request(
        self,
        agent_id: str,
        session_id: str,
        message: str,
        output_format: dict | None = None,
        timeout_seconds: int = 10,
    ):
        """Publish request and wait for response on agent-specific queue."""
        # Declare agent-specific response queue (matches handler routing)
        agent_response_queue = f"{self._response_topic}.{agent_id}"
        setup_channel = self._connection.channel()
        setup_channel.queue_declare(queue=agent_response_queue, durable=True)
        setup_channel.close()

        request = RunAgentRequest(
            agent_id=agent_id,
            session_id=session_id,
            message=message,
            output_format=output_format,
        )
        # Publish to request topic
        self._publisher.publish(
            f":{self._request_subscription}", request.model_dump_json().encode("utf-8")
        ).result()

        # Wait for response on agent-specific queue
        deadline = time.time() + timeout_seconds
        while time.time() < deadline:
            pull_response = self._subscriber.pull(
                request={
                    "subscription": agent_response_queue,
                    "max_messages": 1,
                },
                timeout=1,
            )
            if pull_response.received_messages:
                msg = pull_response.received_messages[0]
                self._subscriber.acknowledge(
                    request={
                        "subscription": agent_response_queue,
                        "ack_ids": [msg.ack_id],
                    }
                )
                data = json.loads(msg.message.data.decode("utf-8"))
                return AgentResponse(**data)
            time.sleep(0.5)

        return None


@pytest.fixture(scope="session")
def test_client(rabbitmq_connection, rabbitmq_infrastructure):
    """Provide a test client with its own publisher/subscriber."""
    # Create separate connection for test client to avoid channel conflicts
    client_connection = pika.BlockingConnection(
        pika.ConnectionParameters(host="localhost", port=RABBITMQ_PORT)
    )
    return TestClient(
        connection=client_connection,
        publisher=RabbitMQPublisher(client_connection),
        subscriber=RabbitMQSubscriber(client_connection),
        request_subscription=REQUEST_SUBSCRIPTION,
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
        "orchestrator_format.txt": (
            "You are a formatting agent. Format the response into the required JSON schema."
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
    import shutil

    shutil.rmtree(temp_dir, ignore_errors=True)


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


# ============================================================================
# HTTP App Fixtures
# ============================================================================


@pytest.fixture(scope="function")
def http_app(
    memory_service,
    llm_adapter,
    file_prompt_adapter,
):
    """Create FastAPI HTTP app for E2E tests.

    Function-scoped to ensure test isolation with parameterized storage.
    """
    app = create_http_app(
        memory_service=memory_service,
        llm_adapter=llm_adapter,
        prompt_adapter=file_prompt_adapter,
        prompt_names=["codie_as_a_service_system"],
    )
    return app


class HTTPTestClient:
    """Client for HTTP E2E tests - simulates external client interacting with the system."""

    def __init__(self, app):
        self._client = StarletteTestClient(app)

    def health(self):
        """GET /health endpoint."""
        return self._client.get("/health")

    def chat(
        self,
        agent_id: str,
        session_id: str | None = None,
        message: str = "",
        output_format: dict | None = None,
    ) -> list[dict]:
        """POST to /chat and collect SSE events."""
        payload: dict = {"agent_id": agent_id, "message": message}
        if session_id is not None:
            payload["session_id"] = session_id
        if output_format:
            payload["output_format"] = output_format
        response = self._client.post("/chat", json=payload)

        # Parse SSE response
        events = []
        for line in response.text.split("\n"):
            if line.startswith("event:"):
                current_event = {"event": line[6:].strip()}
            elif line.startswith("data:"):
                data_str = line[5:].strip()
                if data_str:
                    current_event["data"] = json.loads(data_str)
                    events.append(current_event)

        return events


# ============================================================================
# TestApp - Encapsulated Test Application (Adapter-Agnostic)
# ============================================================================


class TestApp:
    """
    Encapsulates the entire test application.

    Tests interact ONLY through this class. Implementation details
    (which adapter, how it's wired) are hidden. If we change adapters,
    only this class needs to change - not the tests.
    """

    _response_counter = 0

    _PHASE_DEFAULTS = {
        "hydrate": LLMResponseSpec(
            content="Test identity summary for orchestrator testing.",
        ),
        "format": LLMResponseSpec(
            content=json.dumps(
                {"response": "default response", "session_id": "", "done": True}
            ),
        ),
    }

    def __init__(
        self,
        memory_service,
        llm_adapter,
        http_client=None,
        pubsub_client=None,
    ):
        self._memory_service = memory_service
        self._llm_adapter = llm_adapter
        self._http_client = http_client
        self._pubsub_client = pubsub_client
        self._is_claude_cli = isinstance(llm_adapter, ClaudeCliAdapter)
        self._default_response = self._to_adapter_response(
            LLMResponseSpec(content="I'm ready to help you.")
        )

    def setup_agent(
        self,
        memory: dict[str, str] | None = None,
    ) -> tuple[str, str]:
        """
        Create a test user with given memory contents.

        Args:
            memory: Optional dict of memory key -> content.
                   Defaults to minimal identity if not provided.

        Returns:
            Tuple of (agent_id, session_id)
        """
        agent_id = str(uuid.uuid4())
        session_id = str(uuid.uuid4())

        # Default minimal memory if not specified
        if memory is None:
            memory = {
                "frame": "# Frame",
                "me": "# Identity",
                "context_anchors": "# Anchors",
                "current_session": "# Session",
            }

        for key, content in memory.items():
            self._memory_service.write_memory(
                agent_id=agent_id, key=key, content=content
            )

        return agent_id, session_id

    def stub_llm_responses(self, *responses: LLMResponseSpec) -> None:
        """
        Configure LLM to return these responses in sequence.

        Thin delegate to stub_phases() for backward compatibility.
        All responses are mapped to the PROCESS phase.
        """
        self.stub_phases(process=list(responses))

    def stub_phases(
        self,
        *,
        hydrate: list[LLMResponseSpec] | None = None,
        process: list[LLMResponseSpec] | None = None,
        format_: list[LLMResponseSpec] | None = None,
        output_format: dict | None = None,
    ) -> None:
        """
        Configure LLM mock with per-phase response sequences.

        Pipeline: HYDRATE (text) + PROCESS (text) + FORMAT (structured) = 3 LLM calls.
        SYNTHESIZE is deterministic (no LLM call). HYDRATE has skip_on_retry=True.
        FORMAT sets context.done = True — single pass, no retry loop.

        Args:
            hydrate: Custom HYDRATE phase response (plain text)
            process: Custom PROCESS phase response (plain text)
            format_: Custom FORMAT phase response (JSON matching output schema)
            output_format: When set, FORMAT default uses raw content (caller's schema)
        """
        # Derive FORMAT default from process content + output_format
        last_content = ""
        if process:
            last_content = process[-1].content or ""

        if format_ is not None:
            format_response = format_[0]
        elif output_format:
            # Custom output schema — process content is already in target format
            format_response = LLMResponseSpec(content=last_content)
        else:
            # DefaultOutput schema
            format_response = LLMResponseSpec(
                content=json.dumps(
                    {"response": last_content, "session_id": "", "done": True}
                ),
            )

        process_response = LLMResponseSpec(content=last_content)

        full_sequence: list[LLMResponseSpec] = []

        # HYDRATE — plain text
        full_sequence.append(hydrate[0] if hydrate else self._PHASE_DEFAULTS["hydrate"])

        # PROCESS — plain text
        full_sequence.append(process_response)

        # FORMAT — JSON matching output schema
        full_sequence.append(format_response)

        adapter_responses = [self._to_adapter_response(r) for r in full_sequence]
        self._set_mock_side_effect(adapter_responses)

    def _get_llm_mock(self) -> MagicMock:
        """Return the adapter-specific mock (ClaudeCliAdapter or LocalLLMAdapter)."""
        if self._is_claude_cli:
            return self._llm_adapter._run_claude
        return self._llm_adapter._generate

    def _set_mock_side_effect(self, responses: list[str]) -> None:
        """Set side_effect on the adapter-specific LLM mock."""
        self._get_llm_mock().side_effect = responses

    def reset_llm(self) -> None:
        """Reset LLM mock to default state."""
        mock = self._get_llm_mock()
        mock.side_effect = None
        mock.return_value = self._default_response

    def chat(
        self,
        agent_id: str,
        session_id: str | None = None,
        message: str = "",
        output_format: dict | None = None,
    ) -> list[dict]:
        """
        Make a chat request and return SSE events.

        Args:
            agent_id: Agent identifier
            session_id: Session identifier (generated by backend if omitted)
            message: User message
            output_format: Optional JSON schema for structured output

        Returns:
            List of SSE events as dicts with 'event' and 'data' keys
        """
        return self._http_client.chat(agent_id, session_id, message, output_format)

    def health(self):
        """GET /health endpoint."""
        return self._http_client.health()

    def read_memory(self, agent_id: str, key: str) -> str | None:
        """Read agent memory (for verifying side effects)."""
        return self._memory_service.read_memory(agent_id=agent_id, key=key)

    def write_memory(self, agent_id: str, key: str, content: str) -> None:
        """Write agent memory (for test setup)."""
        self._memory_service.write_memory(agent_id=agent_id, key=key, content=content)

    def send_pubsub_request(
        self,
        agent_id: str,
        session_id: str,
        message: str,
        output_format: dict | None = None,
        timeout_seconds: int = 10,
    ) -> AgentResponse | None:
        """
        Send a Pub/Sub request and wait for response.

        Args:
            agent_id: Agent identifier
            session_id: Session identifier
            message: User message
            output_format: Optional JSON schema for structured output
            timeout_seconds: How long to wait for response

        Returns:
            AgentResponse or None if timeout
        """
        if self._pubsub_client is None:
            raise RuntimeError("TestApp not configured for Pub/Sub tests")
        return self._pubsub_client.send_request(
            agent_id, session_id, message, output_format, timeout_seconds
        )

    def _to_adapter_response(self, spec: LLMResponseSpec) -> str:
        """
        Convert domain-level spec to text that the adapter will parse.

        Both adapters return plain text from their mocked internal methods
        (_generate for local, _run_claude for CLI). The call() method
        handles parsing/validation.
        """
        return spec.content or ""


@pytest.fixture(scope="function")
def test_app(memory_service, llm_adapter, http_app) -> TestApp:
    """
    Provide the encapsulated test application for HTTP tests.

    Function-scoped to ensure test isolation with parameterized storage.
    This is the PRIMARY fixture HTTP tests should use. It hides all
    implementation details about adapters and wiring.
    """
    http_client = HTTPTestClient(http_app)
    return TestApp(
        memory_service=memory_service,
        llm_adapter=llm_adapter,
        http_client=http_client,
    )


@pytest.fixture(scope="session")
def pubsub_test_app(
    pubsub_memory_service, pubsub_llm_adapter, agent_app, test_client
) -> TestApp:
    """
    Provide the encapsulated test application for Pub/Sub tests.

    Session-scoped to maintain RabbitMQ connection state.
    Uses dedicated pubsub fixtures (not parameterized).
    This is the PRIMARY fixture Pub/Sub tests should use.
    """
    return TestApp(
        memory_service=pubsub_memory_service,
        llm_adapter=pubsub_llm_adapter,
        pubsub_client=test_client,
    )


@pytest.fixture(scope="function")
def http_test_client(http_app):
    """Provide an HTTP test client for E2E tests."""
    return HTTPTestClient(http_app)
