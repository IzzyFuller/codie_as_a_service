"""Pytest configuration and shared fixtures for ATDD."""

import json
import os
import subprocess
import tempfile
import time
import uuid
from dataclasses import dataclass, field
from pathlib import Path
from unittest.mock import MagicMock, patch

import pika
import pytest
import torch
from google.cloud import firestore, storage

from starlette.testclient import TestClient as StarletteTestClient

from codie_as_a_service.adapters.messaging.rabbitmq_adapter import (
    RabbitMQPublisher,
    RabbitMQSubscriber,
)
from codie_as_a_service.adapters.prompts.file_adapter import FilePromptAdapter
from codie_as_a_service.core.models import RunAgentRequest, AgentResponse
from codie_as_a_service.services.memory.memory_service import MemoryService
from codie_as_a_service.adapters.storage.gcs_adapter import GCSMemoryAdapter
from codie_as_a_service.main_pubsub import create_app
from codie_as_a_service.main_http import create_app as create_http_app

# ============================================================================
# Domain-Level Test Response Specs (Adapter-Agnostic)
# ============================================================================


@dataclass
class ToolCallSpec:
    """Domain-level tool call specification for tests."""

    name: str
    arguments: dict = field(default_factory=dict)
    id: str = field(default_factory=lambda: f"tool_{uuid.uuid4().hex[:8]}")


@dataclass
class LLMResponseSpec:
    """
    Domain-level LLM response specification for tests.

    Tests describe responses in domain terms. The TestApp converts
    these to adapter-specific formats internally.
    """

    stop_reason: str  # "end_turn" | "tool_use"
    content: str | None = None
    tool_calls: list[ToolCallSpec] = field(default_factory=list)


# Test configuration
PROJECT_ID = "test-project"
GCS_BUCKET_NAME = "test-deep-agent-memory"
RABBITMQ_PORT = 5672
FIRESTORE_EMULATOR_PORT = 8086
GCS_EMULATOR_PORT = 4443
# Domain-level messaging constants (adapter translates to implementation)
REQUEST_SUBSCRIPTION = "agent.requests"
RESPONSE_SUBSCRIPTION = "agent.responses"
RESPONSE_TOPIC = "agent.responses"
# Test API key for auth
TEST_API_KEY = "test-secret-key-12345"


# ============================================================================
# Google Cloud Emulator Fixtures (Real Infrastructure for Tests)
# ============================================================================


@pytest.fixture(scope="session")
def rabbitmq_broker():
    """Start RabbitMQ Docker container for test session."""
    # Clean up any existing container
    subprocess.run(["docker", "rm", "-f", "rabbitmq-test"], capture_output=True)

    # Start RabbitMQ
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

    time.sleep(10)  # Wait for RabbitMQ to be ready

    yield

    # Cleanup
    subprocess.run(["docker", "stop", "rabbitmq-test"], capture_output=True)
    subprocess.run(["docker", "rm", "rabbitmq-test"], capture_output=True)


@pytest.fixture(scope="session")
def firestore_emulator():
    """Start Firestore emulator Docker container for test session."""
    # Clean up any existing container
    subprocess.run(
        ["docker", "rm", "-f", "firestore-emulator-test"], capture_output=True
    )

    # Start emulator
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

    os.environ["FIRESTORE_EMULATOR_HOST"] = f"localhost:{FIRESTORE_EMULATOR_PORT}"
    time.sleep(5)  # Wait for emulator to be ready

    yield

    # Cleanup
    subprocess.run(["docker", "stop", "firestore-emulator-test"], capture_output=True)
    subprocess.run(["docker", "rm", "firestore-emulator-test"], capture_output=True)
    os.environ.pop("FIRESTORE_EMULATOR_HOST", None)


@pytest.fixture(scope="session")
def gcs_emulator():
    """Start GCS emulator (fake-gcs-server) Docker container for test session."""
    # Clean up any existing container
    subprocess.run(["docker", "rm", "-f", "gcs-emulator-test"], capture_output=True)

    # Start fake-gcs-server emulator
    subprocess.run(
        [
            "docker",
            "run",
            "-d",
            "--name",
            "gcs-emulator-test",
            "-p",
            f"{GCS_EMULATOR_PORT}:4443",
            "fsouza/fake-gcs-server",
            "-scheme",
            "http",
            "-port",
            "4443",
        ],
        check=True,
        capture_output=True,
    )

    os.environ["STORAGE_EMULATOR_HOST"] = f"http://localhost:{GCS_EMULATOR_PORT}"
    time.sleep(3)  # Wait for emulator to be ready

    yield

    # Cleanup
    subprocess.run(["docker", "stop", "gcs-emulator-test"], capture_output=True)
    subprocess.run(["docker", "rm", "gcs-emulator-test"], capture_output=True)
    os.environ.pop("STORAGE_EMULATOR_HOST", None)


# ============================================================================
# Google Cloud Client Fixtures (Connected to Emulators)
# ============================================================================


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
def gcs_client(gcs_emulator) -> storage.Client:
    """Provide GCS client connected to emulator."""
    return storage.Client(project=PROJECT_ID)


@pytest.fixture(scope="session")
def gcs_bucket(gcs_client) -> storage.Bucket:
    """Create and provide test GCS bucket."""
    bucket = gcs_client.bucket(GCS_BUCKET_NAME)
    if not bucket.exists():
        bucket = gcs_client.create_bucket(GCS_BUCKET_NAME)
    return bucket


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


@pytest.fixture(scope="session")
def memory_service(gcs_bucket):
    """Create memory service with GCS adapter."""
    return MemoryService(storage=GCSMemoryAdapter(bucket=gcs_bucket))


@pytest.fixture(scope="session")
def agent_app(
    memory_service,
    llm_adapter,
    file_prompt_adapter,
    rabbitmq_publisher,
    rabbitmq_subscriber,
    rabbitmq_infrastructure,
):
    """Start the agent app for E2E tests."""
    # Adapters implement synapse protocols
    app = create_app(
        memory_service=memory_service,
        llm_adapter=llm_adapter,
        prompt_adapter=file_prompt_adapter,
        prompt_names=["codie_as_a_service_system"],
        publisher=rabbitmq_publisher,
        subscriber=rabbitmq_subscriber,
        request_subscription_path=REQUEST_SUBSCRIPTION,
        response_topic_path=RESPONSE_TOPIC,
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
        publisher: RabbitMQPublisher,
        subscriber: RabbitMQSubscriber,
        request_subscription: str,
        response_subscription: str,
    ):
        self._publisher = publisher
        self._subscriber = subscriber
        self._request_subscription = request_subscription
        self._response_subscription = response_subscription

    def send_request(
        self,
        user_id: str,
        session_id: str,
        message: str,
        output_format: dict | None = None,
        timeout_seconds: int = 10,
    ):
        """Publish request and wait for response."""
        request = RunAgentRequest(
            user_id=user_id,
            session_id=session_id,
            message=message,
            output_format=output_format,
        )
        # Publish to request topic
        self._publisher.publish(
            f":{self._request_subscription}", request.model_dump_json().encode("utf-8")
        ).result()

        # Wait for response
        deadline = time.time() + timeout_seconds
        while time.time() < deadline:
            pull_response = self._subscriber.pull(
                request={
                    "subscription": self._response_subscription,
                    "max_messages": 1,
                },
                timeout=1,
            )
            if pull_response.received_messages:
                msg = pull_response.received_messages[0]
                self._subscriber.acknowledge(
                    request={
                        "subscription": self._response_subscription,
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
        publisher=RabbitMQPublisher(client_connection),
        subscriber=RabbitMQSubscriber(client_connection),
        request_subscription=REQUEST_SUBSCRIPTION,
        response_subscription=RESPONSE_SUBSCRIPTION,
    )


# ============================================================================
# LLM Adapter Fixtures (Internal - Used by TestApp)
# ============================================================================


@pytest.fixture(scope="session")
def llm_adapter():
    """LocalLLMAdapter with mocked model and tokenizer.

    Used internally by TestApp. Tests should not use this fixture directly -
    use test_app.stub_llm_responses() instead.
    """
    from codie_as_a_service.adapters.llm.local_llm_adapter import LocalLLMAdapter

    # Create mock tokenizer and model
    mock_tokenizer = MagicMock()
    mock_model = MagicMock()

    # Configure tokenizer - return dict with input_ids and attention_mask (return_dict=True)
    mock_tokenizer.apply_chat_template.return_value = {
        "input_ids": torch.tensor([[1, 2, 3]]),
        "attention_mask": torch.tensor([[1, 1, 1]]),
    }
    mock_tokenizer.eos_token_id = 0
    mock_tokenizer.decode.return_value = "I'm ready to help you."

    # Configure model
    mock_model.generate.return_value = torch.tensor([[1, 2, 3, 4, 5, 6]])
    mock_model.to.return_value = mock_model

    # Create adapter without loading real model
    with (
        patch(
            "codie_as_a_service.adapters.llm.local_llm_adapter.AutoTokenizer"
        ) as mock_auto_tok,
        patch(
            "codie_as_a_service.adapters.llm.local_llm_adapter.AutoModelForCausalLM"
        ) as mock_auto_model,
    ):
        mock_auto_tok.from_pretrained.return_value = mock_tokenizer
        mock_auto_model.from_pretrained.return_value = mock_model

        adapter = LocalLLMAdapter(model_name="test-model", device="cpu")

    return adapter


@pytest.fixture(scope="session")
def file_prompt_adapter():
    """Create FilePromptAdapter with test prompts in temporary directory."""
    # Create temporary prompts directory
    temp_dir = tempfile.mkdtemp(prefix="test_prompts_")
    prompts_path = Path(temp_dir)

    # Define test prompts - using variables that ReActAgent actually passes
    # (me, context_anchors, current_session)
    test_prompts = {
        "codie_as_a_service_system.txt": (
            "You are a helpful AI assistant with access to user memory. "
            "Identity: {me}. Context: {context_anchors}. Session: {current_session}. "
            "You can read and write to the user's memory using the provided tools."
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
def test_user_id() -> str:
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


@pytest.fixture(scope="session")
def http_app(
    memory_service,
    llm_adapter,
    file_prompt_adapter,
    api_key_auth_adapter,
):
    """Create FastAPI HTTP app for E2E tests."""
    app = create_http_app(
        memory_service=memory_service,
        llm_adapter=llm_adapter,
        prompt_adapter=file_prompt_adapter,
        prompt_names=["codie_as_a_service_system"],
        auth=api_key_auth_adapter,
    )
    return app


@pytest.fixture(scope="session")
def api_key_auth_adapter():
    """Create API key auth adapter for tests."""
    from codie_as_a_service.adapters.auth.api_key_adapter import APIKeyAuthAdapter

    return APIKeyAuthAdapter(valid_key=TEST_API_KEY)


class HTTPTestClient:
    """Client for HTTP E2E tests - simulates external client interacting with the system."""

    def __init__(self, app, api_key: str):
        self._client = StarletteTestClient(app)
        self._api_key = api_key

    def health(self):
        """GET /health endpoint (no auth required)."""
        return self._client.get("/health")

    def chat_raw(
        self,
        user_id: str,
        session_id: str,
        message: str,
        api_key: str | None,
        output_format: dict | None = None,
    ):
        """POST to /chat and return raw response (for testing auth)."""
        payload = {"user_id": user_id, "session_id": session_id, "message": message}
        if output_format:
            payload["output_format"] = output_format
        headers = {}
        if api_key is not None:
            headers["X-API-Key"] = api_key
        return self._client.post("/chat", json=payload, headers=headers)

    def chat(
        self,
        user_id: str,
        session_id: str,
        message: str,
        output_format: dict | None = None,
    ) -> list[dict]:
        """POST to /chat and collect SSE events (uses configured API key)."""
        response = self.chat_raw(
            user_id, session_id, message, self._api_key, output_format
        )

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
        self._default_response = self._to_adapter_response(
            LLMResponseSpec(stop_reason="end_turn", content="I'm ready to help you.")
        )

    def setup_user(
        self,
        memory: dict[str, str] | None = None,
    ) -> tuple[str, str]:
        """
        Create a test user with given memory contents.

        Args:
            memory: Optional dict of memory key -> content.
                   Defaults to minimal identity if not provided.

        Returns:
            Tuple of (user_id, session_id)
        """
        user_id = str(uuid.uuid4())
        session_id = str(uuid.uuid4())

        # Default minimal memory if not specified
        if memory is None:
            memory = {
                "me": "# Identity",
                "context_anchors": "# Anchors",
                "current_session": "# Session",
            }

        for key, content in memory.items():
            self._memory_service.write_memory(user_id=user_id, key=key, content=content)

        return user_id, session_id

    def stub_llm_responses(self, *responses: LLMResponseSpec) -> None:
        """
        Configure LLM to return these responses in sequence.

        Args:
            responses: Domain-level response specs. Converted internally
                      to text that the adapter will parse.
        """
        adapter_responses = [self._to_adapter_response(r) for r in responses]
        self._llm_adapter._tokenizer.decode.side_effect = adapter_responses

    def reset_llm(self) -> None:
        """Reset LLM mock to default state."""
        self._llm_adapter._tokenizer.decode.side_effect = None
        self._llm_adapter._tokenizer.decode.return_value = self._default_response
        self._llm_adapter._model.generate.side_effect = None

    def chat(
        self,
        user_id: str,
        session_id: str,
        message: str,
        output_format: dict | None = None,
    ) -> list[dict]:
        """
        Make a chat request and return SSE events.

        Args:
            user_id: User identifier
            session_id: Session identifier
            message: User message
            output_format: Optional JSON schema for structured output

        Returns:
            List of SSE events as dicts with 'event' and 'data' keys
        """
        return self._http_client.chat(user_id, session_id, message, output_format)

    def chat_raw(
        self,
        user_id: str,
        session_id: str,
        message: str,
        api_key: str | None,
        output_format: dict | None = None,
    ):
        """Make a chat request and return raw response (for auth testing)."""
        return self._http_client.chat_raw(
            user_id, session_id, message, api_key, output_format
        )

    def health(self):
        """GET /health endpoint."""
        return self._http_client.health()

    def read_memory(self, user_id: str, key: str) -> str | None:
        """Read user memory (for verifying side effects)."""
        return self._memory_service.read_memory(user_id=user_id, key=key)

    def write_memory(self, user_id: str, key: str, content: str) -> None:
        """Write user memory (for test setup)."""
        self._memory_service.write_memory(user_id=user_id, key=key, content=content)

    def send_pubsub_request(
        self,
        user_id: str,
        session_id: str,
        message: str,
        output_format: dict | None = None,
        timeout_seconds: int = 10,
    ) -> AgentResponse | None:
        """
        Send a Pub/Sub request and wait for response.

        Args:
            user_id: User identifier
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
            user_id, session_id, message, output_format, timeout_seconds
        )

    def _to_adapter_response(self, spec: LLMResponseSpec) -> str:
        """
        Convert domain-level spec to text that the adapter will parse.

        This is the ONLY place that knows about adapter response format.
        LocalLLMAdapter parses <tool_call> tags from text content.
        """
        # Build content with tool calls as XML tags (SmolLM3 format)
        content_parts = []
        if spec.content:
            content_parts.append(spec.content)

        for tc in spec.tool_calls:
            tool_call_json = json.dumps({"name": tc.name, "arguments": tc.arguments})
            content_parts.append(f"<tool_call>{tool_call_json}</tool_call>")

        return " ".join(content_parts) if content_parts else ""


@pytest.fixture(scope="session")
def test_app(memory_service, llm_adapter, http_app) -> TestApp:
    """
    Provide the encapsulated test application for HTTP tests.

    This is the PRIMARY fixture HTTP tests should use. It hides all
    implementation details about adapters and wiring.
    """
    http_client = HTTPTestClient(http_app, api_key=TEST_API_KEY)
    return TestApp(
        memory_service=memory_service,
        llm_adapter=llm_adapter,
        http_client=http_client,
    )


@pytest.fixture(scope="session")
def pubsub_test_app(memory_service, llm_adapter, agent_app, test_client) -> TestApp:
    """
    Provide the encapsulated test application for Pub/Sub tests.

    This is the PRIMARY fixture Pub/Sub tests should use.
    """
    return TestApp(
        memory_service=memory_service,
        llm_adapter=llm_adapter,
        pubsub_client=test_client,
    )


@pytest.fixture(scope="session")
def http_test_client(http_app):
    """Provide an HTTP test client for E2E tests."""
    return HTTPTestClient(http_app)
