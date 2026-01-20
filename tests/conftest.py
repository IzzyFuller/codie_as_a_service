"""Pytest configuration and shared fixtures for ATDD."""

import json
import os
import subprocess
import time
import uuid
from unittest.mock import MagicMock

import pytest
from anthropic import Anthropic
from anthropic.types import Message, TextBlock, ToolUseBlock, Usage
from google.cloud import firestore, storage
from google.cloud import pubsub_v1

from starlette.testclient import TestClient as StarletteTestClient

from deep_agent_service.core.models import RunAgentRequest, AgentResponse
from deep_agent_service.services.memory.memory_service import MemoryService
from deep_agent_service.adapters.storage.gcs_adapter import GCSMemoryAdapter
from deep_agent_service.main_pubsub import create_app
from deep_agent_service.main_http import create_app as create_http_app

# Test configuration
PROJECT_ID = "test-project"
GCS_BUCKET_NAME = "test-deep-agent-memory"
PUBSUB_EMULATOR_PORT = 8085
FIRESTORE_EMULATOR_PORT = 8086
GCS_EMULATOR_PORT = 4443
REQUEST_TOPIC_PATH = f"projects/{PROJECT_ID}/topics/agent-requests"
RESPONSE_TOPIC_PATH = f"projects/{PROJECT_ID}/topics/agent-responses"
REQUEST_SUBSCRIPTION_PATH = f"projects/{PROJECT_ID}/subscriptions/agent-requests-sub"
RESPONSE_SUBSCRIPTION_PATH = f"projects/{PROJECT_ID}/subscriptions/agent-responses-sub"


# ============================================================================
# Google Cloud Emulator Fixtures (Real Infrastructure for Tests)
# ============================================================================


@pytest.fixture(scope="session")
def pubsub_emulator():
    """Start Pub/Sub emulator Docker container for test session."""
    # Clean up any existing container
    subprocess.run(["docker", "rm", "-f", "pubsub-emulator-test"], capture_output=True)

    # Start emulator
    subprocess.run(
        [
            "docker",
            "run",
            "-d",
            "--name",
            "pubsub-emulator-test",
            "-p",
            f"{PUBSUB_EMULATOR_PORT}:8085",
            "google/cloud-sdk:emulators",
            "/bin/bash",
            "-c",
            f"gcloud beta emulators pubsub start --project={PROJECT_ID} --host-port=0.0.0.0:8085",
        ],
        check=True,
        capture_output=True,
    )

    os.environ["PUBSUB_EMULATOR_HOST"] = f"localhost:{PUBSUB_EMULATOR_PORT}"
    time.sleep(5)  # Wait for emulator to be ready

    yield

    # Cleanup
    subprocess.run(["docker", "stop", "pubsub-emulator-test"], capture_output=True)
    subprocess.run(["docker", "rm", "pubsub-emulator-test"], capture_output=True)
    os.environ.pop("PUBSUB_EMULATOR_HOST", None)


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
def pubsub_publisher(pubsub_emulator) -> pubsub_v1.PublisherClient:
    """Provide Pub/Sub publisher client connected to emulator."""
    return pubsub_v1.PublisherClient()


@pytest.fixture(scope="session")
def pubsub_subscriber(pubsub_emulator) -> pubsub_v1.SubscriberClient:
    """Provide Pub/Sub subscriber client connected to emulator."""
    return pubsub_v1.SubscriberClient()


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
def pubsub_infrastructure(pubsub_publisher, pubsub_subscriber):
    """Create Pub/Sub topics and subscriptions for tests."""
    pubsub_publisher.create_topic(request={"name": REQUEST_TOPIC_PATH})
    pubsub_publisher.create_topic(request={"name": RESPONSE_TOPIC_PATH})
    pubsub_subscriber.create_subscription(
        request={"name": REQUEST_SUBSCRIPTION_PATH, "topic": REQUEST_TOPIC_PATH}
    )
    pubsub_subscriber.create_subscription(
        request={"name": RESPONSE_SUBSCRIPTION_PATH, "topic": RESPONSE_TOPIC_PATH}
    )


@pytest.fixture(scope="session")
def memory_service(gcs_bucket):
    """Create memory service with GCS adapter."""
    return MemoryService(storage=GCSMemoryAdapter(bucket=gcs_bucket))


@pytest.fixture(scope="session")
def agent_app(
    memory_service,
    anthropic_client,
    mock_langfuse_client,
    pubsub_publisher,
    pubsub_subscriber,
    pubsub_infrastructure,
):
    """Start the agent app for E2E tests."""
    # GCP SubscriberClient directly implements PubSubSubscriber protocol (duck typing)
    app = create_app(
        memory_service=memory_service,
        anthropic_client=anthropic_client,
        langfuse_client=mock_langfuse_client,
        model="test-model",
        prompt_names=["deep_agent_service_system"],
        prompt_label="test",
        publisher=pubsub_publisher,
        subscriber=pubsub_subscriber,
        request_subscription_path=REQUEST_SUBSCRIPTION_PATH,
        response_topic_path=RESPONSE_TOPIC_PATH,
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
        self, publisher, subscriber, request_topic_path, response_subscription_path
    ):
        self._publisher = publisher
        self._subscriber = subscriber
        self._request_topic_path = request_topic_path
        self._response_subscription_path = response_subscription_path

    def send_request(
        self,
        user_id: str,
        session_id: str,
        message: str,
        output_format: dict | None = None,
        timeout_seconds: int = 10,
    ):
        """Publish request and wait for response."""
        # Publish request
        request = RunAgentRequest(
            user_id=user_id,
            session_id=session_id,
            message=message,
            output_format=output_format,
        )
        self._publisher.publish(
            self._request_topic_path, request.model_dump_json().encode("utf-8")
        ).result()

        # Wait for response
        deadline = time.time() + timeout_seconds
        while time.time() < deadline:
            pull_response = self._subscriber.pull(
                request={
                    "subscription": self._response_subscription_path,
                    "max_messages": 1,
                },
                timeout=1,
            )
            if pull_response.received_messages:
                msg = pull_response.received_messages[0]
                self._subscriber.acknowledge(
                    request={
                        "subscription": self._response_subscription_path,
                        "ack_ids": [msg.ack_id],
                    }
                )
                data = json.loads(msg.message.data.decode("utf-8"))
                return AgentResponse(**data)
            time.sleep(0.5)

        return None


@pytest.fixture(scope="session")
def test_client(pubsub_infrastructure):
    """Provide a test client with its own publisher/subscriber."""
    return TestClient(
        publisher=pubsub_v1.PublisherClient(),
        subscriber=pubsub_v1.SubscriberClient(),
        request_topic_path=REQUEST_TOPIC_PATH,
        response_subscription_path=RESPONSE_SUBSCRIPTION_PATH,
    )


# ============================================================================
# Test User Factory Fixture
# ============================================================================


@pytest.fixture
def create_test_session_with_user(memory_service):
    """Factory fixture for creating test users with identity and session."""

    def _create() -> tuple[str, str]:
        user_id = str(uuid.uuid4())
        session_id = str(uuid.uuid4())

        memory_service.write_memory(user_id=user_id, key="me", content="# Identity")
        memory_service.write_memory(
            user_id=user_id, key="context_anchors", content="# Anchors"
        )
        memory_service.write_memory(
            user_id=user_id, key="current_session", content="# Session"
        )

        return user_id, session_id

    return _create


# ============================================================================
# Anthropic SDK Fixtures
# ============================================================================

# Message counter for unique IDs
_message_counter = 0


def create_anthropic_message(
    stop_reason: str,
    content: list[TextBlock | ToolUseBlock],
) -> Message:
    """Factory for creating real Anthropic Message response objects.

    Use this in tests to build LLM responses. Tests configure
    anthropic_client.messages.create.side_effect with a list of these.

    Args:
        stop_reason: "end_turn" or "tool_use"
        content: List of TextBlock/ToolUseBlock objects

    Example:
        response = create_anthropic_message(
            stop_reason="end_turn",
            content=[TextBlock(type="text", text="Hello!")]
        )
        anthropic_client.messages.create.return_value = response
    """
    global _message_counter
    _message_counter += 1

    return Message(
        id=f"msg_test_{_message_counter}",
        content=content,
        model="claude-sonnet-4-20250514",
        role="assistant",
        stop_reason=stop_reason,
        stop_sequence=None,
        type="message",
        usage=Usage(
            input_tokens=100,
            output_tokens=50,
            cache_creation_input_tokens=0,
            cache_read_input_tokens=0,
        ),
    )


@pytest.fixture(scope="session")
def anthropic_client():
    """Real Anthropic client with only messages.create mocked.

    Tests configure the mock's return_value or side_effect as needed:
        anthropic_client.messages.create.return_value = create_anthropic_message(...)
        anthropic_client.messages.create.side_effect = [response1, response2]
    """
    client = Anthropic(api_key="test-key-for-testing")
    # Only mock the method that would hit the network
    client.messages.create = MagicMock()

    # Default response for tests that don't configure their own
    client.messages.create.return_value = create_anthropic_message(
        stop_reason="end_turn",
        content=[TextBlock(type="text", text="I'm ready to help you.")],
    )

    return client


@pytest.fixture(scope="session")
def mock_langfuse_client():
    """Create a mock Langfuse client that returns test prompts."""
    mock_client = MagicMock()

    def mock_get_prompt(name: str, label: str = "development"):
        """Return mock prompts based on name."""
        mock_prompt = MagicMock()

        # Define test prompts - using variables that ReActAgent actually passes
        # (me, context_anchors, current_session)
        prompts = {
            "deep_agent_service_system": (
                "You are a helpful AI assistant with access to user memory. "
                "Identity: {me}. Context: {context_anchors}. Session: {current_session}. "
                "You can read and write to the user's memory using the provided tools."
            ),
        }

        prompt_text = prompts.get(name, "Default test prompt")

        def mock_compile(**variables):
            """Handle template variable injection."""
            if variables:
                return prompt_text.format(**variables)
            return prompt_text

        mock_prompt.compile = mock_compile
        mock_prompt.prompt = prompt_text

        return mock_prompt

    mock_client.get_prompt = mock_get_prompt

    return mock_client


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
    anthropic_client,
    mock_langfuse_client,
):
    """Create FastAPI HTTP app for E2E tests."""
    app = create_http_app(
        memory_service=memory_service,
        anthropic_client=anthropic_client,
        langfuse_client=mock_langfuse_client,
        model="test-model",
        prompt_label="test",
        prompt_names=["deep_agent_service_system"],
    )
    return app


class HTTPTestClient:
    """Client for HTTP E2E tests - simulates external client interacting with the system."""

    def __init__(self, app):
        self._client = StarletteTestClient(app)

    def chat(
        self,
        user_id: str,
        session_id: str,
        message: str,
        output_format: dict | None = None,
    ) -> list[dict]:
        """POST to /chat and collect SSE events."""
        payload = {"user_id": user_id, "session_id": session_id, "message": message}
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


@pytest.fixture(scope="session")
def http_test_client(http_app):
    """Provide an HTTP test client for E2E tests."""
    return HTTPTestClient(http_app)
