"""Pytest configuration and shared fixtures for ATDD."""

import json
import os
import subprocess
import tempfile
import time
import uuid
from pathlib import Path
from unittest.mock import MagicMock

import pytest
from openai.types.chat import ChatCompletion, ChatCompletionMessage
from openai.types.chat.chat_completion import Choice
from google.cloud import firestore, storage
from google.cloud import pubsub_v1

from starlette.testclient import TestClient as StarletteTestClient

from deep_agent_service.adapters.prompts.file_adapter import FilePromptAdapter
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
    openai_client,
    file_prompt_adapter,
    pubsub_publisher,
    pubsub_subscriber,
    pubsub_infrastructure,
):
    """Start the agent app for E2E tests."""
    # GCP SubscriberClient directly implements PubSubSubscriber protocol (duck typing)
    app = create_app(
        memory_service=memory_service,
        llm_adapter=openai_client,
        prompt_adapter=file_prompt_adapter,
        prompt_names=["deep_agent_service_system"],
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
# OpenAI SDK Fixtures
# ============================================================================

# Message counter for unique IDs
_message_counter = 0


def _create_openai_response(
    stop_reason: str,
    content: str | None = None,
    tool_calls: list[dict] | None = None,
) -> ChatCompletion:
    """Factory for creating real OpenAI ChatCompletion response objects.

    Use this in tests to build LLM responses. Tests configure
    openai_client.chat.completions.create.side_effect with a list of these.

    Args:
        stop_reason: "stop" or "tool_calls"
        content: Text response content
        tool_calls: List of tool call objects (OpenAI format)

    Example:
        response = _create_openai_response(
            stop_reason="stop",
            content="Hello!"
        )
        openai_client.chat.completions.create.return_value = response
    """
    global _message_counter
    _message_counter += 1

    # Build the message content
    message_data = {
        "role": "assistant",
        "content": content,
    }
    if tool_calls:
        message_data["tool_calls"] = tool_calls

    message = ChatCompletionMessage(**message_data)

    # Map domain stop_reason to OpenAI finish_reason
    finish_reason_mapping = {
        "end_turn": "stop",
        "tool_use": "tool_calls",
        "max_tokens": "length",
    }
    finish_reason = finish_reason_mapping.get(stop_reason, "stop")

    choice = Choice(
        index=0,
        message=message,
        finish_reason=finish_reason,
        logprobs=None,
    )

    return ChatCompletion(
        id=f"chatcmpl_test_{_message_counter}",
        object="chat.completion",
        created=int(time.time()),
        model="gpt-4",
        choices=[choice],
        usage={
            "prompt_tokens": 100,
            "completion_tokens": 50,
            "total_tokens": 150,
        },
    )


@pytest.fixture(scope="session")
def create_openai_response():
    """Fixture that returns the response factory function for use in tests."""
    return _create_openai_response


@pytest.fixture(scope="session")
def openai_client():
    """Real OpenAI adapter with only chat.completions.create mocked.

    Tests configure the mock's return_value or side_effect as needed:
        openai_client._client.chat.completions.create.return_value = create_openai_response(...)
        openai_client._client.chat.completions.create.side_effect = [response1, response2]

    This follows the same pattern as the old Anthropic adapter testing:
    use real adapter, mock only the network call.
    """
    from deep_agent_service.adapters.llm.openai_adapter import OpenAILLMAdapter

    # Create real adapter (base_url doesn't matter since we mock the network call)
    adapter = OpenAILLMAdapter(base_url="http://localhost:8080/v1", model="test-model")

    # Only mock the method that would hit the network
    adapter._client.chat.completions.create = MagicMock()

    # Default response for tests that don't configure their own
    adapter._client.chat.completions.create.return_value = _create_openai_response(
        stop_reason="end_turn",
        content="I'm ready to help you.",
    )

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
        "deep_agent_service_system.txt": (
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
    openai_client,
    file_prompt_adapter,
):
    """Create FastAPI HTTP app for E2E tests."""
    app = create_http_app(
        memory_service=memory_service,
        llm_adapter=openai_client,
        prompt_adapter=file_prompt_adapter,
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
