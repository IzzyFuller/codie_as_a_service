"""CaaS Python client library.

Transport library for projects that want to use CaaS.
Two transports, two methods:

- stream() — HTTP SSE. POST to /chat, yield events as they arrive.
- request() — Pubsub. Publish request, wait for response, return it.

Usage:
    # HTTP against a live server
    client = CaaSClient(base_url="http://localhost:8080")
    for response in client.stream(agent_id="my-agent", message="Hello"):
        print(response.response)

    # HTTP against a FastAPI app instance (testing)
    client = CaaSClient(app=my_fastapi_app)
    for response in client.stream(agent_id="my-agent", message="Hello"):
        print(response.response)

    # Pubsub (message-driven)
    client = CaaSClient(connection=rabbitmq_connection)
    response = client.request(agent_id="my-agent", message="Hello")
    print(response.response)
"""

import json
import time
from typing import Any, Generator

import httpx
from pydantic import BaseModel
from starlette.testclient import TestClient

from codie_as_a_service.adapters.messaging.models import (
    AgentResponse,
    RunAgentRequest,
)


class ChatResponse(BaseModel):
    """Typed response from a CaaS call.

    For DefaultOutput responses, `response`, `session_id`, and `done`
    are populated directly. For custom output_format schemas, the full
    response is available in `raw_data`.
    """

    response: str = ""
    session_id: str = ""
    done: bool = False
    raw_data: dict[str, Any] = {}


class CaaSError(Exception):
    """Error from the CaaS service.

    Raised on SSE error events, pubsub error status, or timeouts.
    """

    def __init__(self, message: str):
        self.message = message
        super().__init__(message)


def _parse_sse_events(text: str) -> list[tuple[str, dict[str, Any]]]:
    """Parse SSE event stream text into (event_type, data) tuples."""
    events: list[tuple[str, dict[str, Any]]] = []
    current_event: str | None = None

    for line in text.split("\n"):
        line = line.strip()
        if not line:
            continue

        if line.startswith("event:"):
            current_event = line[6:].strip()
        elif line.startswith("data:"):
            data_str = line[5:].strip()
            if data_str and current_event is not None:
                events.append((current_event, json.loads(data_str)))

    return events


class CaaSClient:
    """Transport client for the CaaS service.

    Encapsulates RunAgentRequest construction, SSE parsing,
    queue declaration, and publish/subscribe flow.

    Args:
        base_url: URL of the CaaS HTTP server (live connections).
        app: FastAPI app instance (testing without a server).
        connection: pika.BlockingConnection for pubsub mode.
        timeout: Request timeout in seconds (default: 300).
        request_topic: Pubsub request topic (default: "agent.requests").
        response_topic: Pubsub response topic (default: "agent.responses").
    """

    def __init__(
        self,
        base_url: str | None = None,
        app: Any | None = None,
        connection: Any | None = None,
        timeout: float = 300.0,
        request_topic: str = "agent.requests",
        response_topic: str = "agent.responses",
    ):
        self._http_client: TestClient | httpx.Client | None = None
        self._connection: Any | None = None
        self._publisher: Any | None = None
        self._subscriber: Any | None = None
        self._timeout = timeout
        self._request_topic = request_topic
        self._response_topic = response_topic

        if app is not None:
            self._http_client = TestClient(app)
        elif base_url is not None:
            self._http_client = httpx.Client(base_url=base_url, timeout=timeout)
        elif connection is not None:
            from synapse.adapters.rabbitmq import RabbitMQPublisher, RabbitMQSubscriber

            self._connection = connection
            self._publisher = RabbitMQPublisher(connection)
            self._subscriber = RabbitMQSubscriber(connection)

    def stream(
        self,
        agent_id: str,
        message: str,
        session_id: str | None = None,
        output_format: dict[str, Any] | None = None,
    ) -> Generator[ChatResponse, None, None]:
        """Send a chat message via HTTP and yield responses as they arrive.

        Args:
            agent_id: Agent identifier for memory isolation.
            message: User message to the agent.
            session_id: Optional session ID (None = server generates one).
            output_format: Optional JSON Schema for structured output.

        Yields:
            ChatResponse per response event.

        Raises:
            CaaSError: If the server returns an error event.
        """
        request = RunAgentRequest(
            agent_id=agent_id,
            session_id=session_id,
            message=message,
            output_format=output_format,
        )

        payload = request.model_dump(exclude_none=True)

        response = self._http_client.post("/chat", json=payload)
        response.raise_for_status()

        events = _parse_sse_events(response.text)

        for event_type, data in events:
            if event_type == "error":
                raise CaaSError(data.get("message", "Unknown error"))

            if event_type == "response":
                yield self._build_chat_response(data)

    def request(
        self,
        agent_id: str,
        message: str,
        session_id: str | None = None,
        output_format: dict[str, Any] | None = None,
        timeout: float = 10.0,
    ) -> ChatResponse:
        """Send a message via pubsub and wait for the response.

        Args:
            agent_id: Agent identifier for memory isolation.
            message: User message to the agent.
            session_id: Optional session ID (None = server generates one).
            output_format: Optional JSON Schema for structured output.
            timeout: Seconds to wait for response (default: 10).

        Returns:
            ChatResponse with parsed fields and raw_data.

        Raises:
            CaaSError: On error status or timeout.
        """
        req = RunAgentRequest(
            agent_id=agent_id,
            session_id=session_id,
            message=message,
            output_format=output_format,
        )

        # Declare agent-specific response queue
        agent_response_queue = f"{self._response_topic}.{agent_id}"
        setup_channel = self._connection.channel()
        setup_channel.queue_declare(queue=agent_response_queue, durable=True)
        setup_channel.close()

        # Publish to request topic
        self._publisher.publish(
            f":{self._request_topic}", req.model_dump_json().encode("utf-8")
        ).result()

        # Poll for response
        deadline = time.time() + timeout
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
                agent_response = AgentResponse(**data)

                if agent_response.status == "error":
                    error_msg = agent_response.response_data.get(
                        "error", "Agent processing failed"
                    )
                    raise CaaSError(error_msg)

                return self._build_chat_response(agent_response.response_data)
            time.sleep(0.5)

        raise CaaSError(f"Request to agent '{agent_id}' timed out after {timeout}s")

    @staticmethod
    def _build_chat_response(data: dict[str, Any]) -> ChatResponse:
        """Build ChatResponse from response data dict."""
        return ChatResponse(
            response=data.get("response", ""),
            session_id=data.get("session_id", ""),
            done=data.get("done", False),
            raw_data=data,
        )
