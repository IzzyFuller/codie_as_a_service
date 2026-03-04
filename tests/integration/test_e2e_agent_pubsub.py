"""
E2E Acceptance Test: Agent Pub/Sub Message Processing

Tests describe the system from a CLIENT perspective using CaaSClient:
- Client calls request() with agent_id, message, optional session_id
- Client receives ChatResponse or CaaSError
- Client knows NOTHING about handlers, agents, adapters, memory services, etc.

Tests configure mock LLM responses inline and use caas_pubsub_client directly.
"""

import json
import time

import pytest

from codie_as_a_service.api.client import CaaSError, ChatResponse
from tests.conftest import get_llm_mock, setup_agent_memory


@pytest.mark.integration
class TestE2EAgentPubSub:
    """E2E acceptance tests from client perspective."""

    def test_publish_request_and_receive_response(
        self, pubsub_memory_service, pubsub_llm_adapter, agent_app, caas_pubsub_client
    ):
        """
        Given: A user exists with identity in memory
        When: Client sends request via pubsub
        Then: Client receives ChatResponse
        """
        agent_id, session_id = setup_agent_memory(pubsub_memory_service)

        # 3 LLM calls: HYDRATE → PROCESS → FORMAT
        mock = get_llm_mock(pubsub_llm_adapter)
        mock.side_effect = [
            "Identity summary for pubsub test.",
            "I'm ready to help you.",
            json.dumps(
                {"response": "I'm ready to help you.", "session_id": "", "done": True}
            ),
        ]

        response = caas_pubsub_client.request(
            agent_id=agent_id,
            session_id=session_id,
            message="Hello, can you help me?",
        )

        assert isinstance(response, ChatResponse)
        assert response.response == "I'm ready to help you."
        assert response.done is True

        mock.side_effect = None
        mock.return_value = json.dumps(
            {"response": "I'm ready to help you.", "session_id": "", "done": True}
        )

    def test_receive_error_response_when_processing_fails(
        self, pubsub_memory_service, pubsub_llm_adapter, agent_app, caas_pubsub_client
    ):
        """
        Given: A request for a non-existent user
        When: Client sends request via pubsub
        Then: CaaSError is raised
        """
        with pytest.raises(CaaSError):
            caas_pubsub_client.request(
                agent_id="nonexistent_user",
                session_id="some_session",
                message="This should fail",
            )

    def test_receive_error_response_for_agent_without_me(
        self, pubsub_memory_service, pubsub_llm_adapter, agent_app, caas_pubsub_client
    ):
        """
        Given: An agent exists with frame but no me file
        When: Client sends request via pubsub
        Then: CaaSError is raised (Pydantic ValidationError — me is required)
        """
        agent_id, session_id = setup_agent_memory(
            pubsub_memory_service,
            memory={
                "frame": "# Frame",
                "context_anchors": "# Anchors",
                "current_session": "# Session",
            },
        )

        with pytest.raises(CaaSError, match="me"):
            caas_pubsub_client.request(
                agent_id=agent_id, session_id=session_id, message="Hello"
            )

    def test_receive_error_response_for_agent_without_frame(
        self, pubsub_memory_service, pubsub_llm_adapter, agent_app, caas_pubsub_client
    ):
        """
        Given: An agent exists with identity but no frame file
        When: Client sends request via pubsub
        Then: CaaSError is raised (Pydantic ValidationError — frame is required)
        """
        agent_id, session_id = setup_agent_memory(
            pubsub_memory_service,
            memory={
                "me": "# Identity",
                "context_anchors": "# Anchors",
                "current_session": "# Session",
            },
        )

        with pytest.raises(CaaSError, match="frame"):
            caas_pubsub_client.request(
                agent_id=agent_id, session_id=session_id, message="Hello"
            )

    def test_structured_output_returns_custom_schema_shape(
        self, pubsub_memory_service, pubsub_llm_adapter, agent_app, caas_pubsub_client
    ):
        """
        Given: Client sends request with output_format JSON schema
        When: Agent processes the request
        Then: Client receives ChatResponse with raw_data shaped to custom schema

        This is the critical test: output_format must flow from RunAgentRequest
        through to orchestrator.run(), FORMAT must use the custom schema,
        and the response must match it.
        """
        agent_id, session_id = setup_agent_memory(pubsub_memory_service)

        output_schema = {
            "type": "object",
            "properties": {
                "name": {"type": "string"},
                "age": {"type": "integer"},
            },
        }

        # 3 LLM calls: HYDRATE → PROCESS → FORMAT (custom schema)
        mock = get_llm_mock(pubsub_llm_adapter)
        mock.side_effect = [
            "Identity summary for structured output test.",
            json.dumps({"name": "Jane", "age": 30}),
            json.dumps({"name": "Jane", "age": 30}),
        ]

        response = caas_pubsub_client.request(
            agent_id=agent_id,
            session_id=session_id,
            message="Extract person info: Jane is 30 years old",
            output_format=output_schema,
        )

        assert isinstance(response, ChatResponse)
        assert response.raw_data["name"] == "Jane"
        assert response.raw_data["age"] == 30

        mock.side_effect = None
        mock.return_value = json.dumps(
            {"response": "I'm ready to help you.", "session_id": "", "done": True}
        )

    def test_slow_llm_causes_client_timeout(
        self, pubsub_memory_service, pubsub_llm_adapter, agent_app, caas_pubsub_client
    ):
        """
        Given: LLM takes longer than the client timeout
        When: Client sends request and waits for response
        Then: CaaSError is raised with timeout message

        This exercises the RabbitMQ subscriber's timeout path (inactivity_timeout).
        """
        agent_id, session_id = setup_agent_memory(pubsub_memory_service)

        # Make LLM mock sleep longer than our timeout
        def slow_generate(*args, **kwargs):
            time.sleep(3.0)  # Sleep 3 seconds
            return "I'm ready to help you."

        get_llm_mock(pubsub_llm_adapter).side_effect = slow_generate

        # Use short timeout (2 seconds) - shorter than the 3 second sleep
        with pytest.raises(CaaSError, match="timed out"):
            caas_pubsub_client.request(
                agent_id=agent_id,
                session_id=session_id,
                message="Hello",
                timeout=2.0,
            )

        mock = get_llm_mock(pubsub_llm_adapter)
        mock.side_effect = None
        mock.return_value = json.dumps(
            {"response": "I'm ready to help you.", "session_id": "", "done": True}
        )
