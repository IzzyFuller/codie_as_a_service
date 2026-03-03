"""
E2E Acceptance Test: Agent Pub/Sub Message Processing

Tests describe the system from a CLIENT perspective:
- Client publishes RunAgentRequest to request topic
- Client listens on response topic for AgentResponse
- Client knows NOTHING about handlers, agents, adapters, memory services, etc.

Tests configure mock LLM responses inline and use test_client directly.
"""

import json
import time

import pytest

from tests.conftest import get_llm_mock, setup_agent_memory


@pytest.mark.integration
class TestE2EAgentPubSub:
    """E2E acceptance tests from client perspective."""

    def test_publish_request_and_receive_response(
        self, pubsub_memory_service, pubsub_llm_adapter, agent_app, test_client
    ):
        """
        Given: A user exists with identity in memory
        When: Client publishes RunAgentRequest to request topic
        Then: Client receives AgentResponse on response topic
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

        response = test_client.send_request(
            agent_id, session_id, "Hello, can you help me?"
        )

        assert response is not None, "Timed out waiting for response"
        assert response.agent_id == agent_id
        assert response.session_id == session_id
        assert response.status == "success"

        mock.side_effect = None
        mock.return_value = json.dumps(
            {"response": "I'm ready to help you.", "session_id": "", "done": True}
        )

    def test_receive_error_response_when_processing_fails(
        self, pubsub_memory_service, pubsub_llm_adapter, agent_app, test_client
    ):
        """
        Given: A request for a non-existent user
        When: Client publishes RunAgentRequest
        Then: Client receives AgentResponse with error status
        """
        response = test_client.send_request(
            "nonexistent_user", "some_session", "This should fail"
        )

        assert response is not None, "Timed out waiting for response"
        assert response.agent_id == "nonexistent_user"
        assert response.status == "error"

    def test_receive_error_response_for_agent_without_me(
        self, pubsub_memory_service, pubsub_llm_adapter, agent_app, test_client
    ):
        """
        Given: An agent exists with frame but no me file
        When: Client publishes RunAgentRequest
        Then: Client receives AgentResponse with error status (Pydantic ValidationError — me is required)
        """
        agent_id, session_id = setup_agent_memory(
            pubsub_memory_service,
            memory={
                "frame": "# Frame",
                "context_anchors": "# Anchors",
                "current_session": "# Session",
            },
        )

        response = test_client.send_request(agent_id, session_id, "Hello")

        assert response is not None, "Timed out waiting for response"
        assert response.agent_id == agent_id
        assert response.status == "error"
        # Pydantic ValidationError includes field name in message
        assert "me" in response.response_data["error"].lower()

    def test_receive_error_response_for_agent_without_frame(
        self, pubsub_memory_service, pubsub_llm_adapter, agent_app, test_client
    ):
        """
        Given: An agent exists with identity but no frame file
        When: Client publishes RunAgentRequest
        Then: Client receives AgentResponse with error status (Pydantic ValidationError — frame is required)
        """
        agent_id, session_id = setup_agent_memory(
            pubsub_memory_service,
            memory={
                "me": "# Identity",
                "context_anchors": "# Anchors",
                "current_session": "# Session",
            },
        )

        response = test_client.send_request(agent_id, session_id, "Hello")

        assert response is not None, "Timed out waiting for response"
        assert response.agent_id == agent_id
        assert response.status == "error"
        # Pydantic ValidationError includes field name in message
        assert "frame" in response.response_data["error"].lower()

    def test_structured_output_returns_custom_schema_shape(
        self, pubsub_memory_service, pubsub_llm_adapter, agent_app, test_client
    ):
        """
        Given: Client sends request with output_format JSON schema
        When: Agent processes the request
        Then: Client receives AgentResponse with response_data shaped to custom schema

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

        response = test_client.send_request(
            agent_id,
            session_id,
            "Extract person info: Jane is 30 years old",
            output_format=output_schema,
        )

        assert response is not None, "Timed out waiting for response"
        assert response.agent_id == agent_id
        assert response.session_id == session_id
        assert response.status == "success"
        assert response.response_data is not None
        # Custom schema fields populated by FORMAT
        assert response.response_data["name"] == "Jane"
        assert response.response_data["age"] == 30

        mock.side_effect = None
        mock.return_value = json.dumps(
            {"response": "I'm ready to help you.", "session_id": "", "done": True}
        )

    def test_slow_llm_causes_client_timeout(
        self, pubsub_memory_service, pubsub_llm_adapter, agent_app, test_client
    ):
        """
        Given: LLM takes longer than the client timeout
        When: Client publishes request and waits for response
        Then: Client times out and receives None

        This exercises the RabbitMQ subscriber's timeout path (inactivity_timeout).
        """
        agent_id, session_id = setup_agent_memory(pubsub_memory_service)

        # Make LLM mock sleep longer than our timeout
        def slow_generate(*args, **kwargs):
            time.sleep(3.0)  # Sleep 3 seconds
            return "I'm ready to help you."

        get_llm_mock(pubsub_llm_adapter).side_effect = slow_generate

        # Use short timeout (2 seconds) - shorter than the 3 second sleep
        response = test_client.send_request(
            agent_id, session_id, "Hello", timeout_seconds=2
        )

        # Should timeout and return None
        assert response is None, "Expected timeout (None), but got a response"

        mock = get_llm_mock(pubsub_llm_adapter)
        mock.side_effect = None
        mock.return_value = json.dumps(
            {"response": "I'm ready to help you.", "session_id": "", "done": True}
        )
