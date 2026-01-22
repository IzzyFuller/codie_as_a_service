"""
E2E Acceptance Test: Agent Pub/Sub Message Processing

Tests describe the system from a CLIENT perspective:
- Client publishes RunAgentRequest to request topic
- Client listens on response topic for AgentResponse
- Client knows NOTHING about handlers, agents, adapters, memory services, etc.

Tests use TestApp which encapsulates all implementation details.
If we change LLM adapters, only TestApp needs to change - not these tests.
"""

import time

import pytest
import torch

from tests.conftest import LLMResponseSpec


@pytest.mark.integration
class TestE2EAgentPubSub:
    """E2E acceptance tests from client perspective."""

    def test_publish_request_and_receive_response(self, pubsub_test_app):
        """
        Given: A user exists with identity in memory
        When: Client publishes RunAgentRequest to request topic
        Then: Client receives AgentResponse on response topic
        """
        user_id, session_id = pubsub_test_app.setup_user()

        # Configure LLM responses for two-phase flow
        pubsub_test_app.stub_llm_responses(
            # Phase 1: ReAct loop response
            LLMResponseSpec(stop_reason="end_turn", content="I'm ready to help you."),
            # Phase 2: Default structured output format
            LLMResponseSpec(
                stop_reason="end_turn",
                content='{"response": "I\'m ready to help you."}',
            ),
        )

        response = pubsub_test_app.send_pubsub_request(
            user_id, session_id, "Hello, can you help me?"
        )

        assert response is not None, "Timed out waiting for response"
        assert response.user_id == user_id
        assert response.session_id == session_id
        assert response.status == "success"

        pubsub_test_app.reset_llm()

    def test_receive_error_response_when_processing_fails(self, pubsub_test_app):
        """
        Given: A request for a non-existent user
        When: Client publishes RunAgentRequest
        Then: Client receives AgentResponse with error status
        """
        response = pubsub_test_app.send_pubsub_request(
            "nonexistent_user", "some_session", "This should fail"
        )

        assert response is not None, "Timed out waiting for response"
        assert response.user_id == "nonexistent_user"
        assert response.status == "error"

    def test_structured_output_returns_response_data(self, pubsub_test_app):
        """
        Given: Client sends request with output_format schema
        When: Agent processes with two-phase approach (ReAct loop then structure output)
        Then: Client receives AgentResponse with response_data containing structured output
        """
        user_id, session_id = pubsub_test_app.setup_user()

        # Configure LLM responses for two-phase flow
        pubsub_test_app.stub_llm_responses(
            # Phase 1: ReAct loop response
            LLMResponseSpec(stop_reason="end_turn", content="Jane is 30 years old"),
            # Phase 2: Structured output response
            LLMResponseSpec(
                stop_reason="end_turn", content='{"name": "Jane", "age": 30}'
            ),
        )

        # Define output schema
        output_format = {
            "type": "json_schema",
            "schema": {
                "type": "object",
                "properties": {
                    "name": {"type": "string"},
                    "age": {"type": "integer"},
                },
                "required": ["name", "age"],
            },
        }

        response = pubsub_test_app.send_pubsub_request(
            user_id,
            session_id,
            "Extract person info: Jane is 30 years old",
            output_format=output_format,
        )

        assert response is not None, "Timed out waiting for response"
        assert response.user_id == user_id
        assert response.session_id == session_id
        assert response.status == "success"
        # Response always has response_data (structured output)
        assert response.response_data is not None
        assert response.response_data["name"] == "Jane"
        assert response.response_data["age"] == 30

        pubsub_test_app.reset_llm()

    def test_slow_llm_causes_client_timeout(self, pubsub_test_app):
        """
        Given: LLM takes longer than the client timeout
        When: Client publishes request and waits for response
        Then: Client times out and receives None

        This exercises the RabbitMQ subscriber's timeout path (inactivity_timeout).
        """
        user_id, session_id = pubsub_test_app.setup_user()

        # Make LLM mock sleep longer than our timeout
        def slow_generate(*args, **kwargs):
            time.sleep(3.0)  # Sleep 3 seconds
            return torch.tensor([[1, 2, 3, 4, 5, 6]])

        pubsub_test_app._llm_adapter._model.generate.side_effect = slow_generate

        # Use short timeout (2 seconds) - shorter than the 3 second sleep
        response = pubsub_test_app.send_pubsub_request(
            user_id, session_id, "Hello", timeout_seconds=2
        )

        # Should timeout and return None
        assert response is None, "Expected timeout (None), but got a response"

        pubsub_test_app.reset_llm()
