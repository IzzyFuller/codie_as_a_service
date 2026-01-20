"""
E2E Acceptance Test: Agent Pub/Sub Message Processing

Tests describe the system from a CLIENT perspective:
- Client publishes RunAgentRequest to request topic
- Client listens on response topic for AgentResponse
- Client knows NOTHING about handlers, agents, memory services, etc.

Mocking Strategy:
- Real Pub/Sub emulator (Docker)
- Real GCS emulator for memory
- The APP (not tests) will use mocked Anthropic/Langfuse SDKs
"""

import pytest
from anthropic.types import Message, TextBlock, Usage


def create_anthropic_message(
    stop_reason: str,
    content: list[TextBlock],
) -> Message:
    """Create Anthropic Message for configuring mock responses."""
    return Message(
        id="msg_test",
        content=content,
        model="test-model",
        role="assistant",
        stop_reason=stop_reason,
        stop_sequence=None,
        type="message",
        usage=Usage(
            input_tokens=10,
            output_tokens=10,
            cache_creation_input_tokens=0,
            cache_read_input_tokens=0,
        ),
    )


@pytest.mark.integration
class TestE2EAgentPubSub:
    """E2E acceptance tests from client perspective."""

    def test_publish_request_and_receive_response(
        self,
        agent_app,
        test_client,
        anthropic_client,
        create_test_session_with_user,
    ):
        """
        Given: A user exists with identity in memory
        When: Client publishes RunAgentRequest to request topic
        Then: Client receives AgentResponse on response topic
        """
        user_id, session_id = create_test_session_with_user()

        # Configure mock for two-phase approach (ReAct loop + structure output)
        anthropic_client.messages.create.side_effect = [
            # Phase 1: ReAct loop response
            create_anthropic_message(
                stop_reason="end_turn",
                content=[TextBlock(type="text", text="I'm ready to help you.")],
            ),
            # Phase 2: Default structured output format
            create_anthropic_message(
                stop_reason="end_turn",
                content=[TextBlock(type="text", text='{"response": "I\'m ready to help you."}')],
            ),
        ]

        response = test_client.send_request(
            user_id, session_id, "Hello, can you help me?"
        )

        assert response is not None, "Timed out waiting for response"
        assert response.user_id == user_id
        assert response.session_id == session_id
        assert response.status == "success"

        # Reset mock for other tests
        anthropic_client.messages.create.side_effect = None
        anthropic_client.messages.create.return_value = create_anthropic_message(
            stop_reason="end_turn",
            content=[TextBlock(type="text", text="I'm ready to help you.")],
        )

    def test_receive_error_response_when_processing_fails(
        self,
        agent_app,
        test_client,
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
        assert response.user_id == "nonexistent_user"
        assert response.status == "error"

    def test_structured_output_returns_response_data(
        self,
        agent_app,
        test_client,
        anthropic_client,
        create_test_session_with_user,
    ):
        """
        Given: Client sends request with output_format schema
        When: Agent processes with two-phase approach (ReAct loop then structure output)
        Then: Client receives AgentResponse with response_data containing structured output
        """
        user_id, session_id = create_test_session_with_user()

        # Configure mock to return text from ReAct loop, then JSON in structured output phase
        anthropic_client.messages.create.side_effect = [
            # Phase 1: ReAct loop response
            create_anthropic_message(
                stop_reason="end_turn",
                content=[TextBlock(type="text", text="Jane is 30 years old")],
            ),
            # Phase 2: Structured output response
            create_anthropic_message(
                stop_reason="end_turn",
                content=[TextBlock(type="text", text='{"name": "Jane", "age": 30}')],
            ),
        ]

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

        response = test_client.send_request(
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

        # Reset mock for other tests
        anthropic_client.messages.create.side_effect = None
        anthropic_client.messages.create.return_value = create_anthropic_message(
            stop_reason="end_turn",
            content=[TextBlock(type="text", text="I'm ready to help you.")],
        )
