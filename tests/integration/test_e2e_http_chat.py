"""
E2E Acceptance Test: HTTP Chat Endpoint

Tests describe the system from a CLIENT perspective:
- Client POSTs to /chat endpoint with agent_id, session_id, message
- Client receives SSE stream with text events and done event
- Client knows NOTHING about handlers, agents, adapters, memory services, etc.

Tests use TestApp which encapsulates all implementation details.
If we change LLM adapters, only TestApp needs to change - not these tests.
"""

import json
import uuid

import pytest

from tests.conftest import LLMResponseSpec


@pytest.mark.integration
class TestE2EHTTPChat:
    """E2E acceptance tests for HTTP chat endpoint from client perspective."""

    def test_post_chat_returns_streaming_response(self, test_app):
        """
        Given: A user exists with identity in memory (no output_format specified)
        When: Client POSTs to /chat with message
        Then: Client receives SSE stream with response event containing default structured format
        """
        agent_id, session_id = test_app.setup_agent()

        test_app.stub_phases(
            process=[
                LLMResponseSpec(
                    stop_reason="end_turn", content="I'm ready to help you."
                ),
            ],
        )

        events = test_app.chat(agent_id, session_id, "Hello, can you help me?")

        # Should receive exactly one response event and a done event
        response_events = [e for e in events if e["event"] == "response"]
        done_events = [e for e in events if e["event"] == "done"]

        assert len(response_events) == 1, "Expected exactly one response event"
        assert len(done_events) == 1, "Expected exactly one done event"

        response_data = response_events[0]["data"]
        assert isinstance(response_data, dict), "Response should be structured dict"
        assert response_data["output"] == "I'm ready to help you."

        # Done event should have usage stats
        done_data = done_events[0]["data"]
        assert "usage" in done_data

        test_app.reset_llm()

    def test_chat_returns_error_for_nonexistent_user(self, test_app):
        """
        Given: A request for a non-existent user
        When: Client POSTs to /chat
        Then: Client receives error response
        """
        events = test_app.chat("nonexistent_user", "some_session", "This should fail")

        # Should receive an error event
        error_events = [e for e in events if e["event"] == "error"]
        assert len(error_events) == 1, "Expected exactly one error event"

        error_data = error_events[0]["data"]
        assert "message" in error_data

    def test_chat_returns_error_for_agent_without_frame(self, test_app):
        """
        Given: An agent exists with identity but no frame file
        When: Client POSTs to /chat
        Then: Client receives error response about missing frame
        """
        agent_id, session_id = test_app.setup_agent(
            memory={
                "me": "# Identity",
                "context_anchors": "# Anchors",
                "current_session": "# Session",
            }
        )
        test_app.stub_phases(
            process=[
                LLMResponseSpec(
                    stop_reason="end_turn", content="Should not reach here"
                ),
            ],
        )

        events = test_app.chat(agent_id, session_id, "Hello")

        error_events = [e for e in events if e["event"] == "error"]
        assert len(error_events) == 1, "Expected exactly one error event"
        assert "frame" in error_events[0]["data"]["message"].lower()

        test_app.reset_llm()

    def test_tool_using_request_completes_successfully(self, test_app):
        """
        Given: Client sends a request that would trigger tool use
        When: Adapter processes the request (tool execution is internal)
        Then: Client receives a successful response

        Tool execution is an adapter concern — with mocked adapters,
        we verify the flow completes. Real tool execution is tested
        via adapter-specific integration tests.
        """
        agent_id, session_id = test_app.setup_agent()

        test_app.stub_phases(
            process=[
                LLMResponseSpec(
                    stop_reason="end_turn", content="Got it, preference saved!"
                ),
            ],
        )

        # Client sends request
        events = test_app.chat(agent_id, session_id, "Remember I prefer dark mode")

        # Request completed successfully
        response_events = [e for e in events if e["event"] == "response"]
        done_events = [e for e in events if e["event"] == "done"]
        assert len(response_events) == 1
        assert len(done_events) == 1

        test_app.reset_llm()

    def test_request_completes_with_memory_context(self, test_app):
        """
        Given: User has specific content in memory
        When: Client sends request (adapter has access to memory via tools)
        Then: Request completes successfully

        The adapter handles tool execution internally — we verify
        the pipeline completes with pre-populated memory.
        """
        agent_id, session_id = test_app.setup_agent()

        # Pre-populate memory with content
        test_app.write_memory(
            agent_id, "current_session", "# Session\n\nWorking on PROJECT_ALPHA."
        )

        test_app.stub_phases(
            process=[
                LLMResponseSpec(
                    stop_reason="end_turn", content="You're on PROJECT_ALPHA."
                ),
            ],
        )

        events = test_app.chat(agent_id, session_id, "What am I working on?")

        done_events = [e for e in events if e["event"] == "done"]
        assert len(done_events) == 1

        test_app.reset_llm()

    def test_structured_output_returns_custom_schema_shape(self, test_app):
        """
        Given: Client sends request with output_format JSON schema
        When: Agent processes the request
        Then: Client receives response shaped to the custom schema, not DefaultOutput

        This is the critical test: output_format must flow from request → orchestrator
        and reshape the response to match the caller's schema.
        """
        agent_id, session_id = test_app.setup_agent()

        test_app.stub_phases(
            process=[
                LLMResponseSpec(
                    stop_reason="end_turn", content="John's email is [email protected]"
                ),
            ],
        )

        output_schema = {
            "type": "object",
            "properties": {
                "response": {"type": "string"},
                "done": {"type": "boolean"},
            },
            "required": ["response", "done"],
        }

        events = test_app.chat(
            agent_id,
            session_id,
            "Extract contact: John ([email protected])",
            output_format=output_schema,
        )

        response_events = [e for e in events if e["event"] == "response"]
        done_events = [e for e in events if e["event"] == "done"]

        assert len(response_events) == 1, "Expected exactly one response event"
        assert len(done_events) == 1, "Expected exactly one done event"

        response_data = response_events[0]["data"]
        # Custom schema: only "response" and "done" — no "output" or "session_id"
        assert "response" in response_data
        assert "done" in response_data
        assert "output" not in response_data, (
            "Custom schema should not include DefaultOutput's 'output' field"
        )
        assert "session_id" not in response_data, (
            "Custom schema should not include fields outside the requested schema"
        )

        test_app.reset_llm()

    def test_response_contains_session_context_fields(self, test_app):
        """
        Given: A user exists with identity in memory
        When: Client POSTs to /chat with message
        Then: Response contains session_id and done fields from SessionContext
        """
        agent_id, session_id = test_app.setup_agent()
        test_app.stub_phases(
            process=[
                LLMResponseSpec(stop_reason="end_turn", content="Hello!"),
            ],
        )
        events = test_app.chat(agent_id, session_id, "Hi")
        response_data = [e for e in events if e["event"] == "response"][0]["data"]
        assert "session_id" in response_data
        assert "done" in response_data
        assert response_data["done"] is True

        test_app.reset_llm()

    def test_backend_generates_session_id_when_omitted(self, test_app):
        """
        Given: A user exists with identity in memory
        When: Client POSTs to /chat without session_id
        Then: Backend generates a UUID session_id and returns it in the response
        """
        agent_id, _ = test_app.setup_agent()
        test_app.stub_phases(
            process=[
                LLMResponseSpec(stop_reason="end_turn", content="Hello!"),
            ],
        )
        events = test_app.chat(agent_id, message="Hi")
        response_data = [e for e in events if e["event"] == "response"][0]["data"]
        generated_id = response_data["session_id"]
        # Backend should have generated a valid UUID
        uuid.UUID(generated_id)  # raises ValueError if not a valid UUID

        test_app.reset_llm()

    def test_health_endpoint_does_not_require_auth(self, test_app):
        """
        Given: No authentication header
        When: Client GETs /health
        Then: Client receives 200 OK
        """
        response = test_app.health()
        assert response.status_code == 200
        assert response.json() == {"status": "ok"}

    def test_chat_returns_401_without_auth_header(self, test_app):
        """
        Given: No authentication header
        When: Client POSTs to /chat
        Then: Client receives 401 Unauthorized
        """
        response = test_app.chat_raw(
            agent_id="any",
            session_id="any",
            message="hello",
            api_key=None,
        )
        assert response.status_code == 401

    def test_chat_returns_401_with_invalid_api_key(self, test_app):
        """
        Given: Invalid API key in header
        When: Client POSTs to /chat
        Then: Client receives 401 Unauthorized
        """
        response = test_app.chat_raw(
            agent_id="any",
            session_id="any",
            message="hello",
            api_key="wrong-key",
        )
        assert response.status_code == 401

    def test_request_completes_when_validation_needs_retry(self, test_app):
        """
        Given: Internal validation determines first attempt is incomplete
        When: System retries processing
        Then: Client receives a successful response

        The client doesn't know about iterations - it just gets a response.
        """
        agent_id, session_id = test_app.setup_agent()

        test_app.stub_phases(
            process=[
                LLMResponseSpec(stop_reason="end_turn", content="Working on it."),
            ],
            iterations=2,
        )

        events = test_app.chat(agent_id, session_id, "Help me with something complex")

        response_events = [e for e in events if e["event"] == "response"]
        done_events = [e for e in events if e["event"] == "done"]

        assert len(response_events) == 1
        assert len(done_events) == 1

        test_app.reset_llm()

    def test_stub_phases_simple_end_turn(self, test_app):
        """
        Given: stub_phases configures per-phase responses directly
        When: Client POSTs to /chat with message
        Then: Client receives correct response through the factory path

        Validates the stub_phases factory produces correct side_effect
        ordering for the full 5-phase pipeline.
        """
        agent_id, session_id = test_app.setup_agent()

        test_app.stub_phases(
            process=[
                LLMResponseSpec(stop_reason="end_turn", content="Factory works!"),
            ],
        )

        events = test_app.chat(agent_id, session_id, "Test the factory")

        response_events = [e for e in events if e["event"] == "response"]
        done_events = [e for e in events if e["event"] == "done"]

        assert len(response_events) == 1
        assert len(done_events) == 1
        assert response_events[0]["data"]["output"] == "Factory works!"
        assert response_events[0]["data"]["done"] is True

        test_app.reset_llm()

    def test_synthesize_persists_done_true_after_successful_chat(self, test_app):
        """
        Given: A user exists with identity in memory
        When: Client POSTs to /chat and validation passes (done=true)
        Then: Persisted current_session contains done=true in the JSON entry

        SYNTHESIZE must run AFTER VALIDATE so the persisted snapshot
        reflects the final pipeline state, not an intermediate one.
        """
        agent_id, session_id = test_app.setup_agent()

        test_app.stub_phases(
            process=[
                LLMResponseSpec(stop_reason="end_turn", content="Persisted correctly!"),
            ],
        )

        events = test_app.chat(agent_id, session_id, "Test persistence")

        # Verify the request completed
        done_events = [e for e in events if e["event"] == "done"]
        assert len(done_events) == 1

        # Read persisted current_session and verify done=true in the entry
        persisted = test_app.read_memory(agent_id, "current_session")
        assert persisted is not None, "SYNTHESIZE should have written current_session"

        # Extract the JSON block from the persisted markdown entry
        json_start = persisted.index("{")
        json_end = persisted.rindex("}") + 1
        entry = json.loads(persisted[json_start:json_end])

        assert entry["done"] is True, (
            f"Persisted entry should have done=true, got done={entry['done']}. "
            "SYNTHESIZE likely ran before VALIDATE."
        )

        test_app.reset_llm()

    def test_request_completes_even_when_validation_never_satisfied(self, test_app):
        """
        Given: Internal validation is never fully satisfied
        When: System reaches its internal retry limit
        Then: Client still receives a response (doesn't hang forever)

        Safety mechanism: even if something goes wrong internally,
        the client always gets a response.
        """
        agent_id, session_id = test_app.setup_agent()

        # iterations=4 exceeds max_outer_iterations (3), so validation
        # never passes within the limit - tests the safety exit
        test_app.stub_phases(
            process=[
                LLMResponseSpec(stop_reason="end_turn", content="Still trying..."),
            ],
            iterations=4,
        )

        events = test_app.chat(agent_id, session_id, "This is tricky")

        done_events = [e for e in events if e["event"] == "done"]
        assert len(done_events) == 1

        test_app.reset_llm()


def test_list_memory_keys_for_nonexistent_user(memory_service):
    """
    Integration test: list_memory_keys returns empty list for non-existent user.

    This exercises the empty directory path in storage adapters (line 74).
    Can't be tested via agent E2E because agent requires identity files.
    """
    import uuid

    agent_id = f"nonexistent_user_{uuid.uuid4().hex[:8]}"

    keys = memory_service.list_memory_keys(agent_id=agent_id)

    assert keys == []
