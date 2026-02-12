"""
E2E Acceptance Test: HTTP Chat Endpoint

Tests describe the system from a CLIENT perspective:
- Client POSTs to /chat endpoint with agent_id, session_id, message
- Client receives SSE stream with text events and done event
- Client knows NOTHING about handlers, agents, adapters, memory services, etc.

Tests use TestApp which encapsulates all implementation details.
If we change LLM adapters, only TestApp needs to change - not these tests.
"""

import pytest

from tests.conftest import LLMResponseSpec, ToolCallSpec


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

        test_app.stub_llm_responses(
            LLMResponseSpec(stop_reason="end_turn", content="I'm ready to help you."),
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

    def test_agent_writes_memory_when_tool_is_used(self, test_app):
        """
        Given: Client sends a request that triggers write_memory tool
        When: Agent processes the request and uses the tool
        Then: Memory content is persisted (observable side effect)

        This is E2E from client perspective: client sends message,
        memory contains expected content afterward. Client doesn't
        know HOW it happened, just that it DID happen.
        """
        agent_id, session_id = test_app.setup_agent()

        # Configure LLM to use write_memory tool
        test_app.stub_llm_responses(
            # Phase 1: ReAct loop - tool use
            LLMResponseSpec(
                stop_reason="tool_use",
                content="I'll save that preference.",
                tool_calls=[
                    ToolCallSpec(
                        name="write_memory",
                        arguments={
                            "key": "current_session",
                            "content": "# Session\n\nUser prefers dark mode.",
                        },
                    )
                ],
            ),
            LLMResponseSpec(stop_reason="end_turn", content="Got it!"),
        )

        # Client sends request
        events = test_app.chat(agent_id, session_id, "Remember I prefer dark mode")

        # Request completed successfully
        done_events = [e for e in events if e["event"] == "done"]
        assert len(done_events) == 1

        # E2E verification: memory contains the written content
        content = test_app.read_memory(agent_id, "current_session")
        assert "dark mode" in content

        test_app.reset_llm()

    def test_agent_reads_memory_when_tool_is_used(self, test_app):
        """
        Given: User has specific content in memory
        When: Client sends request that triggers read_memory tool
        Then: Request completes (agent had access to memory content)

        The E2E proof is that the tool execution path works.
        We can't verify the response content (mock is predetermined),
        but we verify the flow completes without error.
        """
        agent_id, session_id = test_app.setup_agent()

        # Pre-populate memory with content
        test_app.write_memory(
            agent_id, "current_session", "# Session\n\nWorking on PROJECT_ALPHA."
        )

        # Configure LLM to use read_memory tool
        test_app.stub_llm_responses(
            # Phase 1: ReAct loop - tool use
            LLMResponseSpec(
                stop_reason="tool_use",
                content="Let me check.",
                tool_calls=[
                    ToolCallSpec(
                        name="read_memory", arguments={"key": "current_session"}
                    )
                ],
            ),
            LLMResponseSpec(stop_reason="end_turn", content="You're on PROJECT_ALPHA."),
        )

        # Client sends request
        events = test_app.chat(agent_id, session_id, "What am I working on?")

        # Request completed successfully (tool execution worked)
        done_events = [e for e in events if e["event"] == "done"]
        assert len(done_events) == 1

        test_app.reset_llm()

    def test_agent_stops_at_max_iterations(self, test_app):
        """
        Given: LLM keeps requesting tool use indefinitely
        When: Agent reaches max iterations limit
        Then: Request completes (doesn't hang forever)

        This tests the safety mechanism from client perspective:
        even if something goes wrong, client gets a response.
        """
        agent_id, session_id = test_app.setup_agent()

        # Configure LLM to always request tool use (infinite loop scenario)
        tool_loop_response = LLMResponseSpec(
            stop_reason="tool_use",
            content="Let me check more...",
            tool_calls=[
                ToolCallSpec(name="read_memory", arguments={"key": "current_session"})
            ],
        )

        test_app.stub_llm_responses(
            *([tool_loop_response] * 10),
        )

        # Client sends request
        events = test_app.chat(agent_id, session_id, "Help me")

        # Request completed (agent stopped at max iterations)
        done_events = [e for e in events if e["event"] == "done"]
        assert len(done_events) == 1

        test_app.reset_llm()

    def test_structured_output_returns_structured_event(self, test_app):
        """
        Given: Client sends request with output_format schema
        When: Agent processes with two-phase approach (ReAct loop then structure output)
        Then: Client receives SSE stream with response event containing JSON data
        """
        agent_id, session_id = test_app.setup_agent()

        test_app.stub_llm_responses(
            LLMResponseSpec(
                stop_reason="end_turn", content="John's email is [email protected]"
            ),
        )

        events = test_app.chat(
            agent_id,
            session_id,
            "Extract contact: John ([email protected])",
        )

        response_events = [e for e in events if e["event"] == "response"]
        done_events = [e for e in events if e["event"] == "done"]

        assert len(response_events) == 1, "Expected exactly one response event"
        assert len(done_events) == 1, "Expected exactly one done event"

        response_data = response_events[0]["data"]
        assert response_data["output"] == "John's email is [email protected]"
        assert "done" in response_data

        test_app.reset_llm()

    def test_response_contains_session_context_fields(self, test_app):
        """
        Given: A user exists with identity in memory
        When: Client POSTs to /chat with message
        Then: Response contains session_id and done fields from SessionContext
        """
        agent_id, session_id = test_app.setup_agent()
        test_app.stub_llm_responses(
            LLMResponseSpec(stop_reason="end_turn", content="Hello!"),
        )
        events = test_app.chat(agent_id, session_id, "Hi")
        response_data = [e for e in events if e["event"] == "response"][0]["data"]
        assert "session_id" in response_data
        assert "done" in response_data
        assert response_data["done"] is True

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

    def test_agent_lists_memory_keys_when_tool_is_used(self, test_app):
        """
        Given: User has multiple memory keys
        When: Client sends request that triggers list_memory_keys tool
        Then: Request completes (agent had access to key list)

        This E2E test exercises list_memory_keys on the storage adapter.
        """
        agent_id, session_id = test_app.setup_agent()

        # Pre-populate memory with multiple keys
        test_app.write_memory(agent_id, "current_session", "# Session")
        test_app.write_memory(agent_id, "notes", "# Notes")

        # Configure LLM to use list_memory_keys tool
        test_app.stub_llm_responses(
            # Phase 1: ReAct loop - tool use
            LLMResponseSpec(
                stop_reason="tool_use",
                content="Let me list your memory keys.",
                tool_calls=[ToolCallSpec(name="list_memory_keys", arguments={})],
            ),
            LLMResponseSpec(
                stop_reason="end_turn", content="You have: current_session, notes"
            ),
        )

        # Client sends request
        events = test_app.chat(agent_id, session_id, "What memory keys do I have?")

        # Request completed successfully (tool execution worked)
        done_events = [e for e in events if e["event"] == "done"]
        assert len(done_events) == 1

        test_app.reset_llm()

    def test_request_completes_when_validation_needs_retry(self, test_app):
        """
        Given: Internal validation determines first attempt is incomplete
        When: System retries processing
        Then: Client receives a successful response

        The client doesn't know about iterations - it just gets a response.
        """
        agent_id, session_id = test_app.setup_agent()

        test_app.stub_llm_responses(
            LLMResponseSpec(stop_reason="end_turn", content="Working on it."),
            iterations=2,
        )

        events = test_app.chat(agent_id, session_id, "Help me with something complex")

        response_events = [e for e in events if e["event"] == "response"]
        done_events = [e for e in events if e["event"] == "done"]

        assert len(response_events) == 1
        assert len(done_events) == 1

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
        test_app.stub_llm_responses(
            LLMResponseSpec(stop_reason="end_turn", content="Still trying..."),
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
