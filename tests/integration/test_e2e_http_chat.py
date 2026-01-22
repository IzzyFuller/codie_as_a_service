"""
E2E Acceptance Test: HTTP Chat Endpoint

Tests describe the system from a CLIENT perspective:
- Client POSTs to /chat endpoint with user_id, session_id, message
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
        user_id, session_id = test_app.setup_user()

        # Configure LLM responses for two-phase flow
        test_app.stub_llm_responses(
            # Phase 1: ReAct loop response
            LLMResponseSpec(stop_reason="end_turn", content="I'm ready to help you."),
            # Phase 2: Default structured output format {"response": "text"}
            LLMResponseSpec(
                stop_reason="end_turn",
                content='{"response": "I\'m ready to help you."}',
            ),
        )

        events = test_app.chat(user_id, session_id, "Hello, can you help me?")

        # Should receive exactly one response event and a done event
        response_events = [e for e in events if e["event"] == "response"]
        done_events = [e for e in events if e["event"] == "done"]

        assert len(response_events) == 1, "Expected exactly one response event"
        assert len(done_events) == 1, "Expected exactly one done event"

        # Response event should have default structured format
        response_data = response_events[0]["data"]
        assert isinstance(response_data, dict), "Response should be structured dict"
        assert "response" in response_data, (
            "Response should have 'response' field (default format)"
        )

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
        user_id, session_id = test_app.setup_user()

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
            # Phase 1: ReAct loop - final response
            LLMResponseSpec(stop_reason="end_turn", content="Got it!"),
            # Phase 2: Default structured output format
            LLMResponseSpec(stop_reason="end_turn", content='{"response": "Got it!"}'),
        )

        # Client sends request
        events = test_app.chat(user_id, session_id, "Remember I prefer dark mode")

        # Request completed successfully
        done_events = [e for e in events if e["event"] == "done"]
        assert len(done_events) == 1

        # E2E verification: memory contains the written content
        content = test_app.read_memory(user_id, "current_session")
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
        user_id, session_id = test_app.setup_user()

        # Pre-populate memory with content
        test_app.write_memory(
            user_id, "current_session", "# Session\n\nWorking on PROJECT_ALPHA."
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
            # Phase 1: ReAct loop - final response
            LLMResponseSpec(stop_reason="end_turn", content="You're on PROJECT_ALPHA."),
            # Phase 2: Default structured output format
            LLMResponseSpec(
                stop_reason="end_turn",
                content='{"response": "You\'re on PROJECT_ALPHA."}',
            ),
        )

        # Client sends request
        events = test_app.chat(user_id, session_id, "What am I working on?")

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
        user_id, session_id = test_app.setup_user()

        # Configure LLM to always request tool use (infinite loop scenario)
        tool_loop_response = LLMResponseSpec(
            stop_reason="tool_use",
            content="Let me check more...",
            tool_calls=[
                ToolCallSpec(name="read_memory", arguments={"key": "current_session"})
            ],
        )

        # max_iterations defaults to 10, so 10 tool_calls responses + 1 structure output
        test_app.stub_llm_responses(
            *([tool_loop_response] * 10),
            LLMResponseSpec(
                stop_reason="end_turn",
                content='{"response": "I couldn\'t complete the request."}',
            ),
        )

        # Client sends request
        events = test_app.chat(user_id, session_id, "Help me")

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
        user_id, session_id = test_app.setup_user()

        # Configure LLM responses for two-phase flow
        test_app.stub_llm_responses(
            # Phase 1: ReAct loop response
            LLMResponseSpec(
                stop_reason="end_turn", content="John's email is [email protected]"
            ),
            # Phase 2: Structured output response
            LLMResponseSpec(
                stop_reason="end_turn",
                content='{"name": "John", "email": "[email protected]"}',
            ),
        )

        # Define output schema
        output_format = {
            "type": "json_schema",
            "schema": {
                "type": "object",
                "properties": {
                    "name": {"type": "string"},
                    "email": {"type": "string"},
                },
                "required": ["name", "email"],
            },
        }

        # Client sends request with output_format
        events = test_app.chat(
            user_id,
            session_id,
            "Extract contact: John ([email protected])",
            output_format,
        )

        # Should receive response event (always)
        response_events = [e for e in events if e["event"] == "response"]
        done_events = [e for e in events if e["event"] == "done"]

        assert len(response_events) == 1, "Expected exactly one response event"
        assert len(done_events) == 1, "Expected exactly one done event"

        # Response event should have the parsed JSON data
        response_data = response_events[0]["data"]
        assert response_data["name"] == "John"
        assert response_data["email"] == "[email protected]"

        test_app.reset_llm()

    def test_structured_output_with_empty_response_returns_error(self, test_app):
        """
        Given: Client sends request with output_format schema
        When: LLM returns empty content in structure phase (edge case)
        Then: Client receives error event
        """
        user_id, session_id = test_app.setup_user()

        # Configure LLM to return text in ReAct loop, then empty content in structure phase
        test_app.stub_llm_responses(
            # Phase 1: ReAct loop response
            LLMResponseSpec(stop_reason="end_turn", content="Some text result"),
            # Phase 2: Structured output response (empty - error case)
            LLMResponseSpec(stop_reason="end_turn", content=None),
        )

        # Define output schema
        output_format = {
            "type": "json_schema",
            "schema": {
                "type": "object",
                "properties": {"name": {"type": "string"}},
            },
        }

        # Client sends request with output_format
        events = test_app.chat(user_id, session_id, "Extract something", output_format)

        # Should receive error event due to empty content in structure phase
        error_events = [e for e in events if e["event"] == "error"]
        assert len(error_events) == 1, "Expected exactly one error event"

        test_app.reset_llm()
