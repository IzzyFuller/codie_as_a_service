"""
E2E Acceptance Test: HTTP Chat Endpoint

Tests describe the system from a CLIENT perspective:
- Client POSTs to /chat endpoint with user_id, session_id, message
- Client receives SSE stream with text events and done event
- Client knows NOTHING about handlers, agents, memory services, etc.

Mocking Strategy:
- Real GCS emulator for memory (Docker)
- The APP (not tests) will use mocked Anthropic/Langfuse SDKs
"""

import pytest
from anthropic.types import Message, TextBlock, ToolUseBlock, Usage


def create_anthropic_message(
    stop_reason: str,
    content: list[TextBlock | ToolUseBlock],
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
class TestE2EHTTPChat:
    """E2E acceptance tests for HTTP chat endpoint from client perspective."""

    def test_post_chat_returns_streaming_response(
        self,
        http_app,
        http_test_client,
        anthropic_client,
        create_test_session_with_user,
    ):
        """
        Given: A user exists with identity in memory (no output_format specified)
        When: Client POSTs to /chat with message
        Then: Client receives SSE stream with response event containing default structured format
        """
        user_id, session_id = create_test_session_with_user()

        # Configure mock to return text from ReAct loop, then default JSON format
        anthropic_client.messages.create.side_effect = [
            # Phase 1: ReAct loop response
            create_anthropic_message(
                stop_reason="end_turn",
                content=[TextBlock(type="text", text="I'm ready to help you.")],
            ),
            # Phase 2: Default structured output format {"response": "text"}
            create_anthropic_message(
                stop_reason="end_turn",
                content=[TextBlock(type="text", text='{"response": "I\'m ready to help you."}')],
            ),
        ]

        events = http_test_client.chat(user_id, session_id, "Hello, can you help me?")

        # Should receive exactly one response event and a done event
        response_events = [e for e in events if e["event"] == "response"]
        done_events = [e for e in events if e["event"] == "done"]

        assert len(response_events) == 1, "Expected exactly one response event"
        assert len(done_events) == 1, "Expected exactly one done event"

        # Response event should have default structured format
        response_data = response_events[0]["data"]
        assert isinstance(response_data, dict), "Response should be structured dict"
        assert "response" in response_data, "Response should have 'response' field (default format)"

        # Done event should have usage stats
        done_data = done_events[0]["data"]
        assert "usage" in done_data

        # Reset mock for other tests
        anthropic_client.messages.create.side_effect = None
        anthropic_client.messages.create.return_value = create_anthropic_message(
            stop_reason="end_turn",
            content=[TextBlock(type="text", text="I'm ready to help you.")],
        )

    def test_chat_returns_error_for_nonexistent_user(
        self,
        http_app,
        http_test_client,
    ):
        """
        Given: A request for a non-existent user
        When: Client POSTs to /chat
        Then: Client receives error response
        """
        events = http_test_client.chat(
            "nonexistent_user", "some_session", "This should fail"
        )

        # Should receive an error event
        error_events = [e for e in events if e["event"] == "error"]
        assert len(error_events) == 1, "Expected exactly one error event"

        error_data = error_events[0]["data"]
        assert "message" in error_data

    def test_agent_writes_memory_when_tool_is_used(
        self,
        http_app,
        http_test_client,
        anthropic_client,
        memory_service,
        create_test_session_with_user,
    ):
        """
        Given: Client sends a request that triggers write_memory tool
        When: Agent processes the request and uses the tool
        Then: Memory content is persisted (observable side effect)

        This is E2E from client perspective: client sends message,
        memory contains expected content afterward. Client doesn't
        know HOW it happened, just that it DID happen.
        """
        user_id, session_id = create_test_session_with_user()

        # Configure mock to simulate agent using write_memory tool + structure output
        anthropic_client.messages.create.side_effect = [
            # Phase 1: ReAct loop - tool use
            create_anthropic_message(
                stop_reason="tool_use",
                content=[
                    TextBlock(type="text", text="I'll save that preference."),
                    ToolUseBlock(
                        type="tool_use",
                        id="tool_write_1",
                        name="write_memory",
                        input={
                            "key": "current_session",
                            "content": "# Session\n\nUser prefers dark mode.",
                        },
                    ),
                ],
            ),
            # Phase 1: ReAct loop - final response
            create_anthropic_message(
                stop_reason="end_turn",
                content=[TextBlock(type="text", text="Got it!")],
            ),
            # Phase 2: Default structured output format
            create_anthropic_message(
                stop_reason="end_turn",
                content=[TextBlock(type="text", text='{"response": "Got it!"}')],
            ),
        ]

        # Client sends request
        events = http_test_client.chat(
            user_id, session_id, "Remember I prefer dark mode"
        )

        # Request completed successfully
        done_events = [e for e in events if e["event"] == "done"]
        assert len(done_events) == 1

        # E2E verification: memory contains the written content
        content = memory_service.read_memory(user_id=user_id, key="current_session")
        assert "dark mode" in content

        # Reset mock for other tests
        anthropic_client.messages.create.side_effect = None
        anthropic_client.messages.create.return_value = create_anthropic_message(
            stop_reason="end_turn",
            content=[TextBlock(type="text", text="I'm ready to help you.")],
        )

    def test_agent_reads_memory_when_tool_is_used(
        self,
        http_app,
        http_test_client,
        anthropic_client,
        memory_service,
        create_test_session_with_user,
    ):
        """
        Given: User has specific content in memory
        When: Client sends request that triggers read_memory tool
        Then: Request completes (agent had access to memory content)

        The E2E proof is that the tool execution path works.
        We can't verify the response content (mock is predetermined),
        but we verify the flow completes without error.
        """
        user_id, session_id = create_test_session_with_user()

        # Pre-populate memory with content
        memory_service.write_memory(
            user_id=user_id,
            key="current_session",
            content="# Session\n\nWorking on PROJECT_ALPHA.",
        )

        # Configure mock to simulate agent using read_memory tool + structure output
        anthropic_client.messages.create.side_effect = [
            # Phase 1: ReAct loop - tool use
            create_anthropic_message(
                stop_reason="tool_use",
                content=[
                    TextBlock(type="text", text="Let me check."),
                    ToolUseBlock(
                        type="tool_use",
                        id="tool_read_1",
                        name="read_memory",
                        input={"key": "current_session"},
                    ),
                ],
            ),
            # Phase 1: ReAct loop - final response
            create_anthropic_message(
                stop_reason="end_turn",
                content=[TextBlock(type="text", text="You're on PROJECT_ALPHA.")],
            ),
            # Phase 2: Default structured output format
            create_anthropic_message(
                stop_reason="end_turn",
                content=[TextBlock(type="text", text='{"response": "You\'re on PROJECT_ALPHA."}')],
            ),
        ]

        # Client sends request
        events = http_test_client.chat(user_id, session_id, "What am I working on?")

        # Request completed successfully (tool execution worked)
        done_events = [e for e in events if e["event"] == "done"]
        assert len(done_events) == 1

        # Reset mock for other tests
        anthropic_client.messages.create.side_effect = None
        anthropic_client.messages.create.return_value = create_anthropic_message(
            stop_reason="end_turn",
            content=[TextBlock(type="text", text="I'm ready to help you.")],
        )

    def test_agent_stops_at_max_iterations(
        self,
        http_app,
        http_test_client,
        anthropic_client,
        create_test_session_with_user,
    ):
        """
        Given: LLM keeps requesting tool use indefinitely
        When: Agent reaches max iterations limit
        Then: Request completes (doesn't hang forever)

        This tests the safety mechanism from client perspective:
        even if something goes wrong, client gets a response.
        """
        user_id, session_id = create_test_session_with_user()

        # Configure mock to always request tool use (infinite loop scenario) + final structure
        # The ReAct loop will hit max iterations, then structure output needs one more mock
        tool_use_response = create_anthropic_message(
            stop_reason="tool_use",
            content=[
                TextBlock(type="text", text="Let me check more..."),
                ToolUseBlock(
                    type="tool_use",
                    id="tool_loop",
                    name="read_memory",
                    input={"key": "current_session"},
                ),
            ],
        )

        # Set up mock to return tool_use repeatedly during ReAct loop, then structure output
        # max_iterations defaults to 10, so 10 tool_use responses + 1 structure output
        anthropic_client.messages.create.side_effect = (
            [tool_use_response] * 10  # Hit max_iterations exactly
            + [
                create_anthropic_message(
                    stop_reason="end_turn",
                    content=[TextBlock(type="text", text='{"response": "I couldn\'t complete the request."}')],
                )
            ]
        )

        # Client sends request
        events = http_test_client.chat(user_id, session_id, "Help me")

        # Request completed (agent stopped at max iterations)
        done_events = [e for e in events if e["event"] == "done"]
        assert len(done_events) == 1

        # Verify multiple iterations occurred
        assert anthropic_client.messages.create.call_count >= 3

        # Reset mock for other tests
        anthropic_client.messages.create.side_effect = None
        anthropic_client.messages.create.return_value = create_anthropic_message(
            stop_reason="end_turn",
            content=[TextBlock(type="text", text="I'm ready to help you.")],
        )

    def test_structured_output_returns_structured_event(
        self,
        http_app,
        http_test_client,
        anthropic_client,
        create_test_session_with_user,
    ):
        """
        Given: Client sends request with output_format schema
        When: Agent processes with two-phase approach (ReAct loop then structure output)
        Then: Client receives SSE stream with response event containing JSON data
        """
        user_id, session_id = create_test_session_with_user()

        # Configure mock to return text from ReAct loop, then JSON in structured output phase
        anthropic_client.messages.create.side_effect = [
            # Phase 1: ReAct loop response
            create_anthropic_message(
                stop_reason="end_turn",
                content=[TextBlock(type="text", text="John's email is [email protected]")],
            ),
            # Phase 2: Structured output response
            create_anthropic_message(
                stop_reason="end_turn",
                content=[TextBlock(type="text", text='{"name": "John", "email": "[email protected]"}')],
            ),
        ]

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
        events = http_test_client.chat(
            user_id,
            session_id,
            "Extract contact: John ([email protected])",
            output_format=output_format,
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

        # Reset mock for other tests
        anthropic_client.messages.create.side_effect = None
        anthropic_client.messages.create.return_value = create_anthropic_message(
            stop_reason="end_turn",
            content=[TextBlock(type="text", text="I'm ready to help you.")],
        )

    def test_structured_output_with_empty_response_returns_error(
        self,
        http_app,
        http_test_client,
        anthropic_client,
        create_test_session_with_user,
    ):
        """
        Given: Client sends request with output_format schema
        When: LLM returns empty content in structure phase (edge case)
        Then: Client receives error event
        """
        user_id, session_id = create_test_session_with_user()

        # Configure mock to return text in ReAct loop, then empty content in structure phase
        anthropic_client.messages.create.side_effect = [
            # Phase 1: ReAct loop response
            create_anthropic_message(
                stop_reason="end_turn",
                content=[TextBlock(type="text", text="Some text result")],
            ),
            # Phase 2: Structured output response (empty - error case)
            create_anthropic_message(
                stop_reason="end_turn",
                content=[],  # Empty content - should trigger error
            ),
        ]

        # Define output schema
        output_format = {
            "type": "json_schema",
            "schema": {
                "type": "object",
                "properties": {"name": {"type": "string"}},
            },
        }

        # Client sends request with output_format
        events = http_test_client.chat(
            user_id,
            session_id,
            "Extract something",
            output_format=output_format,
        )

        # Should receive error event due to empty content in structure phase
        error_events = [e for e in events if e["event"] == "error"]
        assert len(error_events) == 1, "Expected exactly one error event"

        # Reset mock for other tests
        anthropic_client.messages.create.side_effect = None
        anthropic_client.messages.create.return_value = create_anthropic_message(
            stop_reason="end_turn",
            content=[TextBlock(type="text", text="I'm ready to help you.")],
        )
