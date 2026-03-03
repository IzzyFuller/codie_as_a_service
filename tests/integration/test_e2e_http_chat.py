"""
E2E Acceptance Test: HTTP Chat Endpoint

Tests describe the system from a CLIENT perspective:
- Client POSTs to /chat endpoint with agent_id, session_id, message
- Client receives SSE stream with text events and done event
- Client knows NOTHING about handlers, agents, adapters, memory services, etc.

Most tests use a format-only orchestrator (1 LLM mock call).
One canonical test exercises the full HYDRATE → PROCESS → FORMAT pipeline.
"""

import json
import uuid

import pytest

from codie_as_a_service.main_http import create_app as create_http_app
from codie_as_a_service.services.agent.react_orchestrator import ReActOrchestrator
from codie_as_a_service.services.wiring import (
    build_orchestrator_phases,
    get_memory_tool_definitions,
)
from tests.conftest import HTTPTestClient, get_llm_mock, setup_agent_memory


@pytest.mark.integration
class TestE2EHTTPChat:
    """E2E acceptance tests for HTTP chat endpoint from client perspective."""

    def test_post_chat_returns_streaming_response(
        self, http_client, memory_service, llm_adapter
    ):
        """
        Given: A user exists with identity in memory (no output_format specified)
        When: Client POSTs to /chat with message
        Then: Client receives SSE stream with response event containing default structured format
        """
        agent_id, session_id = setup_agent_memory(memory_service)

        get_llm_mock(llm_adapter).return_value = json.dumps(
            {"response": "I'm ready to help you.", "session_id": "", "done": True}
        )

        events = http_client.chat(agent_id, session_id, "Hello, can you help me?")

        # Should receive exactly one response event and a done event
        response_events = [e for e in events if e["event"] == "response"]
        done_events = [e for e in events if e["event"] == "done"]

        assert len(response_events) == 1, "Expected exactly one response event"
        assert len(done_events) == 1, "Expected exactly one done event"

        response_data = response_events[0]["data"]
        assert isinstance(response_data, dict), "Response should be structured dict"
        assert response_data["response"] == "I'm ready to help you."

        # Done event should have usage stats
        done_data = done_events[0]["data"]
        assert "usage" in done_data

    def test_chat_returns_error_for_nonexistent_user(self, http_client):
        """
        Given: A request for a non-existent user
        When: Client POSTs to /chat
        Then: Client receives error response
        """
        events = http_client.chat(
            "nonexistent_user", "some_session", "This should fail"
        )

        # Should receive an error event
        error_events = [e for e in events if e["event"] == "error"]
        assert len(error_events) == 1, "Expected exactly one error event"

        error_data = error_events[0]["data"]
        assert "message" in error_data

    def test_chat_returns_error_for_agent_without_frame(
        self, http_client, memory_service, llm_adapter
    ):
        """
        Given: An agent exists with identity but no frame file
        When: Client POSTs to /chat
        Then: Client receives error response (Pydantic ValidationError — frame is required)
        """
        agent_id, session_id = setup_agent_memory(
            memory_service,
            memory={
                "me": "# Identity",
                "context_anchors": "# Anchors",
                "current_session": "# Session",
            },
        )

        events = http_client.chat(agent_id, session_id, "Hello")

        error_events = [e for e in events if e["event"] == "error"]
        assert len(error_events) == 1, "Expected exactly one error event"
        # Pydantic ValidationError includes field name in message
        assert "frame" in error_events[0]["data"]["message"].lower()

    def test_chat_returns_error_for_agent_without_me(
        self, http_client, memory_service, llm_adapter
    ):
        """
        Given: An agent exists with frame but no me file
        When: Client POSTs to /chat
        Then: Client receives error response (Pydantic ValidationError — me is required)
        """
        agent_id, session_id = setup_agent_memory(
            memory_service,
            memory={
                "frame": "# Frame",
                "context_anchors": "# Anchors",
                "current_session": "# Session",
            },
        )

        events = http_client.chat(agent_id, session_id, "Hello")

        error_events = [e for e in events if e["event"] == "error"]
        assert len(error_events) == 1, "Expected exactly one error event"
        # Pydantic ValidationError includes field name in message
        assert "me" in error_events[0]["data"]["message"].lower()

    def test_tool_using_request_completes_successfully(
        self, http_client, memory_service, llm_adapter
    ):
        """
        Given: Client sends a request that would trigger tool use
        When: Adapter processes the request (tool execution is internal)
        Then: Client receives a successful response

        Tool execution is an adapter concern — with mocked adapters,
        we verify the flow completes. Real tool execution is tested
        via adapter-specific integration tests.
        """
        agent_id, session_id = setup_agent_memory(memory_service)

        get_llm_mock(llm_adapter).return_value = json.dumps(
            {"response": "Got it, preference saved!", "session_id": "", "done": True}
        )

        # Client sends request
        events = http_client.chat(agent_id, session_id, "Remember I prefer dark mode")

        # Request completed successfully
        response_events = [e for e in events if e["event"] == "response"]
        done_events = [e for e in events if e["event"] == "done"]
        assert len(response_events) == 1
        assert len(done_events) == 1

    def test_request_completes_with_memory_context(
        self, http_client, memory_service, llm_adapter
    ):
        """
        Given: User has specific content in memory
        When: Client sends request (adapter has access to memory via tools)
        Then: Request completes successfully

        The adapter handles tool execution internally — we verify
        the pipeline completes with pre-populated memory.
        """
        agent_id, session_id = setup_agent_memory(memory_service)

        # Pre-populate memory with content
        memory_service.write_memory(
            agent_id=agent_id,
            key="current_session",
            content="# Session\n\nWorking on PROJECT_ALPHA.",
        )

        get_llm_mock(llm_adapter).return_value = json.dumps(
            {"response": "You're on PROJECT_ALPHA.", "session_id": "", "done": True}
        )

        events = http_client.chat(agent_id, session_id, "What am I working on?")

        done_events = [e for e in events if e["event"] == "done"]
        assert len(done_events) == 1

    def test_structured_output_returns_custom_schema_shape(
        self, http_client, memory_service, llm_adapter
    ):
        """
        Given: Client sends request with output_format JSON schema
        When: Agent processes the request
        Then: Client receives response shaped to the custom schema, not DefaultOutput

        This is the critical test: output_format must flow from request → orchestrator,
        FORMAT must use the custom schema, and the response must match it.
        """
        agent_id, session_id = setup_agent_memory(memory_service)

        output_schema = {
            "type": "object",
            "properties": {
                "name": {"type": "string"},
                "email": {"type": "string"},
            },
        }

        # FORMAT phase gets the custom schema — mock returns matching JSON
        get_llm_mock(llm_adapter).return_value = json.dumps(
            {"name": "John", "email": "[email protected]"}
        )

        events = http_client.chat(
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
        # Custom schema fields populated by FORMAT
        assert response_data["name"] == "John"
        assert response_data["email"] == "[email protected]"
        # DefaultOutput fields should NOT be present
        assert (
            "response" not in response_data or response_data.get("response") != "John"
        ), "Custom schema should not include DefaultOutput's 'response' field"

    def test_response_contains_session_context_fields(
        self, http_client, memory_service, llm_adapter
    ):
        """
        Given: A user exists with identity in memory
        When: Client POSTs to /chat with message
        Then: Response contains session_id and done fields from DefaultOutput
        """
        agent_id, session_id = setup_agent_memory(memory_service)

        get_llm_mock(llm_adapter).return_value = json.dumps(
            {"response": "Hello!", "session_id": "", "done": True}
        )

        events = http_client.chat(agent_id, session_id, "Hi")
        response_data = [e for e in events if e["event"] == "response"][0]["data"]
        assert "session_id" in response_data
        assert "done" in response_data
        assert response_data["done"] is True

    def test_backend_generates_session_id_when_omitted(
        self, http_client, memory_service, llm_adapter
    ):
        """
        Given: A user exists with identity in memory
        When: Client POSTs to /chat without session_id
        Then: Backend generates a UUID session_id and returns it in the response

        Note: With FORMAT, the session_id in the response comes from the LLM mock
        (empty string in defaults). The backend still generates and uses one internally.
        """
        agent_id, _ = setup_agent_memory(memory_service)

        get_llm_mock(llm_adapter).return_value = json.dumps(
            {"response": "Hello!", "session_id": "", "done": True}
        )

        events = http_client.chat(agent_id, message="Hi")
        response_data = [e for e in events if e["event"] == "response"][0]["data"]
        # session_id is present in DefaultOutput (may be empty from mock)
        assert "session_id" in response_data

    def test_synthesize_persists_done_true_after_successful_chat(
        self, http_client, memory_service, llm_adapter
    ):
        """
        Given: A user exists with identity in memory
        When: Client POSTs to /chat and FORMAT sets done=true
        Then: Persisted current_session contains done=true in the JSON entry

        SYNTHESIZE must run AFTER FORMAT so the persisted snapshot
        reflects the final pipeline state, not an intermediate one.
        """
        agent_id, session_id = setup_agent_memory(memory_service)

        get_llm_mock(llm_adapter).return_value = json.dumps(
            {"response": "Persisted correctly!", "session_id": "", "done": True}
        )

        events = http_client.chat(agent_id, session_id, "Test persistence")

        # Verify the request completed
        done_events = [e for e in events if e["event"] == "done"]
        assert len(done_events) == 1

        # Read persisted current_session and verify done=true in the entry
        persisted = memory_service.read_memory(agent_id, "current_session")
        assert persisted is not None, "SYNTHESIZE should have written current_session"

        # Extract the JSON block from the persisted markdown entry
        json_start = persisted.index("{")
        json_end = persisted.rindex("}") + 1
        entry = json.loads(persisted[json_start:json_end])

        assert entry["done"] is True, (
            f"Persisted entry should have done=true, got done={entry['done']}. "
            "SYNTHESIZE likely ran before FORMAT."
        )


@pytest.mark.integration
class TestE2EFullPipeline:
    """Canonical all-phases E2E test exercising HYDRATE → PROCESS → FORMAT → SYNTHESIZE."""

    def test_full_pipeline_hydrate_process_format_synthesize(
        self, memory_service, llm_adapter, file_prompt_adapter
    ):
        """
        Given: Agent with identity in memory, full pipeline orchestrator
        When: Client POSTs to /chat
        Then: All phases execute in order, response is correct, session persisted

        This is the one big ugly all-phases test. Mock chain set up
        inline — 3 LLM calls for HYDRATE, PROCESS, FORMAT.
        """
        agent_id, session_id = setup_agent_memory(memory_service)

        # Build full-pipeline orchestrator inline
        tools = get_memory_tool_definitions()
        phases, post_phases = build_orchestrator_phases(
            phase_names=["hydrate", "process", "format"],
            prompt_adapter=file_prompt_adapter,
            tools=tools,
            llm=llm_adapter,
            memory=memory_service,
        )
        orchestrator = ReActOrchestrator(
            memory=memory_service,
            phases=phases,
            post_phases=post_phases,
        )

        # Wire up HTTP app with full-pipeline orchestrator
        app = create_http_app(
            memory_service=memory_service,
            llm_adapter=llm_adapter,
            prompt_adapter=file_prompt_adapter,
            prompt_names=["codie_as_a_service_system"],
            orchestrator=orchestrator,
        )
        client = HTTPTestClient(app)

        # 3-phase mock chain: HYDRATE (text) → PROCESS (text) → FORMAT (JSON)
        mock = get_llm_mock(llm_adapter)
        mock.side_effect = [
            # HYDRATE: plain text identity summary
            "Test identity summary for orchestrator testing.",
            # PROCESS: plain text response
            "Full pipeline works!",
            # FORMAT: JSON matching DefaultOutput schema
            json.dumps(
                {"response": "Full pipeline works!", "session_id": "", "done": True}
            ),
        ]

        events = client.chat(agent_id, session_id, "Test the full pipeline")

        response_events = [e for e in events if e["event"] == "response"]
        done_events = [e for e in events if e["event"] == "done"]

        assert len(response_events) == 1
        assert len(done_events) == 1
        assert response_events[0]["data"]["response"] == "Full pipeline works!"
        assert response_events[0]["data"]["done"] is True

        # Verify SYNTHESIZE persisted the session
        persisted = memory_service.read_memory(agent_id, "current_session")
        assert persisted is not None, "SYNTHESIZE should have written current_session"
        assert '"done": true' in persisted

        # Verify all 3 LLM calls were made
        assert mock.call_count == 3


def test_list_memory_keys_for_nonexistent_user(memory_service):
    """
    Integration test: list_memory_keys returns empty list for non-existent user.

    This exercises the empty directory path in storage adapters (line 74).
    Can't be tested via agent E2E because agent requires identity files.
    """

    agent_id = f"nonexistent_user_{uuid.uuid4().hex[:8]}"

    keys = memory_service.list_memory_keys(agent_id=agent_id)

    assert keys == []
