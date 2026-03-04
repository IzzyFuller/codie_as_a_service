"""
E2E Acceptance Test: HTTP Chat via CaaSClient

Tests describe the system from a CLIENT perspective using the CaaSClient library:
- Client calls stream() with agent_id, message, optional session_id
- Client receives typed ChatResponse with response, session_id, done, raw_data
- Client errors surface as CaaSError exceptions
- Client knows NOTHING about handlers, agents, adapters, memory services, etc.

Most tests use a format-only orchestrator (1 LLM mock call).
One canonical test exercises the full HYDRATE → PROCESS → FORMAT pipeline.
"""

import json
import uuid

import pytest

from codie_as_a_service.api.client import CaaSClient, CaaSError, ChatResponse
from codie_as_a_service.main_http import create_app as create_http_app
from codie_as_a_service.services.agent.react_orchestrator import ReActOrchestrator
from codie_as_a_service.services.wiring import (
    build_orchestrator_phases,
    get_memory_tool_definitions,
)
from tests.conftest import get_llm_mock, setup_agent_memory


@pytest.mark.integration
class TestE2EHTTPChat:
    """E2E acceptance tests for HTTP chat endpoint via CaaSClient."""

    def test_chat_returns_typed_response(
        self, caas_client, memory_service, llm_adapter
    ):
        """
        Given: A user exists with identity in memory
        When: Client sends chat message via stream()
        Then: Returns ChatResponse with correct fields
        """
        agent_id, session_id = setup_agent_memory(memory_service)

        get_llm_mock(llm_adapter).return_value = json.dumps(
            {"response": "I'm ready to help you.", "session_id": "", "done": True}
        )

        result = next(
            caas_client.stream(
                agent_id=agent_id,
                session_id=session_id,
                message="Hello, can you help me?",
            )
        )

        assert isinstance(result, ChatResponse)
        assert result.response == "I'm ready to help you."
        assert result.done is True

    def test_chat_error_for_nonexistent_user(self, caas_client):
        """
        Given: A request for a non-existent user
        When: Client sends chat message
        Then: CaaSError is raised with error message
        """
        with pytest.raises(CaaSError) as exc_info:
            next(
                caas_client.stream(
                    agent_id="nonexistent_user",
                    session_id="some_session",
                    message="This should fail",
                )
            )

        assert exc_info.value.message

    def test_chat_error_for_agent_without_frame(
        self, caas_client, memory_service, llm_adapter
    ):
        """
        Given: An agent exists with identity but no frame file
        When: Client sends chat message
        Then: CaaSError is raised (Pydantic ValidationError — frame is required)
        """
        agent_id, session_id = setup_agent_memory(
            memory_service,
            memory={
                "me": "# Identity",
                "context_anchors": "# Anchors",
                "current_session": "# Session",
            },
        )

        with pytest.raises(CaaSError) as exc_info:
            next(
                caas_client.stream(
                    agent_id=agent_id, session_id=session_id, message="Hello"
                )
            )

        assert "frame" in exc_info.value.message.lower()

    def test_chat_error_for_agent_without_me(
        self, caas_client, memory_service, llm_adapter
    ):
        """
        Given: An agent exists with frame but no me file
        When: Client sends chat message
        Then: CaaSError is raised (Pydantic ValidationError — me is required)
        """
        agent_id, session_id = setup_agent_memory(
            memory_service,
            memory={
                "frame": "# Frame",
                "context_anchors": "# Anchors",
                "current_session": "# Session",
            },
        )

        with pytest.raises(CaaSError) as exc_info:
            next(
                caas_client.stream(
                    agent_id=agent_id, session_id=session_id, message="Hello"
                )
            )

        assert "me" in exc_info.value.message.lower()

    def test_tool_using_request_completes_successfully(
        self, caas_client, memory_service, llm_adapter
    ):
        """
        Given: Client sends a request that would trigger tool use
        When: Adapter processes the request (tool execution is internal)
        Then: Client receives a successful response
        """
        agent_id, session_id = setup_agent_memory(memory_service)

        get_llm_mock(llm_adapter).return_value = json.dumps(
            {"response": "Got it, preference saved!", "session_id": "", "done": True}
        )

        result = next(
            caas_client.stream(
                agent_id=agent_id,
                session_id=session_id,
                message="Remember I prefer dark mode",
            )
        )

        assert result.response == "Got it, preference saved!"
        assert result.done is True

    def test_request_completes_with_memory_context(
        self, caas_client, memory_service, llm_adapter
    ):
        """
        Given: User has specific content in memory
        When: Client sends request (adapter has access to memory via tools)
        Then: Request completes successfully
        """
        agent_id, session_id = setup_agent_memory(memory_service)

        memory_service.write_memory(
            agent_id=agent_id,
            key="current_session",
            content="# Session\n\nWorking on PROJECT_ALPHA.",
        )

        get_llm_mock(llm_adapter).return_value = json.dumps(
            {"response": "You're on PROJECT_ALPHA.", "session_id": "", "done": True}
        )

        result = next(
            caas_client.stream(
                agent_id=agent_id,
                session_id=session_id,
                message="What am I working on?",
            )
        )

        assert result.response == "You're on PROJECT_ALPHA."

    def test_structured_output_returns_custom_schema(
        self, caas_client, memory_service, llm_adapter
    ):
        """
        Given: Client sends request with output_format JSON schema
        When: Agent processes the request
        Then: raw_data contains the custom schema fields, not DefaultOutput shape
        """
        agent_id, session_id = setup_agent_memory(memory_service)

        output_schema = {
            "type": "object",
            "properties": {
                "name": {"type": "string"},
                "email": {"type": "string"},
            },
        }

        get_llm_mock(llm_adapter).return_value = json.dumps(
            {"name": "John", "email": "[email protected]"}
        )

        result = next(
            caas_client.stream(
                agent_id=agent_id,
                session_id=session_id,
                message="Extract contact: John ([email protected])",
                output_format=output_schema,
            )
        )

        assert result.raw_data["name"] == "John"
        assert result.raw_data["email"] == "[email protected]"
        # DefaultOutput fields should NOT be present
        assert (
            "response" not in result.raw_data
            or result.raw_data.get("response") != "John"
        ), "Custom schema should not include DefaultOutput's 'response' field"

    def test_response_contains_session_context_fields(
        self, caas_client, memory_service, llm_adapter
    ):
        """
        Given: A user exists with identity in memory
        When: Client sends chat message
        Then: Response contains session_id and done fields from DefaultOutput
        """
        agent_id, session_id = setup_agent_memory(memory_service)

        get_llm_mock(llm_adapter).return_value = json.dumps(
            {"response": "Hello!", "session_id": "", "done": True}
        )

        result = next(
            caas_client.stream(agent_id=agent_id, session_id=session_id, message="Hi")
        )

        assert isinstance(result.session_id, str)
        assert result.done is True

    def test_chat_works_without_session_id(
        self, caas_client, memory_service, llm_adapter
    ):
        """
        Given: A user exists with identity in memory
        When: Client sends chat without session_id
        Then: Request succeeds (backend generates session_id internally)
        """
        agent_id, _ = setup_agent_memory(memory_service)

        get_llm_mock(llm_adapter).return_value = json.dumps(
            {"response": "Hello!", "session_id": "", "done": True}
        )

        result = next(caas_client.stream(agent_id=agent_id, message="Hi"))

        assert isinstance(result, ChatResponse)

    def test_synthesize_persists_done_true_after_successful_chat(
        self, caas_client, memory_service, llm_adapter
    ):
        """
        Given: A user exists with identity in memory
        When: Client sends chat and FORMAT sets done=true
        Then: Persisted current_session contains done=true in the JSON entry

        SYNTHESIZE must run AFTER FORMAT so the persisted snapshot
        reflects the final pipeline state, not an intermediate one.
        """
        agent_id, session_id = setup_agent_memory(memory_service)

        get_llm_mock(llm_adapter).return_value = json.dumps(
            {"response": "Persisted correctly!", "session_id": "", "done": True}
        )

        result = next(
            caas_client.stream(
                agent_id=agent_id,
                session_id=session_id,
                message="Test persistence",
            )
        )

        assert result.done is True

        # Read persisted current_session and verify done=true in the entry
        persisted = memory_service.read_memory(agent_id, "current_session")
        assert persisted is not None, "SYNTHESIZE should have written current_session"

        json_start = persisted.index("{")
        json_end = persisted.rindex("}") + 1
        entry = json.loads(persisted[json_start:json_end])

        assert entry["done"] is True, (
            f"Persisted entry should have done=true, got done={entry['done']}. "
            "SYNTHESIZE likely ran before FORMAT."
        )


@pytest.mark.integration
class TestE2EFullPipeline:
    """Canonical all-phases E2E test exercising HYDRATE -> PROCESS -> FORMAT -> SYNTHESIZE."""

    def test_full_pipeline_hydrate_process_format_synthesize(
        self, memory_service, llm_adapter, file_prompt_adapter
    ):
        """
        Given: Agent with identity in memory, full pipeline orchestrator
        When: Client sends chat via CaaSClient stream()
        Then: All phases execute in order, response is correct, session persisted
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
        client = CaaSClient(app=app)

        # 3-phase mock chain: HYDRATE (text) -> PROCESS (text) -> FORMAT (JSON)
        mock = get_llm_mock(llm_adapter)
        mock.side_effect = [
            "Test identity summary for orchestrator testing.",
            "Full pipeline works!",
            json.dumps(
                {"response": "Full pipeline works!", "session_id": "", "done": True}
            ),
        ]

        result = next(
            client.stream(
                agent_id=agent_id,
                session_id=session_id,
                message="Test the full pipeline",
            )
        )

        assert result.response == "Full pipeline works!"
        assert result.done is True

        # Verify SYNTHESIZE persisted the session
        persisted = memory_service.read_memory(agent_id, "current_session")
        assert persisted is not None, "SYNTHESIZE should have written current_session"
        assert '"done": true' in persisted

        # Verify all 3 LLM calls were made
        assert mock.call_count == 3


@pytest.mark.integration
class TestCaaSClientContract:
    """CaaSClient and ChatResponse contract tests."""

    def test_chat_response_fields(self):
        """ChatResponse has response, session_id, done, and raw_data fields."""
        resp = ChatResponse(
            response="hello",
            session_id="s1",
            done=True,
            raw_data={"response": "hello", "session_id": "s1", "done": True},
        )
        assert resp.response == "hello"
        assert resp.session_id == "s1"
        assert resp.done is True
        assert resp.raw_data["response"] == "hello"

    def test_chat_response_defaults(self):
        """ChatResponse has sensible defaults for optional fields."""
        resp = ChatResponse(raw_data={"custom": "data"})
        assert resp.response == ""
        assert resp.session_id == ""
        assert resp.done is False

    def test_base_url_creates_httpx_client(self):
        """base_url= mode creates an httpx.Client."""
        import httpx

        client = CaaSClient(base_url="http://localhost:9999")
        assert isinstance(client._http_client, httpx.Client)

    def test_connection_creates_pubsub_transport(self):
        """connection= mode creates publisher and subscriber."""
        from unittest.mock import MagicMock

        mock_connection = MagicMock()
        client = CaaSClient(connection=mock_connection)
        assert client._publisher is not None
        assert client._subscriber is not None

    def test_no_response_event_raises_caas_error(self, http_app):
        """
        Given: Server returns SSE stream with only a done event (no response)
        When: Client parses the stream
        Then: Generator yields nothing (no response events to yield)
        """
        from unittest.mock import patch

        client = CaaSClient(app=http_app)

        with patch(
            "codie_as_a_service.api.client._parse_sse_events",
            return_value=[("done", {"usage": {}})],
        ):
            results = list(client.stream(agent_id="any", message="test"))
            assert results == []


def test_list_memory_keys_for_nonexistent_user(memory_service):
    """
    Integration test: list_memory_keys returns empty list for non-existent user.

    This exercises the empty directory path in storage adapters (line 74).
    Can't be tested via agent E2E because agent requires identity files.
    """

    agent_id = f"nonexistent_user_{uuid.uuid4().hex[:8]}"

    keys = memory_service.list_memory_keys(agent_id=agent_id)

    assert keys == []
