"""Pub/Sub message handler for agent requests.

Implements MessageHandler protocol from synapse.
"""

from synapse.protocols.publisher import PubSubPublisher

from codie_as_a_service.core.models import (
    RunAgentRequest,
    AgentResponse,
    ToolDefinition,
)
from codie_as_a_service.core.protocols import LLMProtocol, PromptProtocol, ToolExecutor
from codie_as_a_service.services.memory.memory_service import MemoryService
from codie_as_a_service.services.agent.react_orchestrator import ReActOrchestrator


class AgentMessageHandler:
    """
    Handles validated RunAgentRequest messages.

    Implements MessageHandler protocol from synapse.
    Processes requests through ReActOrchestrator and publishes AgentResponse.
    """

    def __init__(
        self,
        memory_service: MemoryService,
        llm_adapter: LLMProtocol,
        prompt_adapter: PromptProtocol,
        prompt_names: list[str],
        response_topic_path: str,
        publisher: PubSubPublisher,
        tool_executor: ToolExecutor,
        tools: list[ToolDefinition],
        orchestrator: ReActOrchestrator,
    ):
        """
        Initialize message handler.

        Args:
            memory_service: Service for reading/writing agent memory
            llm_adapter: OpenAI-compatible LLM adapter
            prompt_adapter: File-based prompt adapter
            prompt_names: List of prompt names to fetch and combine for system prompt
            response_topic_path: Pub/Sub topic path for publishing responses
            publisher: Pub/Sub publisher client
            tool_executor: Executor for handling tool calls
            tools: Tool definitions available to the agent
            orchestrator: ReActOrchestrator for processing requests
        """
        self.memory_service = memory_service
        self.response_topic_path = response_topic_path
        self.publisher = publisher
        self._orchestrator = orchestrator
        self._tool_executor = tool_executor

    def handle(self, request: RunAgentRequest) -> None:
        """
        Process validated RunAgentRequest and publish response.

        Implements MessageHandler protocol from synapse.

        Args:
            request: Validated RunAgentRequest (parsed by MessageConsumer)
        """
        # Process request through orchestrator
        try:
            result = self._orchestrator.run(
                agent_id=request.agent_id,
                instruction=request.message,
                tool_executor=self._tool_executor,
                output_format=request.output_format,
            )
            response_data = result
            status = "success"

        except ValueError as e:
            # User-facing errors (e.g., no identity configured)
            response_data = {"error": str(e)}
            status = "error"

        # Create response
        response = AgentResponse(
            agent_id=request.agent_id,
            session_id=request.session_id,
            response_data=response_data,
            status=status,
        )

        # Publish response
        response_json = response.model_dump_json().encode("utf-8")
        future = self.publisher.publish(self.response_topic_path, response_json)
        future.result()  # Wait for publish to complete
