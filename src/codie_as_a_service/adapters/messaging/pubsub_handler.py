"""Pub/Sub message handler for agent requests.

Implements MessageHandler protocol from synapse.
"""

from synapse.protocols.publisher import PubSubPublisher

from codie_as_a_service.core.models import RunAgentRequest, AgentResponse
from codie_as_a_service.core.schema_utils import json_schema_to_model
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
        response_topic_path: str,
        publisher: PubSubPublisher,
        orchestrator: ReActOrchestrator,
    ):
        """
        Initialize message handler.

        Args:
            memory_service: Service for reading/writing agent memory
            response_topic_path: Pub/Sub topic path for publishing responses
            publisher: Pub/Sub publisher client
            orchestrator: ReActOrchestrator for processing requests
        """
        self.memory_service = memory_service
        self.response_topic_path = response_topic_path
        self.publisher = publisher
        self._orchestrator = orchestrator

    def handle(self, request: RunAgentRequest) -> None:
        """
        Process validated RunAgentRequest and publish response.

        Implements MessageHandler protocol from synapse.

        Args:
            request: Validated RunAgentRequest (parsed by MessageConsumer)
        """
        try:
            output_model = (
                json_schema_to_model(request.output_format)
                if request.output_format
                else None
            )
            result = self._orchestrator.run(
                session_id=request.session_id,
                agent_id=request.agent_id,
                instruction=request.message,
                output_format=output_model,
            )
            if request.output_format:
                response_data = result.model_dump()
            else:
                response_data = {
                    "output": result.response,
                    "session_id": result.session_id,
                    "done": result.done,
                }
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
