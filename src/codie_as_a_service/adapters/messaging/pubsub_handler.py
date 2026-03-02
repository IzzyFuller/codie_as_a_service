"""Pub/Sub message handler for agent requests.

Implements MessageHandler protocol from synapse.
"""

import logging

from synapse.protocols.publisher import PubSubPublisher

from codie_as_a_service.adapters.messaging.models import AgentResponse, RunAgentRequest
from codie_as_a_service.services.agent.react_orchestrator import ReActOrchestrator
from codie_as_a_service.services.memory.memory_service import MemoryService
from codie_as_a_service.services.schema_utils import json_schema_to_model

logger = logging.getLogger(__name__)


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
        self.memory_service = memory_service
        self.response_topic_path = response_topic_path
        self.publisher = publisher
        self._orchestrator = orchestrator

    def handle(self, request: RunAgentRequest) -> None:
        """Process validated RunAgentRequest and publish response."""
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
            response_data = result.model_dump()
            status = "success"

        except Exception as e:
            logger.exception("Agent %s failed: %s", request.agent_id, e)
            response_data = {"error": str(e)}
            status = "error"

        # Create response
        response = AgentResponse(
            agent_id=request.agent_id,
            session_id=request.session_id,
            response_data=response_data,
            status=status,
        )

        # Publish response to agent-specific queue
        response_json = response.model_dump_json().encode("utf-8")
        agent_response_topic = f"{self.response_topic_path}.{request.agent_id}"
        future = self.publisher.publish(agent_response_topic, response_json)
        future.result()  # Wait for publish to complete
