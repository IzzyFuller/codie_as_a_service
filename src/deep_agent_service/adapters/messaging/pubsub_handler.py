"""Pub/Sub message handler for agent requests.

Implements MessageHandler protocol from synapse.
"""

from synapse.protocols.publisher import PubSubPublisher

from deep_agent_service.core.models import RunAgentRequest, AgentResponse
from deep_agent_service.core.protocols import LLMProtocol, PromptProtocol
from deep_agent_service.services.memory.memory_service import MemoryService
from deep_agent_service.services.agent.react_agent import ReActAgent


class AgentMessageHandler:
    """
    Handles validated RunAgentRequest messages.

    Implements MessageHandler protocol from synapse.
    Processes requests through ReActAgent and publishes AgentResponse.
    """

    def __init__(
        self,
        memory_service: MemoryService,
        llm_adapter: LLMProtocol,
        prompt_adapter: PromptProtocol,
        prompt_names: list[str],
        response_topic_path: str,
        publisher: PubSubPublisher,
    ):
        """
        Initialize message handler.

        Args:
            memory_service: Service for reading/writing user memory
            llm_adapter: OpenAI-compatible LLM adapter
            prompt_adapter: File-based prompt adapter
            prompt_names: List of prompt names to fetch and combine for system prompt
            response_topic_path: Pub/Sub topic path for publishing responses
            publisher: Pub/Sub publisher client
        """
        self.memory_service = memory_service
        self.response_topic_path = response_topic_path
        self.publisher = publisher

        # Initialize agent with adapters
        self.agent = ReActAgent(
            llm=llm_adapter,
            prompts=prompt_adapter,
            memory=memory_service,
            prompt_names=prompt_names,
        )

    def handle(self, request: RunAgentRequest) -> None:
        """
        Process validated RunAgentRequest and publish response.

        Implements MessageHandler protocol from synapse.

        Args:
            request: Validated RunAgentRequest (parsed by MessageConsumer)
        """
        # Process request through agent
        try:
            result = self.agent.process(
                user_id=request.user_id,
                message=request.message,
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
            user_id=request.user_id,
            session_id=request.session_id,
            response_data=response_data,
            status=status,
        )

        # Publish response
        response_json = response.model_dump_json().encode("utf-8")
        future = self.publisher.publish(self.response_topic_path, response_json)
        future.result()  # Wait for publish to complete
