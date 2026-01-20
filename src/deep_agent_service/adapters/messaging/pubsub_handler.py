"""Pub/Sub message handler for agent requests.

Implements MessageHandler protocol from fasteroutcomes-pubsub.
"""

from fasteroutcomes_pubsub.protocols.publisher import PubSubPublisher

from deep_agent_service.core.models import RunAgentRequest, AgentResponse
from deep_agent_service.services.memory.memory_service import MemoryService
from deep_agent_service.services.agent.react_agent import ReActAgent
from deep_agent_service.adapters.llm.anthropic_adapter import AnthropicLLMAdapter
from deep_agent_service.adapters.prompts.langfuse_adapter import LangfusePromptAdapter


class AgentMessageHandler:
    """
    Handles validated RunAgentRequest messages.

    Implements MessageHandler protocol from fasteroutcomes-pubsub.
    Processes requests through ReActAgent and publishes AgentResponse.
    """

    def __init__(
        self,
        memory_service: MemoryService,
        anthropic_client,
        langfuse_client,
        model: str,
        prompt_names: list[str],
        prompt_label: str,
        response_topic_path: str,
        publisher: PubSubPublisher,
    ):
        """
        Initialize message handler.

        Args:
            memory_service: Service for reading/writing user memory
            anthropic_client: Anthropic SDK client for LLM calls
            langfuse_client: Langfuse SDK client for prompt loading
            model: Model identifier for LLM calls
            prompt_names: List of prompt names to fetch and combine for system prompt
            prompt_label: Langfuse prompt label (e.g., development, staging, production)
            response_topic_path: Pub/Sub topic path for publishing responses
            publisher: Pub/Sub publisher client
        """
        self.memory_service = memory_service
        self.response_topic_path = response_topic_path
        self.publisher = publisher

        # Initialize agent with adapters
        self.agent = ReActAgent(
            llm=AnthropicLLMAdapter(client=anthropic_client, model=model),
            prompts=LangfusePromptAdapter(client=langfuse_client, label=prompt_label),
            memory=memory_service,
            prompt_names=prompt_names,
        )

    def handle(self, request: RunAgentRequest) -> None:
        """
        Process validated RunAgentRequest and publish response.

        Implements MessageHandler protocol from fasteroutcomes-pubsub.

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
