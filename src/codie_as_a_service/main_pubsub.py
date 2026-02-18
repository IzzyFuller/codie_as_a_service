"""Pub/Sub entry point for codie_as_a_service.

Uses MessageConsumer from synapse for message handling.
"""

import logging
import os
import signal
import threading

import pika
from dotenv import load_dotenv
from synapse.consumer.message_consumer import MessageConsumer
from synapse.protocols.publisher import PubSubPublisher
from synapse.protocols.subscriber import PubSubSubscriber

from codie_as_a_service.adapters.llm.claude_cli_adapter import ClaudeCliAdapter
from codie_as_a_service.adapters.llm.local_llm_adapter import LocalLLMAdapter
from codie_as_a_service.adapters.messaging.pubsub_handler import AgentMessageHandler
from synapse.adapters.rabbitmq import RabbitMQPublisher, RabbitMQSubscriber
from codie_as_a_service.adapters.prompts.file_adapter import FilePromptAdapter
from codie_as_a_service.adapters.storage.local_adapter import LocalMemoryAdapter
from codie_as_a_service.core.models import RunAgentRequest
from codie_as_a_service.core.protocols import LLMProtocol, MemoryProtocol
from codie_as_a_service.services.agent.react_orchestrator import ReActOrchestrator
from codie_as_a_service.services.memory.memory_service import MemoryService

load_dotenv()
logging.basicConfig(
    level=os.environ.get("LOG_LEVEL", "INFO").upper(),
    format="%(asctime)s %(levelname)s %(name)s: %(message)s",
)
logger = logging.getLogger(__name__)


class AgentApp:
    """
    Main application that listens for requests and processes them.

    Uses MessageConsumer from synapse to handle Pub/Sub messages.
    Subscribes to a queue, processes incoming RunAgentRequest messages
    through the orchestrator, and publishes AgentResponse messages.
    """

    def __init__(
        self,
        memory_service: MemoryProtocol,
        llm_adapter,
        prompt_adapter,
        prompt_names: list[str],
        publisher: PubSubPublisher,
        subscriber: PubSubSubscriber,
        request_subscription_path: str,
        response_topic_path: str,
        orchestrator: ReActOrchestrator,
    ):
        # Create handler that implements MessageHandler protocol
        handler = AgentMessageHandler(
            memory_service=memory_service,
            response_topic_path=response_topic_path,
            publisher=publisher,
            orchestrator=orchestrator,
        )

        # Create consumer from synapse
        self._consumer = MessageConsumer(
            subscription=request_subscription_path,
            handler=handler,
            request_model=RunAgentRequest,
            subscriber=subscriber,
        )

        self._thread: threading.Thread | None = None

    def start(self) -> None:
        """Start listening for messages in a background thread."""
        self._consumer.start()
        self._thread = threading.Thread(target=self._consumer.run, daemon=True)
        self._thread.start()

    def stop(self) -> None:
        """Stop listening for messages."""
        self._consumer.stop()
        if self._thread:
            self._thread.join(timeout=5.0)
            self._thread = None


def create_app(
    memory_service: MemoryProtocol,
    llm_adapter,
    prompt_adapter,
    prompt_names: list[str],
    publisher: PubSubPublisher,
    subscriber: PubSubSubscriber,
    request_subscription_path: str,
    response_topic_path: str,
    orchestrator: ReActOrchestrator,
) -> AgentApp:
    """Factory function to create a configured AgentApp."""
    return AgentApp(
        memory_service=memory_service,
        llm_adapter=llm_adapter,
        prompt_adapter=prompt_adapter,
        prompt_names=prompt_names,
        publisher=publisher,
        subscriber=subscriber,
        request_subscription_path=request_subscription_path,
        response_topic_path=response_topic_path,
        orchestrator=orchestrator,
    )


def main() -> None:
    """Start the message consumer."""
    # Configuration from environment
    llm_adapter_type = os.environ.get("LLM_ADAPTER", "claude_cli")

    prompts_dir = os.environ.get("PROMPTS_DIR")
    if not prompts_dir:
        raise ValueError("PROMPTS_DIR environment variable is required")

    prompt_names_str = os.environ.get("PROMPT_NAMES")
    if not prompt_names_str:
        raise ValueError("PROMPT_NAMES environment variable is required")
    prompt_names = [name.strip() for name in prompt_names_str.split(",")]

    broker_url = os.environ.get("BROKER_URL", "amqp://guest:guest@localhost:5672/")
    request_subscription = os.environ.get("REQUEST_SUBSCRIPTION", "agent.requests")
    response_topic = os.environ.get("RESPONSE_TOPIC", "agent.responses")

    # Initialize messaging
    connection = pika.BlockingConnection(pika.URLParameters(broker_url))
    publisher = RabbitMQPublisher(connection)
    subscriber = RabbitMQSubscriber(connection)

    # Initialize storage adapter
    storage_dir = os.environ.get("STORAGE_DIR")
    if not storage_dir:
        raise ValueError("STORAGE_DIR environment variable is required")
    storage_adapter: MemoryProtocol = LocalMemoryAdapter(base_dir=storage_dir)

    # Initialize LLM adapter based on type
    if llm_adapter_type == "claude_cli":
        llm_adapter: LLMProtocol = ClaudeCliAdapter()
    elif llm_adapter_type == "local":
        model_name = os.environ.get("MODEL_NAME")
        if not model_name:
            raise ValueError("MODEL_NAME required when LLM_ADAPTER=local")
        device = os.environ.get("DEVICE", "mps")
        llm_adapter = LocalLLMAdapter(model_name=model_name, device=device)
    else:
        raise ValueError(f"Unknown LLM_ADAPTER: {llm_adapter_type}")

    prompt_adapter = FilePromptAdapter(prompts_dir=prompts_dir)

    # Build memory service
    memory_service = MemoryService(storage=storage_adapter)

    # Build tool definitions and orchestrator
    from codie_as_a_service.main_http import (
        _get_memory_tool_definitions,
        _build_orchestrator_phases,
    )

    tools = _get_memory_tool_definitions()
    phases = _build_orchestrator_phases(prompt_adapter, tools)
    orchestrator = ReActOrchestrator(
        llm=llm_adapter,
        memory=memory_service,
        phases=phases,
    )

    # Create app
    app = create_app(
        memory_service=memory_service,
        llm_adapter=llm_adapter,
        prompt_adapter=prompt_adapter,
        prompt_names=prompt_names,
        publisher=publisher,
        subscriber=subscriber,
        request_subscription_path=request_subscription,
        response_topic_path=response_topic,
        orchestrator=orchestrator,
    )

    # Handle graceful shutdown
    shutdown_event = threading.Event()

    def signal_handler(signum, _frame):
        print(f"\nReceived signal {signum}, shutting down...")
        shutdown_event.set()

    signal.signal(signal.SIGINT, signal_handler)
    signal.signal(signal.SIGTERM, signal_handler)

    # Start and wait
    print(f"Starting message handler on {request_subscription}...")
    app.start()

    try:
        shutdown_event.wait()
    finally:
        print("Stopping consumer...")
        app.stop()
        print("Shutdown complete.")


if __name__ == "__main__":
    main()
