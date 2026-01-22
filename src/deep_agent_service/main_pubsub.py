"""Pub/Sub entry point for deep agent service.

Uses MessageConsumer from synapse for message handling.
"""

import os
import signal
import threading

from synapse.consumer.message_consumer import MessageConsumer
from synapse.protocols.publisher import PubSubPublisher
from synapse.protocols.subscriber import PubSubSubscriber

from deep_agent_service.adapters.messaging.pubsub_handler import AgentMessageHandler
from deep_agent_service.core.models import RunAgentRequest
from deep_agent_service.core.protocols import MemoryProtocol


class AgentApp:
    """
    Main application that listens for requests and processes them.

    Uses MessageConsumer from synapse to handle Pub/Sub messages.
    Subscribes to a Pub/Sub topic, processes incoming RunAgentRequest messages
    through the ReActAgent, and publishes AgentResponse messages.
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
    ):
        # Create handler that implements MessageHandler protocol
        handler = AgentMessageHandler(
            memory_service=memory_service,
            llm_adapter=llm_adapter,
            prompt_adapter=prompt_adapter,
            prompt_names=prompt_names,
            response_topic_path=response_topic_path,
            publisher=publisher,
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
    )


def main() -> None:
    """Start the Pub/Sub message consumer."""
    # Lazy imports for production dependencies
    from google.cloud import pubsub_v1, storage

    from deep_agent_service.adapters.llm.openai_adapter import OpenAILLMAdapter
    from deep_agent_service.adapters.prompts.file_adapter import FilePromptAdapter
    from deep_agent_service.adapters.storage.gcs_adapter import GCSMemoryAdapter
    from deep_agent_service.services.memory.memory_service import MemoryService

    # Configuration from environment
    gcs_bucket_name = os.environ.get("GCS_BUCKET_NAME")
    if not gcs_bucket_name:
        raise ValueError("GCS_BUCKET_NAME environment variable is required")

    request_subscription_path = os.environ.get("PUBSUB_REQUEST_SUBSCRIPTION")
    if not request_subscription_path:
        raise ValueError("PUBSUB_REQUEST_SUBSCRIPTION environment variable is required")

    response_topic_path = os.environ.get("PUBSUB_RESPONSE_TOPIC")
    if not response_topic_path:
        raise ValueError("PUBSUB_RESPONSE_TOPIC environment variable is required")

    local_llm_url = os.environ.get("LOCAL_LLM_URL")
    if not local_llm_url:
        raise ValueError("LOCAL_LLM_URL environment variable is required")

    local_model_name = os.environ.get("LOCAL_MODEL_NAME")
    if not local_model_name:
        raise ValueError("LOCAL_MODEL_NAME environment variable is required")

    prompts_dir = os.environ.get("PROMPTS_DIR")
    if not prompts_dir:
        raise ValueError("PROMPTS_DIR environment variable is required")

    prompt_names_str = os.environ.get("PROMPT_NAMES")
    if not prompt_names_str:
        raise ValueError("PROMPT_NAMES environment variable is required")
    prompt_names = [name.strip() for name in prompt_names_str.split(",")]

    # Initialize GCP clients
    gcs_client = storage.Client()
    bucket = gcs_client.bucket(gcs_bucket_name)

    # GCP clients directly implement protocols via structural subtyping (duck typing)
    publisher = pubsub_v1.PublisherClient()
    subscriber = pubsub_v1.SubscriberClient()

    # Initialize adapters
    llm_adapter = OpenAILLMAdapter(base_url=local_llm_url, model=local_model_name)
    prompt_adapter = FilePromptAdapter(prompts_dir=prompts_dir)

    # Build memory service
    memory_service = MemoryService(storage=GCSMemoryAdapter(bucket=bucket))

    # Create app
    app = create_app(
        memory_service=memory_service,
        llm_adapter=llm_adapter,
        prompt_adapter=prompt_adapter,
        prompt_names=prompt_names,
        publisher=publisher,
        subscriber=subscriber,
        request_subscription_path=request_subscription_path,
        response_topic_path=response_topic_path,
    )

    # Handle graceful shutdown
    shutdown_event = threading.Event()

    def signal_handler(signum, frame):
        print(f"\nReceived signal {signum}, shutting down...")
        shutdown_event.set()

    signal.signal(signal.SIGINT, signal_handler)
    signal.signal(signal.SIGTERM, signal_handler)

    # Start and wait
    print(f"Starting Pub/Sub consumer on {request_subscription_path}...")
    app.start()

    try:
        shutdown_event.wait()
    finally:
        print("Stopping consumer...")
        app.stop()
        print("Shutdown complete.")


if __name__ == "__main__":
    main()
