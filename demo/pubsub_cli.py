"""CLI demo for pub/sub message-driven entry point.

Simple REPL for sending messages and receiving responses via Pub/Sub.
Demonstrates memory persistence across conversations.

Usage:
    poetry run python demo/pubsub_cli.py
"""

import json
import os
import time
import uuid

from google.cloud import pubsub_v1

from deep_agent_service.core.models import RunAgentRequest, AgentResponse


# Configuration from environment
PROJECT_ID = os.environ.get("PUBSUB_PROJECT_ID", "demo-project")
REQUEST_TOPIC = f"projects/{PROJECT_ID}/topics/agent-requests"
RESPONSE_TOPIC = f"projects/{PROJECT_ID}/topics/agent-responses"
REQUEST_SUBSCRIPTION = f"projects/{PROJECT_ID}/subscriptions/agent-requests-sub"
RESPONSE_SUBSCRIPTION = f"projects/{PROJECT_ID}/subscriptions/agent-responses-sub"


class PubSubDemoClient:
    """Simple client for sending messages and receiving responses."""

    def __init__(self):
        self._publisher = pubsub_v1.PublisherClient()
        self._subscriber = pubsub_v1.SubscriberClient()
        self._user_id = os.environ.get("DEMO_USER_ID", "demo-user")
        self._session_id = str(uuid.uuid4())

    def send_message(
        self, message: str, timeout_seconds: int = 30
    ) -> AgentResponse | None:
        """Send a message and wait for response."""
        request = RunAgentRequest(
            user_id=self._user_id,
            session_id=self._session_id,
            message=message,
        )

        # Publish request
        print(f"  Sending to {REQUEST_TOPIC}...")
        future = self._publisher.publish(
            REQUEST_TOPIC, request.model_dump_json().encode("utf-8")
        )
        future.result()  # Wait for publish
        print("  Message sent. Waiting for response...")

        # Poll for response
        deadline = time.time() + timeout_seconds
        while time.time() < deadline:
            try:
                pull_response = self._subscriber.pull(
                    request={
                        "subscription": RESPONSE_SUBSCRIPTION,
                        "max_messages": 1,
                    },
                    timeout=2,
                )

                if pull_response.received_messages:
                    msg = pull_response.received_messages[0]
                    # Acknowledge the message
                    self._subscriber.acknowledge(
                        request={
                            "subscription": RESPONSE_SUBSCRIPTION,
                            "ack_ids": [msg.ack_id],
                        }
                    )
                    data = json.loads(msg.message.data.decode("utf-8"))
                    return AgentResponse(**data)

            except Exception:
                pass  # Timeout on pull, keep trying

            time.sleep(0.5)

        return None

    @property
    def user_id(self) -> str:
        return self._user_id

    @user_id.setter
    def user_id(self, value: str):
        self._user_id = value


def main():
    """Run the demo REPL."""
    print("=" * 60)
    print("  Deep Agent Pub/Sub Demo")
    print("=" * 60)
    print()
    print("Commands:")
    print("  /user <id>  - Switch user ID")
    print("  /quit       - Exit")
    print("  <message>   - Send message to agent")
    print()

    client = PubSubDemoClient()
    print(f"User ID: {client.user_id}")
    print()

    while True:
        try:
            user_input = input("You: ").strip()

            if not user_input:
                continue

            if user_input.lower() == "/quit":
                print("Goodbye!")
                break

            if user_input.startswith("/user "):
                new_user = user_input[6:].strip()
                if new_user:
                    client.user_id = new_user
                    print(f"  Switched to user: {new_user}")
                continue

            # Send message and get response
            response = client.send_message(user_input)

            if response:
                status_indicator = (
                    "" if response.status == "success" else f" [{response.status}]"
                )
                # Extract response_text from response_data dict
                response_text = response.response_data.get(
                    "response_text", str(response.response_data)
                )
                print(f"\nAgent{status_indicator}: {response_text}\n")
            else:
                print("\n  [Timeout - no response received]\n")

        except KeyboardInterrupt:
            print("\nGoodbye!")
            break
        except Exception as e:
            print(f"\n  [Error: {e}]\n")


if __name__ == "__main__":
    main()
