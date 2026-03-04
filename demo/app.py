"""Gradio chat demo for deep agent service.

Simple UI to demonstrate streaming chat with the HTTP endpoint.
Not for production deployment - demo purposes only.
"""

import os

import gradio as gr

from codie_as_a_service.api.client import CaaSClient, CaaSError


# Configuration
API_BASE_URL = os.environ.get("DEEP_AGENT_API_URL", "http://localhost:8080")
DEFAULT_AGENT_ID = os.environ.get("DEMO_AGENT_ID", "demo-user")
LLM_TIMEOUT = int(os.environ.get("LLM_TIMEOUT", "300"))

# Single client instance
client = CaaSClient(base_url=API_BASE_URL, timeout=float(LLM_TIMEOUT))


def chat(message: str, _history: list[dict], agent_id: str, _session_state: dict):
    """Send message to agent and stream response."""
    try:
        for response in client.stream(
            agent_id=agent_id or DEFAULT_AGENT_ID,
            message=message,
        ):
            yield response.response
    except CaaSError as e:
        yield f"Error: {e.message}"


def create_demo() -> gr.Blocks:
    """Create Gradio demo interface."""
    with gr.Blocks(title="Deep Agent Demo") as demo:
        gr.Markdown("# Deep Agent Service Demo")
        gr.Markdown("Chat with the deep agent. Responses stream in real-time.")

        session_state = gr.State(value={})

        agent_id_input = gr.Textbox(
            label="Agent ID",
            value=DEFAULT_AGENT_ID,
            placeholder="Enter agent ID",
        )

        gr.ChatInterface(
            fn=chat,
            additional_inputs=[agent_id_input, session_state],
            title=None,
            examples=[
                ["Hello, who are you?"],
                ["What can you help me with?"],
                ["Tell me about yourself."],
            ],
        )

    return demo


def main():
    """Launch the demo."""
    demo = create_demo()
    demo.launch(
        server_name=os.environ.get("DEMO_HOST", "127.0.0.1"),
        server_port=int(os.environ.get("DEMO_PORT", "7860")),
        share=os.environ.get("DEMO_SHARE", "").lower() == "true",
    )


if __name__ == "__main__":
    main()
