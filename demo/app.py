"""Gradio chat demo for deep agent service.

Simple UI to demonstrate streaming chat with the HTTP endpoint.
Not for production deployment - demo purposes only.
"""

import json
import os
import uuid

import gradio as gr
import httpx


# Configuration
API_BASE_URL = os.environ.get("DEEP_AGENT_API_URL", "http://localhost:8080")
API_KEY = os.environ.get("API_KEY", "")
DEFAULT_AGENT_ID = os.environ.get("DEMO_AGENT_ID", "demo-user")
LLM_TIMEOUT = int(os.environ.get("LLM_TIMEOUT", "300"))


def parse_sse_events(response: httpx.Response):
    """Parse SSE events from streaming response."""
    current_event = None

    for line in response.iter_lines():
        line = line.strip()
        if not line:
            continue

        if line.startswith("event:"):
            current_event = line[6:].strip()
        elif line.startswith("data:"):
            data_str = line[5:].strip()
            if data_str:
                data = json.loads(data_str)
                yield current_event, data


def chat(message: str, history: list[dict], agent_id: str):
    """Send message to agent and stream response."""
    session_id = str(uuid.uuid4())

    with httpx.Client(timeout=float(LLM_TIMEOUT)) as client:
        headers = {"X-API-Key": API_KEY} if API_KEY else {}
        with client.stream(
            "POST",
            f"{API_BASE_URL}/chat",
            headers=headers,
            json={
                "agent_id": agent_id or DEFAULT_AGENT_ID,
                "session_id": session_id,
                "message": message,
            },
        ) as response:
            response.raise_for_status()

            full_response = ""
            for event_type, data in parse_sse_events(response):
                if event_type == "response":
                    full_response = data.get("output", "")
                    yield full_response
                elif event_type == "error":
                    yield f"Error: {data.get('message', 'Unknown error')}"
                    return


def create_demo() -> gr.Blocks:
    """Create Gradio demo interface."""
    with gr.Blocks(title="Deep Agent Demo") as demo:
        gr.Markdown("# Deep Agent Service Demo")
        gr.Markdown("Chat with the deep agent. Responses stream in real-time.")

        agent_id_input = gr.Textbox(
            label="Agent ID",
            value=DEFAULT_AGENT_ID,
            placeholder="Enter agent ID",
        )

        gr.ChatInterface(
            fn=chat,
            additional_inputs=[agent_id_input],
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
