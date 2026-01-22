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
DEFAULT_USER_ID = os.environ.get("DEMO_USER_ID", "demo-user")


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


def chat(message: str, history: list[dict], user_id: str):
    """Send message to agent and stream response."""
    session_id = str(uuid.uuid4())

    with httpx.Client(timeout=120.0) as client:
        with client.stream(
            "POST",
            f"{API_BASE_URL}/chat",
            json={
                "user_id": user_id or DEFAULT_USER_ID,
                "session_id": session_id,
                "message": message,
            },
        ) as response:
            response.raise_for_status()

            full_response = ""
            for event_type, data in parse_sse_events(response):
                if event_type == "response":
                    # Server sends structured response with response_text
                    full_response = data.get("response_text") or data.get(
                        "response", ""
                    )
                    yield full_response
                elif event_type == "error":
                    yield f"Error: {data.get('message', 'Unknown error')}"
                    return


def create_demo() -> gr.Blocks:
    """Create Gradio demo interface."""
    with gr.Blocks(title="Deep Agent Demo") as demo:
        gr.Markdown("# Deep Agent Service Demo")
        gr.Markdown("Chat with the deep agent. Responses stream in real-time.")

        user_id_input = gr.Textbox(
            label="User ID",
            value=DEFAULT_USER_ID,
            placeholder="Enter user ID",
        )

        gr.ChatInterface(
            fn=chat,
            additional_inputs=[user_id_input],
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
