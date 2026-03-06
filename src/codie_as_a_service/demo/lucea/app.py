"""Lucea Health — Triage Assistant demo application.

NiceGUI web app that plays back a simulated pediatric triage call in a
split-panel layout (nurse side / caller side), then optionally enables
live chat with a CaaS-backed agent.
"""

from __future__ import annotations

import asyncio
import logging
from pathlib import Path
from typing import TYPE_CHECKING

from nicegui import ui

from codie_as_a_service.demo.lucea.presenter import NiceGUITriagePresenter
from codie_as_a_service.demo.lucea.transcript import TranscriptEvent

if TYPE_CHECKING:
    from codie_as_a_service.api.client import CaaSClient

logger = logging.getLogger(__name__)

_CSS_PATH = Path(__file__).parent / "styles.css"


# ── Playback engine ─────────────────────────────────────────────────────────


async def _play_transcript(
    events: list[TranscriptEvent],
    presenter: NiceGUITriagePresenter,
) -> None:
    """Iterate through transcript events with realistic delays."""
    for event in events:
        await asyncio.sleep(event.delay_ms / 1000.0)
        presenter.show_event(event)
    presenter.enable_chat()


# ── App factory ─────────────────────────────────────────────────────────────


def create_lucea_app(
    events: list[TranscriptEvent] | None = None,
    caas_client: CaaSClient | None = None,
) -> None:
    """Build the Lucea demo page.

    Parameters
    ----------
    events:
        Transcript events to play back.  When *None* the app imports the
        default ``TRIAGE_TRANSCRIPT`` from the transcript module.
    caas_client:
        Optional CaaSClient for live chat after playback.  When *None*
        chat submits show a placeholder response.
    """
    # Load default transcript if none supplied
    if events is None:
        from codie_as_a_service.demo.lucea.transcript import TRIAGE_TRANSCRIPT

        events = TRIAGE_TRANSCRIPT

    ui.dark_mode(True)

    # Inject custom CSS
    ui.add_css(_CSS_PATH.read_text())

    # ── Header ───────────────────────────────────────────────────────
    with ui.header().classes("lucea-header"):
        ui.label("Lucea Health").classes("title")
        ui.label("— Triage Assistant").classes("subtitle")

    # ── Main layout ──────────────────────────────────────────────────
    with ui.row().classes("w-full h-full").style("gap: 0; padding: 8px"):
        # Left panel — Nurse side (55%)
        with ui.column().classes("lucea-panel").style("width: 55%"):
            ui.label("Nurse Side").classes("panel-header nurse-side")
            left_scroll = ui.scroll_area().classes("message-feed").style("flex: 1")

            # Chat input — hidden until playback completes
            with (
                ui.row()
                .classes("chat-input-area w-full items-center")
                .style("display: none") as chat_row
            ):
                chat_input = (
                    ui.input(placeholder="Type a message as the nurse...")
                    .classes("flex-grow")
                    .props("outlined dense dark")
                )
                send_button = (
                    ui.button("Send", color="primary")
                    .props("dense flat")
                    .classes("ml-2")
                )

        # Right panel — Caller side (45%)
        with ui.column().classes("lucea-panel").style("width: 45%"):
            ui.label("Caller Side").classes("panel-header caller-side")
            right_scroll = ui.scroll_area().classes("message-feed").style("flex: 1")

    # ── Presenter ────────────────────────────────────────────────────
    presenter = NiceGUITriagePresenter(
        left_panel=left_scroll,
        right_panel=right_scroll,
        chat_input=chat_row,
    )

    # ── Chat handler ─────────────────────────────────────────────────
    async def _on_chat_submit(_event=None) -> None:
        message = chat_input.value
        if not message or not message.strip():
            return
        chat_input.value = ""

        # Show user message on the left panel as a nurse message
        presenter.show_event(
            TranscriptEvent(
                timestamp="",
                panel="left",
                speaker="nurse",
                content=message,
                event_type="message",
                delay_ms=0,
            )
        )

        # Route through CaaS if available, otherwise show placeholder
        if caas_client is not None:
            try:
                # Run blocking HTTP call in a thread to avoid freezing
                # NiceGUI's async event loop (and dropping the websocket)
                loop = asyncio.get_event_loop()
                responses = await loop.run_in_executor(
                    None,
                    lambda: list(caas_client.stream(
                        agent_id="nurse-sarah",
                        message=message,
                    )),
                )
                for response in responses:
                    if response.done:
                        presenter.show_response(response.response)
            except Exception as exc:
                logger.error("CaaS error: %s", exc)
                presenter.show_response(f"⚠ Error: {exc}")
        else:
            await asyncio.sleep(0.3)
            presenter.show_response(
                "Live chat requires CaaS backend. Set ANTHROPIC_API_KEY and restart."
            )

    chat_input.on("keydown.enter", _on_chat_submit)
    send_button.on_click(_on_chat_submit)

    # ── Kick off playback ────────────────────────────────────────────
    ui.timer(0.5, lambda: _play_transcript(events, presenter), once=True)


# ── Standalone entry point ──────────────────────────────────────────────────


def main() -> None:
    """Launch the Lucea demo as a standalone app (playback only)."""

    @ui.page("/")
    def index():
        create_lucea_app()

    ui.run(
        title="Lucea Health — Triage Assistant",
        port=8090,
        reload=False,
    )


if __name__ == "__main__":
    main()
