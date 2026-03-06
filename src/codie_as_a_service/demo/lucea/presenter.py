"""NiceGUI presenter adapter for the Lucea triage demo.

Implements the TriagePresenter protocol, rendering transcript events
into NiceGUI scroll-area containers in a split-panel layout.
"""

from __future__ import annotations

from typing import TYPE_CHECKING

from nicegui import ui

if TYPE_CHECKING:
    from nicegui.element import Element

    from codie_as_a_service.demo.lucea.transcript import TranscriptEvent


# ── Speaker display config ──────────────────────────────────────────────────
_SPEAKER_META: dict[str, dict[str, str]] = {
    "nurse": {
        "label": "Nurse",
        "msg_class": "message-nurse",
        "label_class": "label-nurse",
    },
    "nurse-assistant": {
        "label": "Nurse Assistant",
        "msg_class": "message-nurse-assistant",
        "label_class": "label-nurse-assistant",
    },
    "guardian": {
        "label": "Guardian",
        "msg_class": "message-guardian",
        "label_class": "label-guardian",
    },
    "patient-agent": {
        "label": "Patient Agent",
        "msg_class": "message-patient-agent",
        "label_class": "label-patient-agent",
    },
    "guardian-agent": {
        "label": "Guardian Agent",
        "msg_class": "message-guardian-agent",
        "label_class": "label-guardian-agent",
    },
}


class NiceGUITriagePresenter:
    """Renders transcript events into NiceGUI scroll-area containers."""

    def __init__(
        self,
        left_panel: Element,
        right_panel: Element,
        chat_input: Element,
    ) -> None:
        self._left = left_panel
        self._right = right_panel
        self._chat_input = chat_input

    # ── Public API ───────────────────────────────────────────────────────

    def show_event(self, event: TranscriptEvent) -> None:
        """Add a styled message to the correct panel."""
        container = self._left if event.panel == "left" else self._right
        meta = _SPEAKER_META.get(event.speaker, _SPEAKER_META["nurse"])

        if event.event_type == "protocol":
            self._add_protocol_alert(container, event)
        elif event.event_type == "context":
            self._add_context_card(container, event, meta)
        else:
            self._add_message(container, event, meta)

        self._scroll_to_bottom(container)

    def enable_chat(self) -> None:
        """Reveal the chat input after playback finishes."""
        self._chat_input.style(remove="display: none")

    def show_response(self, response: str) -> None:
        """Add an assistant-style response to the left panel."""
        cleaned = self._clean_response(response)
        meta = _SPEAKER_META["nurse-assistant"]
        with self._left:
            ui.label("Nurse Assistant").classes(f"speaker-label {meta['label_class']}")
            ui.html(f"<div class='message-bubble {meta['msg_class']}'>{cleaned}</div>")
        self._scroll_to_bottom(self._left)

    @staticmethod
    def _clean_response(text: str) -> str:
        """Extract display text from a CaaS JSON response and format as HTML."""
        import json
        import re

        # Try to extract "output" field from JSON
        try:
            data = json.loads(text)
            if isinstance(data, dict) and "output" in data:
                text = data["output"]
        except (json.JSONDecodeError, TypeError):
            pass

        # Convert markdown bold **text** to <b>text</b>
        text = re.sub(r'\*\*(.+?)\*\*', r'<b>\1</b>', text)
        # Convert bullet points
        text = re.sub(r'(?m)^[•·]\s*', '• ', text)
        # Convert \n to <br>
        text = text.replace('\n', '<br>')

        return text

    # ── Private helpers ──────────────────────────────────────────────────

    @staticmethod
    def _add_message(
        container: Element,
        event: TranscriptEvent,
        meta: dict[str, str],
    ) -> None:
        with container:
            ui.label(event.timestamp).classes("timestamp")
            ui.label(meta["label"]).classes(f"speaker-label {meta['label_class']}")
            ui.html(
                f"<div class='message-bubble {meta['msg_class']}'>{event.content}</div>"
            )

    @staticmethod
    def _add_context_card(
        container: Element,
        event: TranscriptEvent,
        meta: dict[str, str],
    ) -> None:
        card_accent = event.speaker
        with container:
            ui.label(event.timestamp).classes("timestamp")
            ui.label(meta["label"]).classes(f"speaker-label {meta['label_class']}")
            ui.html(f"<div class='context-card {card_accent}'>{event.content}</div>")

    @staticmethod
    def _add_protocol_alert(
        container: Element,
        event: TranscriptEvent,
    ) -> None:
        with container:
            ui.html(
                f"<div class='protocol-alert'>"
                f"<div class='alert-title'>Protocol Guidance</div>"
                f"{event.content}"
                f"</div>"
            )

    @staticmethod
    def _scroll_to_bottom(container: Element) -> None:
        container.scroll_to(percent=1.0)
