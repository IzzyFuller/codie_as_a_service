"""Presenter protocol for the Lucea triage demo.

Defines the port that the app uses to render triage events.
Adapter implementations live in separate modules.
"""

from __future__ import annotations

from typing import TYPE_CHECKING, Protocol

if TYPE_CHECKING:
    from codie_as_a_service.demo.lucea.transcript import TranscriptEvent


class TriagePresenter(Protocol):
    """Minimal interface the app uses to render triage events."""

    def show_event(self, event: TranscriptEvent) -> None: ...

    def enable_chat(self) -> None: ...

    def show_response(self, response: str) -> None: ...
