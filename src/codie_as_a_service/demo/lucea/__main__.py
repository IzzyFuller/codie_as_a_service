"""Lucea Health — Triage Demo entry point.

Connects to a running CaaS server and launches the NiceGUI demo.
Run with: python -m codie_as_a_service.demo.lucea

Requires:
    CAAS_URL: URL of the CaaS HTTP server (default: http://localhost:8080)
"""

from __future__ import annotations

import logging
import os

from dotenv import load_dotenv
from nicegui import ui

from codie_as_a_service.api.client import CaaSClient
from codie_as_a_service.demo.lucea.app import create_lucea_app

load_dotenv()
logging.basicConfig(
    level=os.environ.get("LOG_LEVEL", "INFO").upper(),
    format="%(asctime)s %(levelname)s %(name)s: %(message)s",
)
logger = logging.getLogger(__name__)


def main() -> None:
    """Launch the Lucea triage demo."""
    caas_url = os.environ.get("CAAS_URL", "http://localhost:8080")
    client = CaaSClient(base_url=caas_url)
    logger.info("CaaS client ready — connecting to %s", caas_url)

    @ui.page("/")
    def index():
        create_lucea_app(caas_client=client)

    ui.run(
        title="Lucea Health — Triage Assistant",
        port=int(os.environ.get("LUCEA_PORT", "8090")),
        reload=False,
    )


if __name__ == "__main__":
    main()
