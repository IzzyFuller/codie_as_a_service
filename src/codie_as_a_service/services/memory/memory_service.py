"""Memory service for agent memory management."""

from typing import Optional

from codie_as_a_service.core.models import IdentityContext
from codie_as_a_service.core.protocols import MemoryProtocol


class MemoryService:
    """
    Service for managing agent memory operations.

    Provides high-level memory operations backed by storage adapter.
    """

    def __init__(self, storage: MemoryProtocol):
        """
        Initialize memory service.

        Args:
            storage: Storage adapter implementing MemoryProtocol
        """
        self.storage = storage

    def read_memory(self, agent_id: str, key: str) -> Optional[str]:
        """
        Read agent memory.

        Args:
            agent_id: Agent identifier
            key: Memory key (e.g., 'current_session', 'context_anchors')

        Returns:
            Memory content or None if not found
        """
        return self.storage.read_file(agent_id=agent_id, key=key)

    def write_memory(self, agent_id: str, key: str, content: str) -> None:
        """
        Write agent memory.

        Args:
            agent_id: Agent identifier
            key: Memory key
            content: Content to write
        """
        self.storage.write_file(agent_id=agent_id, key=key, content=content)

    def list_memory_keys(self, agent_id: str) -> list[str]:
        """
        List all memory keys for an agent.

        Args:
            agent_id: Agent identifier

        Returns:
            List of memory keys
        """
        return self.storage.list_files(agent_id=agent_id)

    def get_identity_context(
        self, agent_id: str, session_lines: int | None = None
    ) -> IdentityContext:
        """
        Load core identity files for an agent.

        Args:
            agent_id: Agent identifier
            session_lines: If provided, only return the last N lines of current_session

        Returns:
            IdentityContext with current_session, context_anchors, and me
        """
        current_session = (
            self.read_memory(agent_id=agent_id, key="current_session") or ""
        )

        if session_lines is not None and current_session:
            lines = current_session.splitlines()
            current_session = "\n".join(lines[-session_lines:])

        return IdentityContext(
            current_session=current_session,
            context_anchors=self.read_memory(agent_id=agent_id, key="context_anchors")
            or "",
            me=self.read_memory(agent_id=agent_id, key="me") or "",
        )
