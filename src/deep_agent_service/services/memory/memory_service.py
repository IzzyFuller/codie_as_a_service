"""Memory service for user memory management."""

from typing import Optional

from deep_agent_service.core.models import IdentityContext
from deep_agent_service.core.protocols import MemoryProtocol


class MemoryService:
    """
    Service for managing user memory operations.

    Provides high-level memory operations backed by storage adapter.
    """

    def __init__(self, storage: MemoryProtocol):
        """
        Initialize memory service.

        Args:
            storage: Storage adapter implementing MemoryProtocol
        """
        self.storage = storage

    def read_memory(self, user_id: str, key: str) -> Optional[str]:
        """
        Read user memory.

        Args:
            user_id: User identifier
            key: Memory key (e.g., 'current_session', 'context_anchors')

        Returns:
            Memory content or None if not found
        """
        return self.storage.read_file(user_id=user_id, key=key)

    def write_memory(self, user_id: str, key: str, content: str) -> None:
        """
        Write user memory.

        Args:
            user_id: User identifier
            key: Memory key
            content: Content to write
        """
        self.storage.write_file(user_id=user_id, key=key, content=content)

    def list_memory_keys(self, user_id: str) -> list[str]:
        """
        List all memory keys for a user.

        Args:
            user_id: User identifier

        Returns:
            List of memory keys
        """
        return self.storage.list_files(user_id=user_id)

    def get_identity_context(self, user_id: str) -> IdentityContext:
        """
        Load core identity files for a user.

        Args:
            user_id: User identifier

        Returns:
            IdentityContext with current_session, context_anchors, and me
        """
        return IdentityContext(
            current_session=self.read_memory(user_id=user_id, key="current_session")
            or "",
            context_anchors=self.read_memory(user_id=user_id, key="context_anchors")
            or "",
            me=self.read_memory(user_id=user_id, key="me") or "",
        )
