"""Local filesystem adapter for memory storage implementing MemoryProtocol."""

from pathlib import Path


class LocalMemoryAdapter:
    """
    Local filesystem-based memory storage adapter.

    Stores agent memory files at: {base_dir}/agents/{agent_id}/{key}.md
    """

    def __init__(self, base_dir: str | Path):
        """
        Initialize local memory adapter.

        Args:
            base_dir: Base directory for memory storage
        """
        self.base_dir = Path(base_dir)

    def _get_agent_dir(self, agent_id: str) -> Path:
        """Get directory path for agent memory files."""
        return self.base_dir / "agents" / agent_id

    def _get_file_path(self, agent_id: str, key: str) -> Path:
        """Get file path for user memory file."""
        return self._get_agent_dir(agent_id) / f"{key}.md"

    def read_file(self, agent_id: str, key: str) -> str | None:
        """
        Read memory file from local filesystem.

        Args:
            agent_id: Agent identifier
            key: Memory key

        Returns:
            File content as string, or None if file doesn't exist
        """
        file_path = self._get_file_path(agent_id, key)

        if not file_path.exists():
            return None

        return file_path.read_text()

    def write_file(self, agent_id: str, key: str, content: str) -> None:
        """
        Write memory file to local filesystem.

        Args:
            agent_id: Agent identifier
            key: Memory key
            content: Content to write
        """
        file_path = self._get_file_path(agent_id, key)
        file_path.parent.mkdir(parents=True, exist_ok=True)
        file_path.write_text(content)

    def list_files(self, agent_id: str) -> list[str]:
        """
        List all memory files for an agent.

        Args:
            agent_id: Agent identifier

        Returns:
            List of memory keys (without .md extension)
        """
        user_dir = self._get_agent_dir(agent_id)

        if not user_dir.exists():
            return []

        return [f.stem for f in user_dir.glob("*.md")]
