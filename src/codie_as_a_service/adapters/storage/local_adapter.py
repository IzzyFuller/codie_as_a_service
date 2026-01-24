"""Local filesystem adapter for memory storage implementing MemoryProtocol."""

from pathlib import Path


class LocalMemoryAdapter:
    """
    Local filesystem-based memory storage adapter.

    Stores user memory files at: {base_dir}/users/{user_id}/{key}.md
    """

    def __init__(self, base_dir: str | Path):
        """
        Initialize local memory adapter.

        Args:
            base_dir: Base directory for memory storage
        """
        self.base_dir = Path(base_dir)

    def _get_user_dir(self, user_id: str) -> Path:
        """Get directory path for user memory files."""
        return self.base_dir / "users" / user_id

    def _get_file_path(self, user_id: str, key: str) -> Path:
        """Get file path for user memory file."""
        return self._get_user_dir(user_id) / f"{key}.md"

    def read_file(self, user_id: str, key: str) -> str | None:
        """
        Read memory file from local filesystem.

        Args:
            user_id: User identifier
            key: Memory key

        Returns:
            File content as string, or None if file doesn't exist
        """
        file_path = self._get_file_path(user_id, key)

        if not file_path.exists():
            return None

        return file_path.read_text()

    def write_file(self, user_id: str, key: str, content: str) -> None:
        """
        Write memory file to local filesystem.

        Args:
            user_id: User identifier
            key: Memory key
            content: Content to write
        """
        file_path = self._get_file_path(user_id, key)
        file_path.parent.mkdir(parents=True, exist_ok=True)
        file_path.write_text(content)

    def list_files(self, user_id: str) -> list[str]:
        """
        List all memory files for a user.

        Args:
            user_id: User identifier

        Returns:
            List of memory keys (without .md extension)
        """
        user_dir = self._get_user_dir(user_id)

        if not user_dir.exists():
            return []

        return [f.stem for f in user_dir.glob("*.md")]
