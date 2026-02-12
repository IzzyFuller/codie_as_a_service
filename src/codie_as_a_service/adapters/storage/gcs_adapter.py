"""GCS adapter for memory storage implementing MemoryProtocol."""

from typing import Optional

from google.cloud import storage
from google.cloud.exceptions import NotFound


class GCSMemoryAdapter:
    """
    GCS-based memory storage adapter.

    Stores agent memory files at: agents/{agent_id}/{key}.md
    """

    def __init__(self, bucket: storage.Bucket):
        """
        Initialize GCS memory adapter.

        Args:
            bucket: GCS bucket for memory storage
        """
        self.bucket = bucket

    def _get_blob_path(self, agent_id: str, key: str) -> str:
        """
        Get GCS blob path for agent memory file.

        Args:
            agent_id: Agent identifier
            key: Memory key

        Returns:
            Blob path: agents/{agent_id}/{key}.md
        """
        return f"agents/{agent_id}/{key}.md"

    def read_file(self, agent_id: str, key: str) -> Optional[str]:
        """
        Read memory file from GCS.

        Args:
            agent_id: Agent identifier
            key: Memory key

        Returns:
            File content as string, or None if file doesn't exist
        """
        blob_path = self._get_blob_path(agent_id, key)
        blob = self.bucket.blob(blob_path)

        try:
            content = blob.download_as_text()
            return content
        except NotFound:
            return None

    def write_file(self, agent_id: str, key: str, content: str) -> None:
        """
        Write memory file to GCS.

        Args:
            agent_id: Agent identifier
            key: Memory key
            content: Content to write
        """
        blob_path = self._get_blob_path(agent_id, key)
        blob = self.bucket.blob(blob_path)
        blob.upload_from_string(content, content_type="text/markdown")

    def list_files(self, agent_id: str) -> list[str]:
        """
        List all memory files for an agent.

        Args:
            agent_id: Agent identifier

        Returns:
            List of memory keys (without .md extension)
        """
        prefix = f"agents/{agent_id}/"
        blobs = self.bucket.list_blobs(prefix=prefix)

        return [blob.name.removeprefix(prefix).removesuffix(".md") for blob in blobs]
