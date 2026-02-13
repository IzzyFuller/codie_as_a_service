"""GCS adapter for memory storage implementing MemoryProtocol."""

from typing import Optional

from google.cloud import storage
from google.cloud.exceptions import NotFound


class GCSMemoryAdapter:
    """
    GCS-based memory storage adapter.

    Stores agent memory files at: {agent_path_template}/{key}.md
    Default template is "agents/{agent_id}" for backward compatibility.
    """

    def __init__(
        self, bucket: storage.Bucket, agent_path_template: str = "agents/{agent_id}"
    ):
        """
        Initialize GCS memory adapter.

        Args:
            bucket: GCS bucket for memory storage
            agent_path_template: Template for agent path resolution.
                Supports {agent_id} placeholder. Examples:
                - "agents/{agent_id}" (default): agents/{agent_id}/{key}.md
                - "" (empty): {key}.md (flat, bucket root)
                - "{agent_id}": {agent_id}/{key}.md (no agents/ prefix)
        """
        self.bucket = bucket
        self._agent_path_template = agent_path_template

    def _get_blob_path(self, agent_id: str, key: str) -> str:
        """
        Get GCS blob path for agent memory file.

        Args:
            agent_id: Agent identifier
            key: Memory key

        Returns:
            Blob path based on agent_path_template
        """
        prefix = self._agent_path_template.format(agent_id=agent_id)
        return f"{prefix}/{key}.md" if prefix else f"{key}.md"

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
        agent_prefix = self._agent_path_template.format(agent_id=agent_id)
        prefix = f"{agent_prefix}/" if agent_prefix else ""
        blobs = self.bucket.list_blobs(prefix=prefix)

        return [blob.name.removeprefix(prefix).removesuffix(".md") for blob in blobs]
