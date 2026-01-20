"""GCS adapter for memory storage implementing MemoryProtocol."""

from typing import Optional

from google.cloud import storage
from google.cloud.exceptions import NotFound


class GCSMemoryAdapter:
    """
    GCS-based memory storage adapter.

    Stores user memory files at: users/{user_id}/{key}.md
    """

    def __init__(self, bucket: storage.Bucket):
        """
        Initialize GCS memory adapter.

        Args:
            bucket: GCS bucket for memory storage
        """
        self.bucket = bucket

    def _get_blob_path(self, user_id: str, key: str) -> str:
        """
        Get GCS blob path for user memory file.

        Args:
            user_id: User identifier
            key: Memory key

        Returns:
            Blob path: users/{user_id}/{key}.md
        """
        return f"users/{user_id}/{key}.md"

    def read_file(self, user_id: str, key: str) -> Optional[str]:
        """
        Read memory file from GCS.

        Args:
            user_id: User identifier
            key: Memory key

        Returns:
            File content as string, or None if file doesn't exist
        """
        blob_path = self._get_blob_path(user_id, key)
        blob = self.bucket.blob(blob_path)

        try:
            content = blob.download_as_text()
            return content
        except NotFound:
            return None

    def write_file(self, user_id: str, key: str, content: str) -> None:
        """
        Write memory file to GCS.

        Args:
            user_id: User identifier
            key: Memory key
            content: Content to write
        """
        blob_path = self._get_blob_path(user_id, key)
        blob = self.bucket.blob(blob_path)
        blob.upload_from_string(content, content_type="text/markdown")

    def list_files(self, user_id: str) -> list[str]:
        """
        List all memory files for a user.

        Args:
            user_id: User identifier

        Returns:
            List of memory keys (without .md extension)
        """
        prefix = f"users/{user_id}/"
        blobs = self.bucket.list_blobs(prefix=prefix)

        return [blob.name.removeprefix(prefix).removesuffix(".md") for blob in blobs]
