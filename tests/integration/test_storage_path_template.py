"""
Tests for configurable agent_path_template in storage adapters.

Verifies that LocalMemoryAdapter and GCSMemoryAdapter can resolve
agent directories using a configurable path template instead of
hardcoded 'agents/{agent_id}'.
"""

from unittest.mock import MagicMock

from codie_as_a_service.adapters.storage.local_adapter import LocalMemoryAdapter
from codie_as_a_service.adapters.storage.gcs_adapter import GCSMemoryAdapter


class TestLocalMemoryAdapterPathTemplate:
    """Tests for LocalMemoryAdapter path template configuration."""

    def test_default_template_preserves_existing_behavior(self, tmp_path):
        """
        Given: LocalMemoryAdapter with no agent_path_template specified
        When: Writing and reading a file
        Then: Files stored at {base_dir}/agents/{agent_id}/{key}.md
        """
        adapter = LocalMemoryAdapter(base_dir=tmp_path)

        adapter.write_file(agent_id="tess", key="me", content="# Tess")
        expected_path = tmp_path / "agents" / "tess" / "me.md"

        assert expected_path.exists()
        assert expected_path.read_text() == "# Tess"
        assert adapter.read_file(agent_id="tess", key="me") == "# Tess"

    def test_empty_template_resolves_to_flat_base_dir(self, tmp_path):
        """
        Given: LocalMemoryAdapter with agent_path_template=""
        When: Writing and reading a file
        Then: Files stored at {base_dir}/{key}.md (flat, no agent subdirectory)
        """
        adapter = LocalMemoryAdapter(base_dir=tmp_path, agent_path_template="")

        adapter.write_file(agent_id="tess", key="me", content="# Tess")
        expected_path = tmp_path / "me.md"

        assert expected_path.exists()
        assert expected_path.read_text() == "# Tess"
        assert adapter.read_file(agent_id="tess", key="me") == "# Tess"

    def test_custom_template_without_agents_prefix(self, tmp_path):
        """
        Given: LocalMemoryAdapter with agent_path_template="{agent_id}"
        When: Writing and reading a file
        Then: Files stored at {base_dir}/{agent_id}/{key}.md (no agents/ prefix)
        """
        adapter = LocalMemoryAdapter(
            base_dir=tmp_path, agent_path_template="{agent_id}"
        )

        adapter.write_file(agent_id="tess", key="me", content="# Tess")
        expected_path = tmp_path / "tess" / "me.md"

        assert expected_path.exists()
        assert expected_path.read_text() == "# Tess"
        assert adapter.read_file(agent_id="tess", key="me") == "# Tess"

    def test_empty_template_list_files_from_flat_dir(self, tmp_path):
        """
        Given: LocalMemoryAdapter with flat base_dir (empty template)
        When: Listing files
        Then: Returns keys from base_dir directly
        """
        adapter = LocalMemoryAdapter(base_dir=tmp_path, agent_path_template="")

        # Write files directly to base_dir
        (tmp_path / "me.md").write_text("# Identity")
        (tmp_path / "context_anchors.md").write_text("# Anchors")

        keys = adapter.list_files(agent_id="ignored")
        assert sorted(keys) == ["context_anchors", "me"]

    def test_empty_template_read_returns_none_for_missing(self, tmp_path):
        """
        Given: LocalMemoryAdapter with flat base_dir (empty template)
        When: Reading a non-existent key
        Then: Returns None
        """
        adapter = LocalMemoryAdapter(base_dir=tmp_path, agent_path_template="")

        assert adapter.read_file(agent_id="any", key="nonexistent") is None


class TestGCSMemoryAdapterPathTemplate:
    """Tests for GCSMemoryAdapter path template configuration."""

    def test_default_template_preserves_existing_behavior(self):
        """
        Given: GCSMemoryAdapter with no agent_path_template specified
        When: Getting blob path
        Then: Path is agents/{agent_id}/{key}.md
        """
        mock_bucket = MagicMock()
        adapter = GCSMemoryAdapter(bucket=mock_bucket)

        blob_path = adapter._get_blob_path(agent_id="tess", key="me")
        assert blob_path == "agents/tess/me.md"

    def test_empty_template_resolves_to_flat_path(self):
        """
        Given: GCSMemoryAdapter with agent_path_template=""
        When: Getting blob path
        Then: Path is {key}.md (flat, no agent subdirectory)
        """
        mock_bucket = MagicMock()
        adapter = GCSMemoryAdapter(bucket=mock_bucket, agent_path_template="")

        blob_path = adapter._get_blob_path(agent_id="tess", key="me")
        assert blob_path == "me.md"

    def test_custom_template_without_agents_prefix(self):
        """
        Given: GCSMemoryAdapter with agent_path_template="{agent_id}"
        When: Getting blob path
        Then: Path is {agent_id}/{key}.md (no agents/ prefix)
        """
        mock_bucket = MagicMock()
        adapter = GCSMemoryAdapter(bucket=mock_bucket, agent_path_template="{agent_id}")

        blob_path = adapter._get_blob_path(agent_id="tess", key="me")
        assert blob_path == "tess/me.md"

    def test_empty_template_list_files_uses_flat_prefix(self):
        """
        Given: GCSMemoryAdapter with flat path (empty template)
        When: Listing files
        Then: Uses empty prefix (lists from bucket root)
        """
        mock_bucket = MagicMock()
        mock_blob1 = MagicMock()
        mock_blob1.name = "me.md"
        mock_blob2 = MagicMock()
        mock_blob2.name = "context_anchors.md"
        mock_bucket.list_blobs.return_value = [mock_blob1, mock_blob2]

        adapter = GCSMemoryAdapter(bucket=mock_bucket, agent_path_template="")

        keys = adapter.list_files(agent_id="ignored")
        assert sorted(keys) == ["context_anchors", "me"]
        mock_bucket.list_blobs.assert_called_once_with(prefix="")
