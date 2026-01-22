"""
Acceptance Test: User Memory Lifecycle

ATDD Acceptance Criteria:
1. User can write memory to their isolated storage
2. User can read their own memory back
3. User A cannot read User B's memory (isolation)
4. Memory is persisted in GCS at correct path: gs://bucket/users/{user_id}/
"""

import pytest

from codie_as_a_service.services.memory.memory_service import MemoryService
from codie_as_a_service.adapters.storage.gcs_adapter import GCSMemoryAdapter


@pytest.mark.integration
class TestUserMemoryLifecycle:
    """Test user memory operations using real GCS emulator."""

    def test_user_can_write_and_read_memory(
        self, gcs_bucket, test_user_id, sample_memory_content
    ):
        """
        ACCEPTANCE TEST: User can write memory and read it back.

        Given: A user with ID
        When: User writes to current_session memory
        Then: User can read the same content back
        """
        # Arrange: Create memory service with real GCS adapter
        gcs_adapter = GCSMemoryAdapter(bucket=gcs_bucket)
        memory_service = MemoryService(storage=gcs_adapter)

        # Act: Write memory
        memory_service.write_memory(
            user_id=test_user_id, key="current_session", content=sample_memory_content
        )

        # Assert: Read memory returns same content
        retrieved_content = memory_service.read_memory(
            user_id=test_user_id, key="current_session"
        )

        assert retrieved_content == sample_memory_content

    def test_memory_isolation_between_users(self, gcs_bucket):
        """
        ACCEPTANCE TEST: User A cannot read User B's memory.

        Given: Two different users
        When: User A writes memory
        Then: User B cannot read User A's memory (returns None or empty)
        """
        # Arrange: Create memory service
        gcs_adapter = GCSMemoryAdapter(bucket=gcs_bucket)
        memory_service = MemoryService(storage=gcs_adapter)

        user_a_id = "user_a_123"
        user_b_id = "user_b_456"
        user_a_content = "# User A's Private Memory\n\nThis is secret."

        # Act: User A writes memory
        memory_service.write_memory(
            user_id=user_a_id, key="current_session", content=user_a_content
        )

        # Assert: User B cannot read User A's memory
        user_b_read = memory_service.read_memory(
            user_id=user_b_id, key="current_session"
        )

        # User A's private content must not be present
        assert user_a_content not in (user_b_read or "")

        # Verify User A can still read their own memory
        user_a_read = memory_service.read_memory(
            user_id=user_a_id, key="current_session"
        )
        assert user_a_read == user_a_content

    def test_read_nonexistent_memory_returns_none(self, gcs_bucket, test_user_id):
        """
        ACCEPTANCE TEST: Reading nonexistent memory returns None.

        Given: A user who has not written memory
        When: User attempts to read memory
        Then: Returns None (not an error)
        """
        # Arrange
        gcs_adapter = GCSMemoryAdapter(bucket=gcs_bucket)
        memory_service = MemoryService(storage=gcs_adapter)

        # Act: Read memory that doesn't exist
        result = memory_service.read_memory(user_id=test_user_id, key="nonexistent_key")

        # Assert: Returns None
        assert result is None

    def test_key_filter_returns_specified_memory_keys(self, gcs_bucket, test_user_id):
        """
        ACCEPTANCE TEST: User can have multiple memory files (current_session, context_anchors, etc.)

        Given: A user with ID
        When: User writes to multiple memory keys
        Then: Each key is stored and retrievable independently
        """
        # Arrange
        gcs_adapter = GCSMemoryAdapter(bucket=gcs_bucket)
        memory_service = MemoryService(storage=gcs_adapter)

        session_content = "# Session Memory"
        anchors_content = "# Context Anchors"

        # Act: Write multiple keys
        memory_service.write_memory(
            user_id=test_user_id, key="current_session", content=session_content
        )
        memory_service.write_memory(
            user_id=test_user_id, key="context_anchors", content=anchors_content
        )

        # Assert: Both can be read independently
        assert (
            memory_service.read_memory(user_id=test_user_id, key="current_session")
            == session_content
        )
        assert (
            memory_service.read_memory(user_id=test_user_id, key="context_anchors")
            == anchors_content
        )

    def test_memory_overwrite(self, gcs_bucket, test_user_id):
        """
        ACCEPTANCE TEST: Writing to same key overwrites previous content.

        Given: User has written memory
        When: User writes to same key again
        Then: New content replaces old content
        """
        # Arrange
        gcs_adapter = GCSMemoryAdapter(bucket=gcs_bucket)
        memory_service = MemoryService(storage=gcs_adapter)

        original_content = "# Original"
        new_content = "# Updated"

        # Act: Write, then overwrite
        memory_service.write_memory(
            user_id=test_user_id, key="current_session", content=original_content
        )
        memory_service.write_memory(
            user_id=test_user_id, key="current_session", content=new_content
        )

        # Assert: Latest content is retrieved
        result = memory_service.read_memory(user_id=test_user_id, key="current_session")
        assert result == new_content

    def test_list_memory_keys(self, gcs_bucket, test_user_id):
        """
        ACCEPTANCE TEST: User can list all their memory keys.

        Given: User has multiple memory files
        When: User requests list of keys
        Then: All keys are returned
        """
        # Arrange
        gcs_adapter = GCSMemoryAdapter(bucket=gcs_bucket)
        memory_service = MemoryService(storage=gcs_adapter)

        expected_keys = ["current_session", "context_anchors", "dream_journal"]
        test_content = "# Test Content"

        # Act: Write multiple memory files
        for key in expected_keys:
            memory_service.write_memory(
                user_id=test_user_id, key=key, content=test_content
            )

        # Assert: List returns all keys
        keys = memory_service.list_memory_keys(user_id=test_user_id)
        assert len(keys) == len(expected_keys)
        for expected_key in expected_keys:
            assert expected_key in keys

    def test_get_identity_context(self, gcs_bucket):
        """
        ACCEPTANCE TEST: User can load core identity files.

        Given: User has identity files (current_session, context_anchors, me)
        When: User requests identity context
        Then: IdentityContext with typed fields is returned
        """
        # Arrange: Use unique user ID to avoid test pollution
        unique_user_id = "identity_test_user_789"
        gcs_adapter = GCSMemoryAdapter(bucket=gcs_bucket)
        memory_service = MemoryService(storage=gcs_adapter)

        session_content = "# Session Memory\n\nActive work here."
        anchors_content = "# Context Anchors\n\nPriorities here."
        me_content = "# Identity\n\nWho I am."

        # Act: Write identity files
        memory_service.write_memory(
            user_id=unique_user_id, key="current_session", content=session_content
        )
        memory_service.write_memory(
            user_id=unique_user_id, key="context_anchors", content=anchors_content
        )
        memory_service.write_memory(
            user_id=unique_user_id, key="me", content=me_content
        )

        # Get identity context
        identity = memory_service.get_identity_context(user_id=unique_user_id)

        # Assert: Identity contains expected files
        assert identity.current_session == session_content
        assert identity.context_anchors == anchors_content
        assert identity.me == me_content
