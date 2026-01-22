"""API key authentication adapter."""


class APIKeyAuthAdapter:
    """Authenticates requests using a static API key."""

    def __init__(self, valid_key: str):
        """
        Initialize with the valid API key.

        Args:
            valid_key: The API key that will be accepted as valid
        """
        self._valid_key = valid_key

    def verify(self, credentials: str) -> bool:
        """
        Verify credentials against the stored key.

        Args:
            credentials: The API key to verify

        Returns:
            True if credentials match the valid key, False otherwise
        """
        return credentials == self._valid_key
