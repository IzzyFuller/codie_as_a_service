"""Langfuse adapter implementing PromptProtocol."""

from langfuse import Langfuse


class LangfusePromptAdapter:
    """Adapter for Langfuse prompt management."""

    def __init__(self, client: Langfuse, label: str = "development"):
        """
        Initialize adapter with Langfuse client.

        Args:
            client: Langfuse client instance
            label: Prompt label to use (development, staging, production)
        """
        self._client = client
        self._label = label

    def get_prompt(self, name: str, **variables: str) -> str:
        """
        Get a prompt by name with variable substitution.

        Args:
            name: Prompt identifier in Langfuse
            **variables: Template variables to inject

        Returns:
            Compiled prompt string with variables substituted
        """
        prompt = self._client.get_prompt(name=name, label=self._label)
        return prompt.compile(**variables)
