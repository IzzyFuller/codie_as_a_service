"""File-based adapter implementing PromptProtocol."""

from pathlib import Path


class FilePromptAdapter:
    """Adapter for loading prompts from local files."""

    def __init__(self, prompts_dir: str | Path):
        """Initialize adapter with a directory path."""
        self._prompts_dir = Path(prompts_dir)

    def get_prompt(self, name: str, **variables: str) -> str:
        """Get a prompt by name with variable substitution."""
        prompt_file = self._prompts_dir / f"{name}.txt"
        prompt_content = prompt_file.read_text(encoding="utf-8")
        return prompt_content.format(**variables)
