"""File-based adapter implementing PromptProtocol."""

from pathlib import Path


class FilePromptAdapter:
    """Adapter for loading prompts from local files."""

    def __init__(self, prompts_dir: str | Path):
        """
        Initialize adapter with a directory path.

        Args:
            prompts_dir: Path to directory containing prompt files (.txt or .md)
        """
        self._prompts_dir = Path(prompts_dir)
        if not self._prompts_dir.is_dir():
            raise ValueError(f"Prompts directory does not exist: {self._prompts_dir}")

    def get_prompt(self, name: str, **variables: str) -> str:
        """
        Get a prompt by name with variable substitution.

        Args:
            name: Prompt identifier (filename without extension)
            **variables: Template variables to inject into the prompt

        Returns:
            Compiled prompt string with variables substituted

        Raises:
            FileNotFoundError: If prompt file doesn't exist
            KeyError: If a variable in the template is missing from variables
        """
        prompt_content = self._load_prompt_file(name)
        return prompt_content.format(**variables)

    def _load_prompt_file(self, name: str) -> str:
        """
        Load a prompt file from disk.

        Tries .txt first, then .md.

        Args:
            name: Prompt identifier (filename without extension)

        Returns:
            File contents as string

        Raises:
            FileNotFoundError: If neither .txt nor .md file exists
        """
        # Try .txt first
        txt_file = self._prompts_dir / f"{name}.txt"
        if txt_file.is_file():
            return txt_file.read_text(encoding="utf-8")

        # Try .md
        md_file = self._prompts_dir / f"{name}.md"
        if md_file.is_file():
            return md_file.read_text(encoding="utf-8")

        # Neither found
        raise FileNotFoundError(
            f"Prompt file not found: {name} (tried {txt_file} and {md_file})"
        )
