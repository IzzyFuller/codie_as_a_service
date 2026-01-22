"""Unit tests for FilePromptAdapter edge cases."""

import tempfile
from pathlib import Path

import pytest

from deep_agent_service.adapters.prompts.file_adapter import FilePromptAdapter


class TestFilePromptAdapter:
    """Unit tests for FilePromptAdapter."""

    def test_raises_value_error_for_nonexistent_directory(self):
        """Adapter raises ValueError if prompts directory doesn't exist."""
        with pytest.raises(ValueError, match="Prompts directory does not exist"):
            FilePromptAdapter(prompts_dir="/nonexistent/path/to/prompts")

    def test_loads_markdown_file_when_txt_not_found(self):
        """Adapter falls back to .md file when .txt doesn't exist."""
        with tempfile.TemporaryDirectory() as temp_dir:
            # Create only a .md file (no .txt)
            md_file = Path(temp_dir) / "test_prompt.md"
            md_file.write_text("# Markdown Prompt\n\nHello World!", encoding="utf-8")

            adapter = FilePromptAdapter(prompts_dir=temp_dir)
            result = adapter.get_prompt("test_prompt")

            assert result == "# Markdown Prompt\n\nHello World!"

    def test_raises_file_not_found_when_neither_txt_nor_md_exists(self):
        """Adapter raises FileNotFoundError when prompt file doesn't exist."""
        with tempfile.TemporaryDirectory() as temp_dir:
            adapter = FilePromptAdapter(prompts_dir=temp_dir)

            with pytest.raises(FileNotFoundError, match="Prompt file not found"):
                adapter.get_prompt("nonexistent_prompt")
