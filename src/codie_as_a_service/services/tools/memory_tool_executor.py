"""Memory tool executor implementing ToolExecutor protocol."""

from typing import Any

from codie_as_a_service.services.memory.memory_service import MemoryService


class MemoryToolExecutor:
    """Executes memory-related tools (read_memory, write_memory, list_memory_keys)."""

    def __init__(self, memory: MemoryService) -> None:
        self._memory = memory

    def execute(self, user_id: str, tool_name: str, tool_input: dict[str, Any]) -> str:
        """Execute a memory tool. Raises ValueError for unknown tools."""
        if tool_name == "read_memory":
            key = tool_input.get("key", "")
            content = self._memory.read_memory(user_id=user_id, key=key)
            return content if content else f"No memory found for key: {key}"

        elif tool_name == "write_memory":
            key = tool_input.get("key", "")
            content = tool_input.get("content", "")
            self._memory.write_memory(user_id=user_id, key=key, content=content)
            return f"Successfully wrote to {key}"

        elif tool_name == "list_memory_keys":
            keys = self._memory.list_memory_keys(user_id=user_id)
            return ", ".join(keys) if keys else "No memory keys found"
