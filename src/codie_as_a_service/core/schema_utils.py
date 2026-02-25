"""Utilities for converting JSON Schema dicts to Pydantic models at runtime."""

from typing import Any

from pydantic import create_model

from codie_as_a_service.core.phase_models import PhaseOutputModel, SessionContext

# JSON Schema type → Python type mapping
_TYPE_MAP: dict[str, type] = {
    "string": str,
    "integer": int,
    "number": float,
    "boolean": bool,
    "array": list,
    "object": dict,
}


class _DynamicPhaseOutput(PhaseOutputModel):
    """Base for dynamically-created output models.

    Default to_session_context stores the full model as JSON in context.response.
    Callers using the SDK can override this with domain-specific logic.
    """

    def to_session_context(self, context: SessionContext) -> SessionContext:
        context.response = self.model_dump_json()
        context.conversation_history.append(f"PROCESS: {self.model_dump_json()}")
        return context


def json_schema_to_model(schema: dict[str, Any]) -> type[_DynamicPhaseOutput]:
    """Convert a JSON Schema dict to a PhaseOutputModel subclass."""
    fields: dict[str, Any] = {
        name: (_TYPE_MAP.get(prop.get("type", "string"), Any), ...)
        for name, prop in schema.get("properties", {}).items()
    }
    return create_model(
        "DynamicOutput",
        __base__=_DynamicPhaseOutput,
        **fields,
    )
