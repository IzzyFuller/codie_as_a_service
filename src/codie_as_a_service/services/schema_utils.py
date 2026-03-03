"""Utilities for converting JSON Schema dicts to Pydantic models at runtime."""

from typing import Any

from pydantic import create_model

from codie_as_a_service.core.phase_models import PhaseOutputModel

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

    Used by FormatPhaseDefinition when client provides output_format.
    FORMAT handles context mutation directly — no to_session_context needed.
    """

    pass


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
