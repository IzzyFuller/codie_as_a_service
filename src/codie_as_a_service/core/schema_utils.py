"""Utilities for converting JSON Schema dicts to Pydantic models at runtime."""

from typing import Any

from pydantic import BaseModel, create_model

# JSON Schema type → Python type mapping
_TYPE_MAP: dict[str, type] = {
    "string": str,
    "integer": int,
    "number": float,
    "boolean": bool,
    "array": list,
    "object": dict,
}


def json_schema_to_model(schema: dict[str, Any]) -> type[BaseModel]:
    """Convert a JSON Schema dict to a Pydantic BaseModel class."""
    return create_model(
        "DynamicOutput",
        **{
            name: (_TYPE_MAP.get(prop.get("type", "string"), Any), ...)
            for name, prop in schema.get("properties", {}).items()
        },
    )
