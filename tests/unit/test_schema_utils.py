"""Unit tests for JSON schema to Pydantic model conversion."""

from pydantic import BaseModel

from codie_as_a_service.core.schema_utils import json_schema_to_model


class TestJsonSchemaToModel:
    """json_schema_to_model converts JSON schema dicts to Pydantic model classes."""

    def test_simple_schema_creates_working_model(self):
        """
        Given: A JSON schema with string and integer fields
        When: Converted to a Pydantic model
        Then: The model class can be instantiated with those fields
        """
        schema = {
            "type": "object",
            "properties": {
                "name": {"type": "string"},
                "age": {"type": "integer"},
            },
        }

        Model = json_schema_to_model(schema)

        instance = Model(name="Jane", age=30)
        assert instance.name == "Jane"
        assert instance.age == 30

    def test_model_validate_works_on_result(self):
        """
        Given: A model created from JSON schema
        When: model_validate is called with a dict
        Then: It produces a valid instance
        """
        schema = {
            "type": "object",
            "properties": {
                "name": {"type": "string"},
                "age": {"type": "integer"},
            },
        }

        Model = json_schema_to_model(schema)
        instance = Model.model_validate({"name": "Jane", "age": 30})

        assert instance.name == "Jane"
        assert instance.age == 30

    def test_model_validate_ignores_extra_fields(self):
        """
        Given: A model created from JSON schema
        When: model_validate receives extra fields (e.g., from SessionContext.model_dump())
        Then: Extra fields are silently ignored — only schema fields are kept

        This is critical: orchestrator does output_format.model_validate(context.model_dump()),
        and SessionContext has many fields the caller doesn't care about.
        """
        schema = {
            "type": "object",
            "properties": {
                "response": {"type": "string"},
                "done": {"type": "boolean"},
            },
        }

        Model = json_schema_to_model(schema)
        instance = Model.model_validate(
            {
                "response": "Hello",
                "done": True,
                "session_id": "abc-123",
                "agent_id": "test",
                "instruction": "say hello",
                "frame": "be direct",
                "iteration": 0,
            }
        )

        assert instance.response == "Hello"
        assert instance.done is True
        assert not hasattr(instance, "session_id")

    def test_supports_number_and_boolean_types(self):
        """
        Given: A schema with number and boolean fields
        When: Converted to a Pydantic model
        Then: Types map correctly (number → float, boolean → bool)
        """
        schema = {
            "type": "object",
            "properties": {
                "score": {"type": "number"},
                "passed": {"type": "boolean"},
            },
        }

        Model = json_schema_to_model(schema)
        instance = Model(score=95.5, passed=True)

        assert instance.score == 95.5
        assert instance.passed is True

    def test_supports_array_type(self):
        """
        Given: A schema with an array field
        When: Converted to a Pydantic model
        Then: The field accepts lists
        """
        schema = {
            "type": "object",
            "properties": {
                "tags": {"type": "array"},
            },
        }

        Model = json_schema_to_model(schema)
        instance = Model(tags=["a", "b", "c"])

        assert instance.tags == ["a", "b", "c"]

    def test_result_is_base_model_subclass(self):
        """
        Given: Any valid JSON schema
        When: Converted to a Pydantic model
        Then: The result is a subclass of BaseModel
        """
        schema = {
            "type": "object",
            "properties": {"x": {"type": "string"}},
        }

        Model = json_schema_to_model(schema)

        assert issubclass(Model, BaseModel)

    def test_empty_properties_creates_empty_model(self):
        """
        Given: A schema with no properties
        When: Converted to a Pydantic model
        Then: An empty model is created (edge case safety)
        """
        schema = {
            "type": "object",
            "properties": {},
        }

        Model = json_schema_to_model(schema)
        instance = Model()

        assert isinstance(instance, BaseModel)
