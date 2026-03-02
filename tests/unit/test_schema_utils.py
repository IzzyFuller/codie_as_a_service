"""Unit tests for JSON schema to Pydantic model conversion."""

from pydantic import BaseModel

from codie_as_a_service.core.phase_models import PhaseOutputModel, SessionContext
from codie_as_a_service.services.schema_utils import json_schema_to_model


class TestJsonSchemaToModel:
    """json_schema_to_model converts JSON schema dicts to PhaseOutputModel subclasses."""

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

    def test_result_is_phase_output_model_subclass(self):
        """
        Given: Any valid JSON schema
        When: Converted to a model
        Then: The result is a PhaseOutputModel subclass (participates in pipeline)
        """
        schema = {
            "type": "object",
            "properties": {"x": {"type": "string"}},
        }

        Model = json_schema_to_model(schema)

        assert issubclass(Model, PhaseOutputModel)
        assert issubclass(Model, BaseModel)

    def test_to_session_context_stores_json_in_response(self):
        """
        Given: A dynamic output model instance
        When: to_session_context is called
        Then: context.response contains the model's JSON dump

        This is how the custom output flows back through the pipeline:
        PROCESS produces the custom schema, to_session_context stores it,
        orchestrator parses context.response on return.
        """
        schema = {
            "type": "object",
            "properties": {
                "name": {"type": "string"},
                "score": {"type": "number"},
            },
        }

        Model = json_schema_to_model(schema)
        instance = Model(name="Jane", score=95.5)

        context = SessionContext(
            session_id="s1",
            agent_id="a1",
            instruction="test",
            frame="be direct",
        )
        instance.to_session_context(context)

        assert (
            '"name":"Jane"' in context.response or '"name": "Jane"' in context.response
        )
        assert context.response  # non-empty

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

    def test_supports_number_and_boolean_types(self):
        """
        Given: A schema with number and boolean fields
        When: Converted to a Pydantic model
        Then: Types map correctly (number -> float, boolean -> bool)
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
