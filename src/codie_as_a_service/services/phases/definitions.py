"""Phase definitions for the orchestration pipeline."""

import logging

from codie_as_a_service.core.models import Message, ToolDefinition
from codie_as_a_service.core.phase_models import PhaseOutputModel, SessionContext
from codie_as_a_service.core.protocols import LLMProtocol
from codie_as_a_service.services.memory.memory_service import MemoryService

logger = logging.getLogger(__name__)


class LLMPhaseDefinition:
    """Phase backed by an LLM call.

    Owns its LLM adapter, system prompt, tools, and output schema.
    execute() calls the LLM, validates the response against the schema,
    and applies the result to the session context.
    """

    def __init__(
        self,
        name: str,
        llm: LLMProtocol,
        system_prompt: str,
        output_schema: type[PhaseOutputModel],
        tools: list[ToolDefinition] | None = None,
        max_new_tokens: int | None = None,
        skip_on_retry: bool = False,
    ) -> None:
        self.name = name
        self._llm = llm
        self._system_prompt = system_prompt
        self._output_schema = output_schema
        self._tools = tools or []
        self._max_new_tokens = max_new_tokens
        self._skip_on_retry = skip_on_retry

    def execute(self, context: SessionContext) -> None:
        """Call LLM, validate response, apply to context."""
        if self._skip_on_retry and context.iteration > 0:
            logger.info("Phase %s skipped (iteration %d)", self.name, context.iteration)
            return
        logger.info("Phase %s starting (iteration %d)", self.name, context.iteration)
        self._output_schema.model_validate(
            self._llm.call(
                messages=[Message(role="user", content=context.model_dump_json())],
                system_prompt=self._system_prompt,
                tools=self._tools,
                output_format={
                    "type": "json_schema",
                    "schema": self._output_schema.model_json_schema(),
                },
                max_new_tokens=self._max_new_tokens,
            ).data
        ).to_session_context(context)


class TextLLMPhaseDefinition:
    """Phase backed by an LLM call that returns plain text.

    No structured output schema — the LLM returns free-form text which
    is written directly to a named field on the session context.
    Saves the extra formatting turn that schema validation requires.
    """

    def __init__(
        self,
        name: str,
        llm: LLMProtocol,
        system_prompt: str,
        context_field: str,
        skip_on_retry: bool = False,
    ) -> None:
        self.name = name
        self._llm = llm
        self._system_prompt = system_prompt
        self._context_field = context_field
        self._skip_on_retry = skip_on_retry

    def execute(self, context: SessionContext) -> None:
        """Call LLM, write raw text to context field."""
        if self._skip_on_retry and context.iteration > 0:
            logger.info("Phase %s skipped (iteration %d)", self.name, context.iteration)
            return
        logger.info("Phase %s starting (iteration %d)", self.name, context.iteration)
        response = self._llm.call(
            messages=[Message(role="user", content=context.model_dump_json())],
            system_prompt=self._system_prompt,
        )
        setattr(context, self._context_field, response.content[0].text)


class SynthesizePhaseDefinition:
    """Deterministic phase that persists the current interaction to memory.

    No LLM call — reads current_session, appends the interaction entry,
    writes it back.
    """

    def __init__(self, name: str, memory: MemoryService) -> None:
        self.name = name
        self._memory = memory

    def execute(self, context: SessionContext) -> None:
        """Append current interaction to session memory."""
        logger.info("Phase %s starting (iteration %d)", self.name, context.iteration)
        current = (
            self._memory.read_memory(agent_id=context.agent_id, key="current_session")
            or ""
        )
        entry = (
            f"\n\n## Interaction (iteration {context.iteration})\n"
            f"**Session:** {context.session_id}\n"
            f"**Instruction:** {context.instruction}\n"
            f"**Response:** {context.response}\n"
        )
        self._memory.write_memory(
            agent_id=context.agent_id,
            key="current_session",
            content=current + entry,
        )
