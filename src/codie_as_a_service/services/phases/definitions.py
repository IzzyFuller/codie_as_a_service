"""Phase definitions for the orchestration pipeline."""

import logging

from codie_as_a_service.core.models import Message, ToolDefinition
from codie_as_a_service.core.phase_models import PhaseOutputModel, SessionContext
from codie_as_a_service.core.protocols import LLMProtocol
from codie_as_a_service.services.memory.memory_service import MemoryService

logger = logging.getLogger(__name__)


class TextLLMPhaseDefinition:
    """Phase backed by an LLM call that returns plain text."""

    def __init__(
        self,
        name: str,
        llm: LLMProtocol,
        system_prompt: str,
        output_schema: type[PhaseOutputModel],
        tools: list[ToolDefinition] = [],
        skip_on_retry: bool = False,
        max_new_tokens: int | None = None,
    ) -> None:
        self.name = name
        self._llm = llm
        self._system_prompt = system_prompt
        self._output_schema = output_schema
        self._tools = tools
        self._skip_on_retry = skip_on_retry
        self._max_new_tokens = max_new_tokens

    def execute(self, context: SessionContext) -> None:
        """Call LLM, wrap text in output schema, apply to context."""
        if self._skip_on_retry and context.iteration > 0:
            logger.info("Phase %s skipped (iteration %d)", self.name, context.iteration)
            return
        logger.info("Phase %s starting (iteration %d)", self.name, context.iteration)
        text_output = self._llm.call(
            messages=[
                Message(
                    role="user",
                    content=f"{context.instruction}{context.response}{context.conversation_history}",
                )
            ],
            system_prompt=f"{context.identity_summary}{self._system_prompt}",
            tools=self._tools,
            max_new_tokens=self._max_new_tokens,
        )
        self._output_schema(text_output=text_output).to_session_context(context)


class FormatPhaseDefinition:
    """Phase that shapes PROCESS text into structured output matching context.output_schema.

    The only phase that uses schema-constrained LLM output. Takes context.response
    (plain text from PROCESS) and produces validated JSON matching the client's
    requested schema (or DefaultOutput).
    """

    def __init__(
        self,
        name: str,
        llm: LLMProtocol,
        system_prompt: str,
    ) -> None:
        self.name = name
        self._llm = llm
        self._system_prompt = system_prompt

    def execute(self, context: SessionContext) -> None:
        """Call LLM with output_schema constraint, write JSON to context.response."""
        logger.info("Phase %s starting (iteration %d)", self.name, context.iteration)
        result = self._llm.call(
            messages=[
                Message(
                    role="user",
                    content=context.response,
                )
            ],
            system_prompt=f"{context.identity_summary}{self._system_prompt}",
            output_model=context.output_schema,
        )
        # result is a validated BaseModel (DefaultOutput or client schema)
        context.response = result.model_dump_json()
        context.done = True
        context.conversation_history.append(f"FORMAT: {context.response}")


class SynthesizePhaseDefinition:
    """Deterministic phase that persists the current interaction to memory.

    No LLM call — reads current_session, appends the interaction entry,
    writes it back.
    """

    def __init__(self, name: str, memory: MemoryService) -> None:
        self.name = name
        self._memory = memory

    def execute(self, context: SessionContext) -> None:
        """Append full SessionContext to session memory."""
        logger.info("Phase %s starting (iteration %d)", self.name, context.iteration)
        context.conversation_history.append(
            '{"phase": "synthesize", "action": "persist_session"}'
        )
        current = (
            self._memory.read_memory(agent_id=context.agent_id, key="current_session")
            or ""
        )
        entry = (
            f"\n\n## Interaction (iteration {context.iteration})\n"
            f"{context.model_dump_json(indent=2)}\n"
        )
        self._memory.write_memory(
            agent_id=context.agent_id,
            key="current_session",
            content=current + entry,
        )
