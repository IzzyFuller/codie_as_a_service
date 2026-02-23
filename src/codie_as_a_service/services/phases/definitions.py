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

    Message building: identity goes into system prompt, instruction into
    user message. Each phase's prompt tells the LLM what to pay attention to.
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

    def _build_system_prompt(self, context: SessionContext) -> str:
        """Build system prompt: identity_summary + phase instructions."""
        if context.identity_summary:
            return f"{context.identity_summary}\n\n{self._system_prompt}"
        return self._system_prompt

    def _build_user_message(self, context: SessionContext) -> str:
        """Build user message: instruction + non-empty context fields."""
        parts = [context.instruction]
        if context.response:
            parts.append(f"\n\n## Current Response\n{context.response}")
        if context.conversation_history:
            history = "\n".join(context.conversation_history)
            parts.append(f"\n\n## Conversation History\n{history}")
        return "".join(parts)

    def execute(self, context: SessionContext) -> None:
        """Call LLM, validate response, apply to context."""
        if self._skip_on_retry and context.iteration > 0:
            logger.info("Phase %s skipped (iteration %d)", self.name, context.iteration)
            return
        logger.info("Phase %s starting (iteration %d)", self.name, context.iteration)
        self._output_schema.model_validate(
            self._llm.call(
                messages=[Message(role="user", content=self._build_user_message(context))],
                system_prompt=self._build_system_prompt(context),
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

    Message building: same as LLMPhaseDefinition — identity in system
    prompt, instruction in user message.
    """

    def __init__(
        self,
        name: str,
        llm: LLMProtocol,
        system_prompt: str,
        output_schema: type[PhaseOutputModel],
        skip_on_retry: bool = False,
    ) -> None:
        self.name = name
        self._llm = llm
        self._system_prompt = system_prompt
        self._output_schema = output_schema
        self._skip_on_retry = skip_on_retry

    def _build_system_prompt(self, context: SessionContext) -> str:
        """Build system prompt: identity_summary + phase instructions."""
        if context.identity_summary:
            return f"{context.identity_summary}\n\n{self._system_prompt}"
        return self._system_prompt

    def _build_user_message(self, context: SessionContext) -> str:
        """Build user message: instruction + non-empty context fields."""
        parts = [context.instruction]
        if context.response:
            parts.append(f"\n\n## Current Response\n{context.response}")
        if context.conversation_history:
            history = "\n".join(context.conversation_history)
            parts.append(f"\n\n## Conversation History\n{history}")
        return "".join(parts)

    def execute(self, context: SessionContext) -> None:
        """Call LLM, wrap text in output schema, apply to context."""
        if self._skip_on_retry and context.iteration > 0:
            logger.info("Phase %s skipped (iteration %d)", self.name, context.iteration)
            return
        logger.info("Phase %s starting (iteration %d)", self.name, context.iteration)
        response = self._llm.call(
            messages=[Message(role="user", content=self._build_user_message(context))],
            system_prompt=self._build_system_prompt(context),
        )
        self._output_schema(text_output=response.content[0].text).to_session_context(
            context
        )


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
