"""ReActOrchestrator - Multi-phase orchestration loop."""

import logging

from pydantic import BaseModel

from codie_as_a_service.core.models import Message, ToolDefinition
from codie_as_a_service.core.phase_models import PhaseOutputModel, SessionContext
from codie_as_a_service.core.protocols import LLMProtocol, Phase
from codie_as_a_service.services.memory.memory_service import MemoryService

logger = logging.getLogger(__name__)


# =============================================================================
# Phase Definitions (plain classes — no Pydantic)
# =============================================================================


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


# =============================================================================
# Orchestrator
# =============================================================================


class ReActOrchestrator:
    """
    Multi-phase orchestration loop.

    Runs phases in sequence per iteration, checks for completion,
    loops back when not done. Each phase owns its own execution —
    the orchestrator just calls execute().
    """

    def __init__(
        self,
        memory: MemoryService,
        phases: list[Phase],
        max_outer_iterations: int = 3,
        session_lines: int | None = 50,
    ) -> None:
        self._memory = memory
        self._phases = phases
        self._max_outer_iterations = max_outer_iterations
        self._session_lines = session_lines

    def run(
        self,
        session_id: str,
        agent_id: str,
        instruction: str,
        output_format: type[BaseModel] | None = None,
    ) -> BaseModel:
        """
        Run the full orchestration loop.

        Returns output_format model (defaults to SessionContext) populated
        from the final context state.
        """
        if output_format is None:
            output_format = SessionContext

        identity = self._memory.get_identity_context(
            agent_id=agent_id, session_lines=self._session_lines
        )
        if not identity.me:
            raise ValueError(f"No assistant identity configured for agent '{agent_id}'")

        context = SessionContext(
            session_id=session_id,
            agent_id=agent_id,
            instruction=instruction,
            identity_summary=(
                f"Identity: {identity.me}\n"
                f"Context Anchors: {identity.context_anchors}\n"
                f"Current Session: {identity.current_session}"
            ),
        )

        for iteration in range(self._max_outer_iterations):
            context.iteration = iteration

            for phase in self._phases:
                phase.execute(context)

                if context.done:
                    return output_format.model_validate(context.model_dump())

            # Archive response and reset for next iteration
            if iteration < self._max_outer_iterations - 1:
                context.conversation_history.append(context.response)
                context.done = False
                context.response = ""

        return output_format.model_validate(context.model_dump())
