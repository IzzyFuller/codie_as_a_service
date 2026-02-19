"""ReActOrchestrator - Multi-phase orchestration loop."""

import logging

from pydantic import BaseModel

from codie_as_a_service.core.models import Message
from codie_as_a_service.core.phase_models import PhaseDefinition, SessionContext
from codie_as_a_service.core.protocols import LLMProtocol
from codie_as_a_service.services.memory.memory_service import MemoryService

logger = logging.getLogger(__name__)


class ReActOrchestrator:
    """
    Multi-phase orchestration loop.

    Runs phases in sequence per iteration, checks validation for completion,
    loops back when not done. Each phase is a single adapter.call() —
    the adapter handles tool execution internally.
    """

    def __init__(
        self,
        llm: LLMProtocol,
        memory: MemoryService,
        phases: list[PhaseDefinition],
        max_outer_iterations: int = 3,
        session_lines: int | None = 50,
    ) -> None:
        self._llm = llm
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
                self._execute_phase(phase, context)

                if context.done:
                    return output_format.model_validate(context.model_dump())

            # Archive response and reset for next iteration
            if iteration < self._max_outer_iterations - 1:
                context.conversation_history.append(context.response)
                context.done = False
                context.response = ""

        return output_format.model_validate(context.model_dump())

    def _execute_phase(self, phase: PhaseDefinition, context: SessionContext) -> None:
        """Execute a single phase: call LLM, apply result to context."""
        logger.info("Phase %s starting (iteration %d)", phase.name, context.iteration)
        phase.output_schema.model_validate(
            self._llm.call(
                messages=[Message(role="user", content=context.model_dump_json())],
                system_prompt=phase.system_prompt,
                tools=phase.tools,
                output_format={
                    "type": "json_schema",
                    "schema": phase.output_schema.model_json_schema(),
                },
                max_new_tokens=phase.max_new_tokens,
            ).data
        ).to_session_context(context)
