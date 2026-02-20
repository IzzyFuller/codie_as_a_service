"""ReActOrchestrator - Multi-phase orchestration loop."""

import logging

from pydantic import BaseModel

from codie_as_a_service.core.phase_models import SessionContext
from codie_as_a_service.core.protocols import Phase
from codie_as_a_service.services.memory.memory_service import MemoryService

logger = logging.getLogger(__name__)


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
        post_phases: list[Phase] | None = None,
        max_outer_iterations: int = 3,
        session_lines: int | None = None,
    ) -> None:
        self._memory = memory
        self._phases = phases
        self._post_phases = post_phases or []
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
        if not identity.frame:
            raise ValueError(f"No frame configured for agent '{agent_id}'")

        context = SessionContext(
            session_id=session_id,
            agent_id=agent_id,
            instruction=instruction,
            identity_summary=(
                f"Frame: {identity.frame}\n"
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
                    for post_phase in self._post_phases:
                        post_phase.execute(context)
                    return output_format.model_validate(context.model_dump())

            # Archive response and reset for next iteration
            if iteration < self._max_outer_iterations - 1:
                context.conversation_history.append(context.response)
                context.done = False
                context.response = ""

        for post_phase in self._post_phases:
            post_phase.execute(context)
        return output_format.model_validate(context.model_dump())
