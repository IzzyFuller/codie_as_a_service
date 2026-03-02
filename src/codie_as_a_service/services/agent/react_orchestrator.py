"""ReActOrchestrator - Multi-phase orchestration loop."""

import logging

from pydantic import BaseModel

from codie_as_a_service.core.phase_models import DefaultOutput, SessionContext
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
    ) -> None:
        self._memory = memory
        self._phases = phases
        self._post_phases = post_phases or []
        self._max_outer_iterations = max_outer_iterations

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
        identity = self._memory.get_identity_context(agent_id=agent_id)

        context = SessionContext(
            session_id=session_id,
            agent_id=agent_id,
            instruction=instruction,
            frame=identity.frame,
            identity_summary=(
                f"Identity: {identity.me}\n"
                f"Context Anchors: {identity.context_anchors or ''}\n"
                f"Current Session: {identity.current_session or ''}"
            ),
            output_schema=output_format or DefaultOutput,
        )

        for iteration in range(self._max_outer_iterations):
            context.iteration = iteration

            for phase in self._phases:
                phase.execute(context)

            if context.done:
                break

        for post_phase in self._post_phases:
            post_phase.execute(context)

        return context.output_schema.model_validate_json(context.response)
