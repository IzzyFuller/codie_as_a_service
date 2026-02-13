"""ReActOrchestrator - Multi-phase orchestration loop."""

import json
import logging
from pydantic import BaseModel

from codie_as_a_service.core.models import (
    ContentBlock,
    Message,
)
from codie_as_a_service.core.phase_models import (
    PhaseDefinition,
    SessionContext,
)
from codie_as_a_service.core.protocols import LLMProtocol, ToolExecutor
from codie_as_a_service.services.agent.react_agent import ReActAgent
from codie_as_a_service.services.memory.memory_service import MemoryService

logger = logging.getLogger(__name__)


class ReActOrchestrator:
    """
    Multi-phase orchestration loop.

    Runs phases in sequence per iteration, checks validation for completion,
    loops back when not done.
    """

    def __init__(
        self,
        react_agent: ReActAgent,
        llm: LLMProtocol,
        memory: MemoryService,
        phases: list[PhaseDefinition],
        max_outer_iterations: int = 3,
        session_lines: int | None = 50,
    ) -> None:
        self._react_agent = react_agent
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
        tool_executor: ToolExecutor,
        output_format: type[BaseModel] | None = None,
    ) -> BaseModel:
        """
        Run the full orchestration loop.

        Returns output_format model (defaults to SessionContext) populated
        from the final context state.
        """
        if output_format is None:
            output_format = SessionContext

        # Load identity context for this agent
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
                self._execute_phase(phase, context, tool_executor)

                if context.done:
                    return output_format.model_validate(context.model_dump())

            # Archive response and reset for next iteration
            if iteration < self._max_outer_iterations - 1:
                context.conversation_history.append(context.response)
                context.done = False
                context.response = ""

        return output_format.model_validate(context.model_dump())

    def _execute_phase(
        self,
        phase: PhaseDefinition,
        context: SessionContext,
        tool_executor: ToolExecutor,
    ) -> None:
        """Execute a single phase, updating context with results."""
        logger.info("Phase %s starting (iteration %d)", phase.name, context.iteration)
        phase_input = context.model_dump_json()
        messages = [Message(role="user", content=phase_input)]

        # Tool-using phases: run tool loop first to enrich messages
        if phase.tools:
            messages = self._react_agent.run_tool_loop(
                system_prompt=phase.system_prompt,
                messages=messages,
                tools=phase.tools,
                tool_executor=tool_executor,
                agent_id=context.agent_id,
                max_new_tokens=phase.max_new_tokens,
            )

        # Schema-constrained call — always
        output_format = {
            "type": "json_schema",
            "schema": phase.output_schema.model_json_schema(),
        }
        response = self._llm.call(
            messages=messages,
            system_prompt=phase.system_prompt,
            tools=None,
            output_format=output_format,
            max_new_tokens=phase.max_new_tokens,
        )

        text_parts = []
        for block in response.content:
            if isinstance(block, ContentBlock):
                text_parts.append(block.text)

        text_result = " ".join(text_parts)
        logger.info("Phase %s got LLM response: %.200s", phase.name, text_result)
        result = self._parse_phase_output(text_result, phase.output_schema)

        # Phase updates context
        if phase.sets_identity_from:
            context.identity_summary = getattr(result, phase.sets_identity_from)
        if phase.sets_response_from:
            context.response = getattr(result, phase.sets_response_from)
        if phase.sets_done_from:
            context.done = getattr(result, phase.sets_done_from)

    def _parse_phase_output(
        self, text: str, output_schema: type[BaseModel]
    ) -> BaseModel:
        """Parse text output into the phase's Pydantic model."""
        text = text.strip()
        logger.debug("Parsing %s output: %.500s", output_schema.__name__, text)
        data = json.loads(text)
        return output_schema(**data)
