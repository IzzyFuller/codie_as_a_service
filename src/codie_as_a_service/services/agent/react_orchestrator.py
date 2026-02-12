"""ReActOrchestrator - Multi-phase orchestration loop."""

import json
import logging
from typing import Any

from pydantic import BaseModel

from codie_as_a_service.core.models import (
    ContentBlock,
    Message,
)
from codie_as_a_service.core.phase_models import (
    PhaseDefinition,
    ProcessResult,
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
        output_format: dict[str, Any] | None = None,
    ) -> SessionContext:
        """
        Run the full orchestration loop.

        Returns SessionContext with response, done flag, and all phase outputs.
        """
        # Load identity context for this agent
        identity = self._memory.get_identity_context(
            agent_id=agent_id, session_lines=self._session_lines
        )
        if not identity.me:
            raise ValueError(f"No assistant identity configured for agent '{agent_id}'")

        context = SessionContext(
            session_id=session_id, agent_id=agent_id, instruction=instruction
        )

        for iteration in range(self._max_outer_iterations):
            context.iteration = iteration

            for phase in self._phases:
                result = self._execute_phase(phase, context, tool_executor, identity)
                context.phase_outputs[phase.name] = result.model_dump()

                # Declarative merge from PhaseDefinition fields
                if phase.sets_identity_from:
                    context.identity_summary = getattr(result, phase.sets_identity_from)
                if phase.sets_response_from:
                    context.response = getattr(result, phase.sets_response_from)
                if phase.sets_done_from:
                    context.done = getattr(result, phase.sets_done_from)

                if phase.completes_request and context.done:
                    return context

            # Reset for next iteration
            if iteration < self._max_outer_iterations - 1:
                context.phase_outputs = {}
                context.done = False
                context.response = ""

        return context

    def _execute_phase(
        self,
        phase: PhaseDefinition,
        context: SessionContext,
        tool_executor: ToolExecutor,
        identity: Any = None,
    ) -> BaseModel:
        """Execute a single phase and return its typed output."""
        logger.info("Phase %s starting (iteration %d)", phase.name, context.iteration)
        phase_input = self._build_phase_input(phase, context, identity)
        logger.debug("Phase %s input: %.200s", phase.name, phase_input)

        if phase.tools:
            # Tool-using phase: delegate to mini-loop engine
            messages = [Message(role="user", content=phase_input)]
            text_result = self._react_agent.run_tool_loop(
                system_prompt=phase.system_prompt,
                messages=messages,
                tools=phase.tools,
                tool_executor=tool_executor,
                agent_id=context.agent_id,
                max_iterations=phase.max_iterations,
                max_new_tokens=phase.max_new_tokens,
            )
            try:
                return self._parse_phase_output(text_result, phase.output_schema)
            except (json.JSONDecodeError, ValueError):
                return self._wrap_tool_loop_output(text_result)
        else:
            # Single LLM call phase - use JSON schema to force structured output
            output_format = {
                "type": "json_schema",
                "schema": phase.output_schema.model_json_schema(),
            }
            messages = [Message(role="user", content=phase_input)]
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
            return self._parse_phase_output(text_result, phase.output_schema)

    def _build_phase_input(
        self,
        phase: PhaseDefinition,
        context: SessionContext,
        identity: Any = None,
    ) -> str:
        """Build the user message input for a phase from context."""
        parts = [f"Instruction: {context.instruction}"]
        parts.append(f"Iteration: {context.iteration}")

        # Include identity context for HYDRATE phase
        if phase.name == "hydrate" and identity is not None:
            parts.append(f"Identity: {identity.me}")
            parts.append(f"Context Anchors: {identity.context_anchors}")
            parts.append(f"Current Session: {identity.current_session}")

        # All prior phase outputs (generic)
        for phase_name, output in context.phase_outputs.items():
            for key, value in output.items():
                if isinstance(value, list):
                    parts.append(
                        f"{phase_name}.{key}: {', '.join(str(v) for v in value)}"
                    )
                else:
                    parts.append(f"{phase_name}.{key}: {value}")

        return "\n\n".join(parts)

    def _parse_phase_output(
        self, text: str, output_schema: type[BaseModel]
    ) -> BaseModel:
        """Parse text output into the phase's Pydantic model."""
        text = text.strip()
        logger.debug("Parsing %s output: %.500s", output_schema.__name__, text)
        data = json.loads(text)
        return output_schema(**data)

    def _wrap_tool_loop_output(self, text: str) -> ProcessResult:
        """Wrap raw tool loop text into ProcessResult when JSON parsing fails."""
        return ProcessResult(output=text, tools_used=[], trace="")
