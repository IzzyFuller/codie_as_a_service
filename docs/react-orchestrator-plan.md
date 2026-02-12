# ReActOrchestrator Implementation Plan

## Status: Ready for Implementation
## Approach: Incremental Refactor, Integration Tests for New Behavior

---

## Overview

Decompose the existing single-agent `ReActAgent` into a 6-phase multi-agent process loop
(`ReActOrchestrator`). Each phase is a separate LLM call (or mini tool-loop) with its own
fresh context window. The outer loop IS the ReAct loop.

This refactor does NOT change externally observable behavior. The HTTP endpoint, SSE events,
pub/sub messages, and request/response shapes all stay identical. The existing ~50
integration tests already cover this contract. If they pass with the new orchestrator
underneath, we have not broken anything.

New integration tests are added ONLY for genuinely new observable behavior: the validation
iteration loop (VALIDATE says not done, orchestrator loops back for another iteration).

Each step is independently committable with all existing integration tests green.

---

## Architecture Reference

```
LOOP (max_iterations):
  HYDRATE    -> 1 LLM call    -> HydratedIdentity
  EXTEND     -> mini tool-loop -> ExtendedInstruction (includes tool_manifest)
  PROCESS    -> mini tool-loop -> ProcessResult
  VALIDATE   -> 1 LLM call    -> ValidationResult {done, justification, feedback}
  SYNTHESIZE -> mini tool-loop -> SynthesisResult (writes to memory = evolved seed)

  if completes_request: return

FORMAT       -> 1 LLM call    -> structured JSON output
```

---

## Testing Philosophy

The existing test philosophy is E2E integration tests with mock-at-boundaries (mock LLM,
real everything else). No unit tests exist in the codebase today. This plan maintains that
philosophy.

- **No `tests/unit/` directory.** No unit tests for Pydantic validation, Protocol
  satisfiability, dict wrappers, or other framework behavior.
- **Acceptance criteria for every step:** All existing integration tests still pass.
- **New integration tests** only for the validation iteration loop, which is genuinely
  new observable behavior not covered by existing tests.

---

## File Map (New and Modified)

### New Files
```
src/codie_as_a_service/core/phase_models.py                    # PhaseDefinition, OrchestrationContext, phase output models
src/codie_as_a_service/services/tools/registry.py              # ToolRegistry
src/codie_as_a_service/services/tools/memory_tool_executor.py  # MemoryToolExecutor
src/codie_as_a_service/services/agent/react_orchestrator.py    # ReActOrchestrator
prompts/orchestrator_hydrate.txt                               # Phase prompt templates
prompts/orchestrator_extend.txt
prompts/orchestrator_process.txt
prompts/orchestrator_validate.txt
prompts/orchestrator_synthesize.txt
```

### Modified Files
```
src/codie_as_a_service/services/agent/react_agent.py           # Refactor as reusable mini-loop engine
src/codie_as_a_service/core/protocols.py                       # Add ToolExecutor protocol
src/codie_as_a_service/main_http.py                            # Wire up orchestrator
src/codie_as_a_service/adapters/messaging/pubsub_handler.py    # Wire up orchestrator
tests/conftest.py                                              # Update fixtures for orchestrator wiring
tests/integration/test_e2e_http_chat.py                        # Add validation loop integration tests
```

### Files NOT Modified
```
src/codie_as_a_service/core/models.py                          # Existing IdentityContext stays here
tests/integration/test_e2e_agent_pubsub.py                     # Existing tests must keep passing as-is
```

### Rules
- All `__init__.py` files are EMPTY (no docstrings, no imports, no `__all__`)
- All models use Pydantic `BaseModel` (no dataclasses, no dict access patterns)

---

## Step 1: Phase Output Models and ToolExecutor Protocol

### What
Create Pydantic models for the data contracts between phases AND add a `ToolExecutor`
protocol to `core/protocols.py`. These are pure data models and a protocol definition
with no behavior, making them safe to build first in a single step.

Note: `IdentityContext` already exists in `core/models.py` with fields
`{current_session, context_anchors, me}`. The design document's HYDRATE phase produces a
*different* `IdentityContext` with fields `{summary, key_patterns, session_state}`. To
avoid collision, the HYDRATE output model will be named `HydratedIdentity`.

### Files
- CREATE `src/codie_as_a_service/core/phase_models.py`
- MODIFY `src/codie_as_a_service/core/protocols.py`

### Models to Define (`core/phase_models.py`)

```python
# Phase output models
class HydratedIdentity(BaseModel):
    summary: str
    key_patterns: list[str]
    session_state: str

class ExtendedInstruction(BaseModel):
    instruction: str
    tool_manifest: list[str]
    rationale: str
    memory_references: list[str]

class ProcessResult(BaseModel):
    output: str
    tools_used: list[str]
    trace: str

class ValidationResult(BaseModel):
    done: bool
    justification: str
    feedback: str

class SynthesisResult(BaseModel):
    writes: list[str]
    summary: str

# Orchestration config
class PhaseDefinition(BaseModel):
    name: str
    system_prompt: str
    tools: list[ToolDefinition] = []
    output_schema: type[BaseModel]
    max_iterations: int = 1
    completes_request: bool = False

    model_config = ConfigDict(arbitrary_types_allowed=True)

# Orchestration state
class OrchestrationContext(BaseModel):
    agent_id: str
    instruction: str
    iteration: int = 0
    hydrate: HydratedIdentity | None = None
    extend: ExtendedInstruction | None = None
    process: ProcessResult | None = None
    validate: ValidationResult | None = None
    synthesize: SynthesisResult | None = None
```

### Protocol to Define (`core/protocols.py`)

```python
@runtime_checkable
class ToolExecutor(Protocol):
    def execute(self, agent_id: str, tool_name: str, tool_input: dict[str, Any]) -> str:
        """
        Execute a tool and return the result as a string.

        Args:
            agent_id: User identifier for scoped operations
            tool_name: Name of the tool to execute
            tool_input: Input arguments for the tool

        Returns:
            Tool result as string
        """
        ...
```

### Acceptance Criteria
- All models are importable from `codie_as_a_service.core.phase_models`
- All models are Pydantic BaseModel subclasses
- `ToolExecutor` is importable from `codie_as_a_service.core.protocols`
- All existing integration tests still pass

### Dependencies
- None

### Commit Message Pattern
`feat: Add phase output models, OrchestrationContext, and ToolExecutor protocol`

---

## Step 2: ToolRegistry and MemoryToolExecutor

### What
Create a registry that catalogs all available tools AND extract the hardcoded tool dispatch
logic from `ReActAgent._execute_single_tool` into a standalone `MemoryToolExecutor` class
implementing the `ToolExecutor` protocol. These are small, self-contained classes that
belong together.

### Files
- CREATE `src/codie_as_a_service/services/tools/registry.py`
- CREATE `src/codie_as_a_service/services/tools/memory_tool_executor.py`

### ToolRegistry

```python
class ToolRegistry:
    def __init__(self) -> None:
        self._tools: dict[str, ToolDefinition] = {}

    def register(self, tool: ToolDefinition) -> None:
        """Register a tool definition."""

    def get(self, name: str) -> ToolDefinition:
        """Get tool by name. Raises KeyError if not found."""

    def get_many(self, names: list[str]) -> list[ToolDefinition]:
        """Get multiple tools by name. Raises KeyError if any not found."""

    def list_all(self) -> list[ToolDefinition]:
        """Return all registered tool definitions."""

    def names(self) -> list[str]:
        """Return all registered tool names."""
```

### MemoryToolExecutor

```python
class MemoryToolExecutor:
    """Executes memory-related tools (read_memory, write_memory, list_memory_keys)."""

    def __init__(self, memory: MemoryService) -> None:
        self._memory = memory

    def execute(self, agent_id: str, tool_name: str, tool_input: dict[str, Any]) -> str:
        """Execute a memory tool. Raises ValueError for unknown tools."""
```

### Acceptance Criteria
- `ToolRegistry` is importable from `codie_as_a_service.services.tools.registry`
- `MemoryToolExecutor` is importable from `codie_as_a_service.services.tools.memory_tool_executor`
- `MemoryToolExecutor` implements the `ToolExecutor` protocol
- All existing integration tests still pass

### Dependencies
- Step 1 (ToolExecutor protocol, ToolDefinition)

### Commit Message Pattern
`feat: Add ToolRegistry and MemoryToolExecutor`

---

## Step 3: Refactor ReActAgent as Reusable Mini-Loop Engine

### What
Refactor `ReActAgent` to use `ToolExecutor` for tool dispatch and extract a public
`run_tool_loop` method that the orchestrator can call for any phase.

The approach:

1. Add `tool_executor` and `tools` parameters to `ReActAgent.__init__`
2. Replace `_execute_single_tool` calls with `self._tool_executor.execute()`
3. Remove `_get_tool_definitions` (tools are now injected)
4. Extract `_react_loop` into a public `run_tool_loop` method that can be called by the
   orchestrator for any phase
5. Keep `process()` working exactly as before
6. Update wiring in `main_http.py`, `pubsub_handler.py`, and `conftest.py`

**All existing integration tests must pass.**

### Files
- MODIFY `src/codie_as_a_service/services/agent/react_agent.py`
- MODIFY `src/codie_as_a_service/main_http.py` (pass `MemoryToolExecutor` + tools to `ReActAgent`)
- MODIFY `src/codie_as_a_service/adapters/messaging/pubsub_handler.py` (same)
- MODIFY `tests/conftest.py` (update `ReActAgent` construction in fixtures)

### Refactored ReActAgent Signature

```python
class ReActAgent:
    def __init__(
        self,
        llm: LLMProtocol,
        prompts: PromptProtocol,
        memory: MemoryService,
        prompt_names: list[str],
        tool_executor: ToolExecutor,
        tools: list[ToolDefinition],
        max_iterations: int = 10,
        session_lines: int | None = 50,
    ):
```

### New Public Method

```python
def run_tool_loop(
    self,
    system_prompt: str,
    messages: list[Message],
    tools: list[ToolDefinition],
    tool_executor: ToolExecutor,
    agent_id: str,
    max_iterations: int | None = None,
) -> str:
    """
    Execute a tool-calling loop: call LLM, execute tools, repeat until end_turn.

    This is the reusable mini-loop engine used by the orchestrator for
    tool-using phases (EXTEND, PROCESS, SYNTHESIZE).

    Args:
        system_prompt: System prompt for this loop
        messages: Initial messages (at minimum one user message)
        tools: Tool definitions available in this loop
        tool_executor: Executor for handling tool calls
        agent_id: User identifier for scoped tool operations
        max_iterations: Override max iterations (defaults to self._max_iterations)

    Returns:
        Collected text from LLM responses
    """
```

### Acceptance Criteria
- `ReActAgent.run_tool_loop()` is a public method usable by the orchestrator
- `ReActAgent.process()` still works exactly as before
- All existing integration tests pass unchanged
- `_execute_single_tool` and `_get_tool_definitions` are removed from `ReActAgent`

### Risk Mitigation
Run existing integration tests after each sub-change within the step:
1. Modify `ReActAgent.__init__` to accept `tool_executor` and `tools` params
2. Update wiring in `main_http.py`, `pubsub_handler.py`, and `conftest.py`
3. Replace `_execute_single_tool` with `self._tool_executor.execute()`
4. Extract `run_tool_loop` as a public method

### Dependencies
- Step 2 (MemoryToolExecutor, ToolRegistry)

### Commit Message Pattern
`refactor: Make ReActAgent a reusable mini-loop engine with injected ToolExecutor`

---

## Step 4: ReActOrchestrator -- Full Implementation

### What
Create the `ReActOrchestrator` class with the complete outer loop. This includes:

1. Single-phase execution (tooled and tool-free)
2. Full outer loop: iterate through phases, store results in context, check
   `completes_request`, and loop back when validation says not done
3. FORMAT phase that runs after the outer loop exits

Phase execution strategy:
- If `phase.tools` is empty: single LLM call with schema enforcement (no tools)
- If `phase.tools` is non-empty: delegate to `ReActAgent.run_tool_loop()`

```
for iteration in range(max_outer_iterations):
    for phase in phases:
        result = execute_phase(phase, context)
        setattr(context, phase.name, result)

        if phase.completes_request and result.done:
            break out of both loops

    # Reset phase outputs for next iteration (except synthesize writes persist via memory)
```

### Files
- CREATE `src/codie_as_a_service/services/agent/react_orchestrator.py`

### Class to Define

```python
class ReActOrchestrator:
    def __init__(
        self,
        react_agent: ReActAgent,
        phases: list[PhaseDefinition],
        max_outer_iterations: int = 3,
    ) -> None:
        self._react_agent = react_agent
        self._phases = phases
        self._max_outer_iterations = max_outer_iterations

    def run(
        self,
        agent_id: str,
        instruction: str,
        tool_executor: ToolExecutor,
        output_format: dict[str, Any] | None = None,
    ) -> dict[str, Any]:
        """
        Run the full orchestration loop then format output.

        Returns structured dict matching output_format schema.
        """

    def _execute_phase(
        self,
        phase: PhaseDefinition,
        context: OrchestrationContext,
        tool_executor: ToolExecutor | None = None,
    ) -> BaseModel:
        """Execute a single phase and return its typed output."""

    def _build_phase_input(
        self,
        phase: PhaseDefinition,
        context: OrchestrationContext,
    ) -> str:
        """Build the user message input for a phase from context."""
```

### Acceptance Criteria
- Full loop executes phases in order
- `completes_request` + `done=True` on ValidationResult exits the loop
- `done=False` causes loop to continue to next iteration
- Max outer iterations is respected
- Context is properly populated after each phase
- After loop exits, FORMAT phase structures the `ProcessResult.output`
- All existing integration tests still pass

### Dependencies
- Step 1 (phase models)
- Step 3 (refactored ReActAgent with `run_tool_loop`)

### Commit Message Pattern
`feat: Add ReActOrchestrator with full outer loop and FORMAT phase`

---

## Step 5: Phase Prompt Templates

### What
Create the actual system prompt templates for each phase. These are the prompt strings
that go into `PhaseDefinition.system_prompt`. They instruct the LLM on what each phase
should do and what output format to produce.

### Files
- CREATE `prompts/orchestrator_hydrate.txt`
- CREATE `prompts/orchestrator_extend.txt`
- CREATE `prompts/orchestrator_process.txt`
- CREATE `prompts/orchestrator_validate.txt`
- CREATE `prompts/orchestrator_synthesize.txt`

### Acceptance Criteria
- All 5 prompt template files exist and are loadable
- Each prompt instructs the LLM to return JSON matching the phase's output schema
- Templates use `{variable}` placeholders consistent with `FilePromptAdapter`
- All existing integration tests still pass

### Dependencies
- Step 1 (phase models define the output schemas referenced in prompts)

### Commit Message Pattern
`feat: Add orchestrator phase prompt templates`

---

## Step 6: Wire Orchestrator and Add Validation Loop Integration Tests

### What
Wire `ReActOrchestrator` into `main_http.py` and `pubsub_handler.py` as a replacement for
direct `ReActAgent.process()` calls. Then add integration tests covering the validation
iteration loop -- the only genuinely new observable behavior.

### Files
- MODIFY `src/codie_as_a_service/main_http.py`
- MODIFY `src/codie_as_a_service/adapters/messaging/pubsub_handler.py`
- MODIFY `tests/conftest.py` (update fixtures for orchestrator wiring)
- MODIFY `tests/integration/test_e2e_http_chat.py` (add new integration tests)

### Approach
- Replace `ReActAgent.process()` calls with `ReActOrchestrator.run()` directly
- No feature flags, no forks -- one path, the new path
- Both HTTP and Pub/Sub entry points use the orchestrator
- All existing integration tests must continue to pass
- Add new integration tests for the validation iteration loop

### New Integration Tests (in `tests/integration/test_e2e_http_chat.py`)

```
test_orchestrator_validation_loop_iterates_when_not_done
    # VALIDATE returns done=False with feedback on first iteration
    # VALIDATE returns done=True on second iteration
    # Verify: request completes successfully, response is well-formed
    # This exercises the genuinely new behavior: the outer loop retry

test_orchestrator_stops_at_max_outer_iterations
    # VALIDATE always returns done=False
    # Verify: request completes (doesn't hang), response is returned
    # This exercises the safety bound on the outer loop
```

These tests use the same `TestApp` / `LLMResponseSpec` pattern as existing integration
tests. They mock the LLM at the boundary and exercise the full stack through the HTTP
endpoint.

### Acceptance Criteria
- All requests route through `ReActOrchestrator.run()`
- No feature flags or conditional paths
- All existing integration tests pass unchanged
- New integration tests for validation loop pass
- Both HTTP and Pub/Sub entry points use the orchestrator

### Dependencies
- Step 4 (complete orchestrator)
- Step 5 (prompt templates)

### Commit Message Pattern
`feat: Wire ReActOrchestrator into entry points and add validation loop tests`

---

## Dependency Graph

```
Step 1: Phase models + ToolExecutor protocol
  |
  +---> Step 2: ToolRegistry + MemoryToolExecutor
  |       |
  |       +---> Step 3: Refactor ReActAgent
  |               |
  |               +---> Step 4: ReActOrchestrator (full implementation)
  |                       |
  |                       +---> Step 6: Wire entry points + integration tests
  |                               ^
  +---> Step 5: Phase prompts -----+
```

### Parallelizable Work
- Step 5 can be done in parallel with Steps 2-4 (only needs Step 1)

---

## Risk Mitigation

### Risk: Breaking Existing Integration Tests During ReActAgent Refactor (Step 3)

**Mitigation:** Run existing integration tests after each sub-change within the step:

1. Modify `ReActAgent.__init__` to accept `tool_executor` and `tools` params
2. Update wiring in `main_http.py`, `pubsub_handler.py`, and `conftest.py`
3. Replace `_execute_single_tool` with `self._tool_executor.execute()`
4. Extract `run_tool_loop` as a public method

Integration tests verify each sub-change. Greenfield project -- keep moving.

### Risk: LLM Output Parsing for Phase Models

**Mitigation:** Phases without tools use `output_format` schema enforcement (the same
mechanism already proven by `_structure_output`). The LLM adapter already supports this.
Tooled phases return text that we parse; if parsing fails, the phase raises a clear error.

### Risk: OrchestrationContext State Management Between Iterations

**Mitigation:** OrchestrationContext is a Pydantic model with explicit typed fields.
Between iterations, we create a fresh context (preserving only `agent_id`, `instruction`,
and `iteration` counter). Memory persistence between iterations happens via SYNTHESIZE
writing to actual memory storage, which HYDRATE reads on the next iteration. This is the
design's core insight: state transfer via memory, not messages.

---

## Testing Strategy Summary

| Step | What Changes | How It Is Tested |
|------|-------------|-----------------|
| 1 | Phase models, ToolExecutor protocol | Existing integration tests pass (no behavior change) |
| 2 | ToolRegistry, MemoryToolExecutor | Existing integration tests pass (no behavior change) |
| 3 | ReActAgent refactor + wiring | Existing integration tests pass (same behavior, new internals) |
| 4 | ReActOrchestrator implementation | Existing integration tests pass (not wired yet) |
| 5 | Phase prompt templates | Existing integration tests pass (not wired yet) |
| 6 | Wire orchestrator + new tests | Existing integration tests pass + new validation loop tests |

### Coverage
- All new files added to coverage (no exclusions except `__init__.py` which are already excluded)
- Existing coverage maintained (no regressions)
- New integration tests cover the validation iteration loop (genuinely new behavior)

---

## Estimated Effort Per Step

| Step | Size | Description |
|------|------|-------------|
| 1 | S | 6 Pydantic models + 1 protocol |
| 2 | S | ToolRegistry + MemoryToolExecutor |
| 3 | M | ReActAgent refactor + wiring updates across 4 files |
| 4 | L | Full ReActOrchestrator (skeleton + outer loop + FORMAT) |
| 5 | S | 5 prompt template files |
| 6 | L | Wire entry points + update fixtures + 2 new integration tests |
