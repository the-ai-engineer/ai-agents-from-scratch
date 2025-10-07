from dataclasses import dataclass, field
from typing import Callable, TypeVar, Generic, List, Awaitable
from abc import ABC
import asyncio
from openai import AsyncOpenAI
from pydantic import BaseModel


# ============================================================================
# Base Framework (Async)
# ============================================================================
class WorkflowState(ABC):
    """Base class for workflow state."""

    should_continue: bool = True


StateT = TypeVar("StateT", bound=WorkflowState)
Step = Callable[[StateT], Awaitable[StateT]]


class Workflow(Generic[StateT]):
    """Compose async steps into a workflow."""

    def __init__(self, *steps: Step[StateT]):
        self.steps = steps

    async def run(self, initial_state: StateT) -> StateT:
        state = initial_state
        for step in self.steps:
            if not state.should_continue:
                break
            state = await step(state)
        return state


# ============================================================================
# LLM Client
# ============================================================================
client = AsyncOpenAI()


async def llm(prompt: str, model: str = "gpt-4o-mini") -> str:
    """Call OpenAI API."""
    response = await client.responses.create(model=model, input=prompt)
    return response.output_text


# ============================================================================
# PATTERN 1: PROMPT CHAINING
# Sequential steps where each builds on the previous output
# Use case: Multi-stage content generation (outline → draft → polish)
# ============================================================================
@dataclass
class ChainState(WorkflowState):
    topic: str = ""
    outline: str = ""
    draft: str = ""
    final: str = ""
    should_continue: bool = True


async def create_outline(state: ChainState) -> ChainState:
    state.outline = await llm(f"Create a detailed outline for: {state.topic}")
    return state


async def write_draft(state: ChainState) -> ChainState:
    state.draft = await llm(f"Write a draft based on this outline:\n{state.outline}")
    return state


async def polish_content(state: ChainState) -> ChainState:
    state.final = await llm(f"Polish and improve this draft:\n{state.draft}")
    return state


# Usage
chain_workflow = Workflow[ChainState](create_outline, write_draft, polish_content)


# ============================================================================
# PATTERN 2: ROUTING
# Classify and route to specialized handlers
# Use case: Customer support (billing vs technical vs general)
# ============================================================================
@dataclass
class RoutingState(WorkflowState):
    query: str = ""
    category: str = ""
    response: str = ""
    should_continue: bool = True


async def classify_query(state: RoutingState) -> RoutingState:
    """Classify the query into a category."""

    class Classification(BaseModel):
        category: str  # "billing", "technical", or "general"

    result = await client.responses.parse(
        model="gpt-4o-mini",
        input=f"Classify into [billing, technical, general]: {state.query}",
        text_format=Classification,
    )

    state.category = result.output_parsed.category
    return state


async def route_to_handler(state: RoutingState) -> RoutingState:
    """Route to specialized handler based on category."""
    prompts = {
        "billing": f"As a billing specialist, help with: {state.query}",
        "technical": f"As a technical expert, help with: {state.query}",
        "general": f"As a customer service rep, help with: {state.query}",
    }

    prompt = prompts.get(state.category, prompts["general"])
    state.response = await llm(prompt)
    return state


# Usage
routing_workflow = Workflow[RoutingState](classify_query, route_to_handler)


# ============================================================================
# PATTERN 3: PARALLELIZATION
# Process multiple independent items concurrently
# Use case: Bulk classification, multi-perspective analysis
# ============================================================================
@dataclass
class ParallelState(WorkflowState):
    items: List[str] = field(default_factory=list)
    results: List[str] = field(default_factory=list)
    should_continue: bool = True


async def parallel_process(state: ParallelState) -> ParallelState:
    """Process all items concurrently."""
    tasks = [llm(f"Analyze: {item}") for item in state.items]
    state.results = await asyncio.gather(*tasks)
    return state


# Usage
parallel_workflow = Workflow[ParallelState](parallel_process)


# ============================================================================
# PATTERN 4: ORCHESTRATOR-WORKERS
# Decompose task dynamically, execute in parallel, synthesize
# Use case: Research, complex multi-step analysis
# ============================================================================
@dataclass
class OrchestratorState(WorkflowState):
    task: str = ""
    subtasks: List[str] = field(default_factory=list)
    results: List[str] = field(default_factory=list)
    synthesis: str = ""
    should_continue: bool = True


async def decompose_task(state: OrchestratorState) -> OrchestratorState:
    """Orchestrator breaks down the task dynamically."""

    class TaskPlan(BaseModel):
        subtasks: List[str]

    plan = await client.responses.parse(
        model="gpt-4o-mini",
        input=f"Break this into 3 specific subtasks: {state.task}",
        text_format=TaskPlan,
    )

    state.subtasks = plan.output_parsed.subtasks
    return state


async def execute_workers(state: OrchestratorState) -> OrchestratorState:
    """Workers execute subtasks in parallel."""
    tasks = [llm(f"Complete this subtask: {subtask}") for subtask in state.subtasks]
    state.results = await asyncio.gather(*tasks)
    return state


async def synthesize_results(state: OrchestratorState) -> OrchestratorState:
    """Orchestrator combines results."""
    combined = "\n".join([f"{i + 1}. {r}" for i, r in enumerate(state.results)])
    state.synthesis = await llm(
        f"Synthesize these results for '{state.task}':\n{combined}"
    )
    return state


# Usage
orchestrator_workflow = Workflow[OrchestratorState](
    decompose_task, execute_workers, synthesize_results
)


# ============================================================================
# PATTERN 5: EVALUATOR-OPTIMIZER
# Generate → Evaluate → Refine loop for quality control
# Use case: Code generation, technical writing
# ============================================================================
@dataclass
class OptimizerState(WorkflowState):
    task: str = ""
    draft: str = ""
    score: int = 0
    feedback: str = ""
    iterations: int = 0
    final: str = ""
    should_continue: bool = True


async def generate_or_refine(state: OptimizerState) -> OptimizerState:
    """Generate initial or refined version."""
    if state.iterations == 0:
        state.draft = await llm(f"Generate: {state.task}")
    else:
        state.draft = await llm(
            f"Improve based on feedback: {state.feedback}\n\nCurrent: {state.draft}"
        )
    return state


async def evaluate_quality(state: OptimizerState) -> OptimizerState:
    """Evaluate and provide feedback."""

    class Evaluation(BaseModel):
        score: int  # 1-10
        feedback: str

    eval_result = await client.responses.parse(
        model="gpt-4o-mini",
        input=f"Score 1-10 and provide feedback:\n{state.draft}",
        text_format=Evaluation,
    )

    state.score = eval_result.output_parsed.score
    state.feedback = eval_result.output_parsed.feedback
    state.iterations += 1
    return state


async def check_quality(state: OptimizerState) -> OptimizerState:
    """Decide if quality is good enough."""
    if state.score >= 8 or state.iterations >= 3:
        state.final = state.draft
        state.should_continue = False
    return state


# Usage (loop via workflow repetition)
async def optimize_loop(state: OptimizerState) -> OptimizerState:
    """Run generate-evaluate loop until quality threshold met."""
    while state.should_continue:
        state = await generate_or_refine(state)
        state = await evaluate_quality(state)
        state = await check_quality(state)
    return state


optimizer_workflow = Workflow[OptimizerState](optimize_loop)


# ============================================================================
# DEMO
# ============================================================================
async def main():
    print("=" * 70)
    print("WORKFLOW PATTERNS DEMO")
    print("=" * 70)

    # 1. Chaining
    print("\n1. PROMPT CHAINING")
    result = await chain_workflow.run(ChainState(topic="AI Agents"))
    print(f"Final: {result.final[:100]}...")

    # 2. Routing
    print("\n2. ROUTING")
    result = await routing_workflow.run(RoutingState(query="My payment failed"))
    print(f"Category: {result.category}")
    print(f"Response: {result.response[:100]}...")

    # 3. Parallelization
    print("\n3. PARALLELIZATION")
    result = await parallel_workflow.run(
        ParallelState(
            items=[
                "Email 1: Meeting at 3pm",
                "Email 2: Invoice attached",
                "Email 3: URGENT alert",
            ]
        )
    )
    print(f"Results: {len(result.results)} items processed")

    # 4. Orchestrator
    print("\n4. ORCHESTRATOR-WORKERS")
    result = await orchestrator_workflow.run(
        OrchestratorState(task="Research AI safety")
    )
    print(f"Subtasks: {result.subtasks}")
    print(f"Synthesis: {result.synthesis[:100]}...")

    # 5. Evaluator-Optimizer
    print("\n5. EVALUATOR-OPTIMIZER")
    result = await optimizer_workflow.run(
        OptimizerState(task="Write Python Fibonacci function")
    )
    print(f"Iterations: {result.iterations}")
    print(f"Final score: {result.score}/10")
    print(f"Final: {result.final[:100]}...")


if __name__ == "__main__":
    asyncio.run(main())
