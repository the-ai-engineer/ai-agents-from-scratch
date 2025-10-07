"""
Lesson 05: Workflow Patterns

Learn to orchestrate multiple LLM calls using five fundamental patterns:
1. Prompt Chaining - Sequential workflows
2. Routing - Conditional branching
3. Parallelization - Concurrent execution
4. Orchestrator-Workers - Dynamic task decomposition
5. Evaluator-Optimizer - Generate-evaluate-refine loops

WORKFLOWS vs AGENTS:
- Workflows: YOU define the flow (chain → route → parallelize)
  - Deterministic execution paths
  - Use WorkflowState to pass intermediate results
  - Example: outline → draft → polish (always in this order)

- Agents: LLM decides the flow dynamically
  - Non-deterministic, adaptive behavior
  - Use ConversationMemory for conversational context
  - Example: agent decides which tools to call based on reasoning

NEW: This lesson introduces a reusable Workflow framework that composes async steps.
"""

import os
import asyncio
import time
from typing import Literal, Optional, Callable, TypeVar, Generic, List, Awaitable
from dataclasses import dataclass, field
from abc import ABC
from openai import OpenAI, AsyncOpenAI
from pydantic import BaseModel, Field
from dotenv import load_dotenv

load_dotenv()

client = OpenAI(api_key=os.getenv("OPENAI_API_KEY"))
async_client = AsyncOpenAI(api_key=os.getenv("OPENAI_API_KEY"))


# ============================================================================
# WORKFLOW FRAMEWORK
# Reusable abstraction for composing async workflow steps
# ============================================================================

class WorkflowState(ABC):
    """Base class for workflow state with control flow."""
    should_continue: bool = True


StateT = TypeVar("StateT", bound=WorkflowState)
Step = Callable[[StateT], Awaitable[StateT]]


class Workflow(Generic[StateT]):
    """
    Compose async steps into a workflow.

    Usage:
        workflow = Workflow(step1, step2, step3)
        result = await workflow.run(initial_state)
    """
    def __init__(self, *steps: Step[StateT]):
        self.steps = steps

    async def run(self, initial_state: StateT) -> StateT:
        state = initial_state
        for step in self.steps:
            if not state.should_continue:
                break
            state = await step(state)
        return state


# Helper function for LLM calls
async def llm(prompt: str, instructions: str = "", model: str = "gpt-4o-mini") -> str:
    """Call OpenAI API with optional instructions."""
    response = await async_client.responses.create(
        model=model,
        instructions=instructions if instructions else None,
        input=prompt,
        temperature=0.7
    )
    return response.output_text


# ============================================================================
# PATTERN 1: PROMPT CHAINING
# Sequential steps where each builds on the previous output
# ============================================================================

@dataclass
class ChainState(WorkflowState):
    """State for prompt chaining pattern."""
    topic: str = ""
    outline: str = ""
    draft: str = ""
    final: str = ""
    should_continue: bool = True


async def create_outline(state: ChainState) -> ChainState:
    """Step 1: Generate outline."""
    print("  → Generating outline...")
    state.outline = await llm(f"Create a 3-section outline for: {state.topic}")
    return state


async def write_draft(state: ChainState) -> ChainState:
    """Step 2: Write draft based on outline."""
    print("  → Writing draft...")
    state.draft = await llm(f"Write first section (150 words):\n{state.outline}")
    return state


async def polish_content(state: ChainState) -> ChainState:
    """Step 3: Polish the draft."""
    print("  → Polishing content...")
    state.final = await llm(f"Edit for clarity:\n{state.draft}")
    return state


async def prompt_chaining():
    """
    Chain multiple LLM calls using the Workflow framework.
    Each step depends on the previous step's output.
    """
    # Create workflow by composing steps
    workflow = Workflow[ChainState](create_outline, write_draft, polish_content)

    # Run workflow
    initial_state = ChainState(topic="AI agents from first principles")
    result = await workflow.run(initial_state)

    print(f"\nOutline:\n{result.outline[:100]}...")
    print(f"\nDraft:\n{result.draft[:100]}...")
    print(f"\nFinal:\n{result.final[:100]}...")

    return result


# ============================================================================
# PATTERN 2: ROUTING
# Classify and route to specialized handlers
# ============================================================================

@dataclass
class RoutingState(WorkflowState):
    """State for routing pattern."""
    query: str = ""
    category: str = ""
    response: str = ""
    should_continue: bool = True


class Classification(BaseModel):
    """Schema for query classification."""
    category: Literal["technical", "billing", "general"]


async def classify_query(state: RoutingState) -> RoutingState:
    """Step 1: Classify the query."""
    print(f"  → Classifying: {state.query[:50]}...")

    result = await async_client.responses.parse(
        model="gpt-4o-mini",
        input=f"Classify as [technical, billing, general]: {state.query}",
        text_format=Classification
    )

    state.category = result.output_parsed.category
    print(f"  → Category: {state.category}")
    return state


async def route_to_handler(state: RoutingState) -> RoutingState:
    """Step 2: Route to specialized handler."""
    instructions_map = {
        "technical": "You are a technical support specialist. Provide step-by-step solutions.",
        "billing": "You are a billing specialist. Be empathetic about payment issues.",
        "general": "You are a friendly customer service rep."
    }

    instructions = instructions_map.get(state.category, instructions_map["general"])
    state.response = await llm(state.query, instructions=instructions)
    return state


async def routing():
    """
    Route queries to specialized handlers based on classification.
    Demonstrates conditional logic in workflows.
    """
    workflow = Workflow[RoutingState](classify_query, route_to_handler)

    queries = [
        "My app crashes when I export data",
        "I was charged twice this month",
        "What are your office hours?"
    ]

    for query in queries:
        print(f"\nQuery: {query}")
        result = await workflow.run(RoutingState(query=query))
        print(f"Response: {result.response[:100]}...\n")


# ============================================================================
# PATTERN 3: PARALLELIZATION
# Process multiple independent items concurrently
# ============================================================================

@dataclass
class ParallelState(WorkflowState):
    """State for parallelization pattern."""
    items: List[str] = field(default_factory=list)
    results: List[str] = field(default_factory=list)
    should_continue: bool = True


async def parallel_process(state: ParallelState) -> ParallelState:
    """Process all items concurrently."""
    print(f"  → Processing {len(state.items)} items in parallel...")

    async def classify_item(item: str) -> str:
        return await llm(
            f"Classify as: spam, urgent, normal, or newsletter. Reply with just the category.\n\n{item}",
            instructions="You are an email classifier."
        )

    # Execute all tasks concurrently
    tasks = [classify_item(item) for item in state.items]
    state.results = await asyncio.gather(*tasks)

    return state


async def parallelization():
    """
    Process multiple items concurrently for speed.
    Compare sequential vs parallel execution.
    """
    emails = [
        "URGENT: Your account will be suspended!",
        "Weekly newsletter: Top 10 AI developments",
        "Meeting reminder: Team standup at 10am",
        "Congratulations! You've won a million dollars!",
        "Your invoice for March is attached"
    ]

    workflow = Workflow[ParallelState](parallel_process)

    print("\nBenchmarking parallel vs sequential...")

    # Parallel execution
    start = time.time()
    result = await workflow.run(ParallelState(items=emails))
    parallel_time = time.time() - start

    print(f"\n  Parallel time: {parallel_time:.2f}s")
    print(f"  Results:")
    for email, category in zip(result.items, result.results):
        print(f"    '{email[:45]}...' → {category.strip()}")


# ============================================================================
# PATTERN 4: ORCHESTRATOR-WORKERS
# Decompose task dynamically, execute in parallel, synthesize
# ============================================================================

@dataclass
class OrchestratorState(WorkflowState):
    """State for orchestrator-workers pattern."""
    task: str = ""
    subtasks: List[str] = field(default_factory=list)
    results: List[str] = field(default_factory=list)
    synthesis: str = ""
    should_continue: bool = True


class TaskPlan(BaseModel):
    """Schema for task decomposition."""
    subtasks: List[str]


async def decompose_task(state: OrchestratorState) -> OrchestratorState:
    """Step 1: Orchestrator breaks down the task dynamically."""
    print(f"  → Decomposing task: {state.task[:50]}...")

    plan = await async_client.responses.parse(
        model="gpt-4o-mini",
        input=f"Break this into 3 specific subtasks: {state.task}",
        text_format=TaskPlan
    )

    state.subtasks = plan.output_parsed.subtasks
    print(f"  → Identified {len(state.subtasks)} subtasks")
    for i, subtask in enumerate(state.subtasks, 1):
        print(f"    {i}. {subtask}")

    return state


async def execute_workers(state: OrchestratorState) -> OrchestratorState:
    """Step 2: Workers execute subtasks in parallel."""
    print(f"  → Executing {len(state.subtasks)} workers in parallel...")

    tasks = [llm(f"Complete this subtask: {subtask}") for subtask in state.subtasks]
    state.results = await asyncio.gather(*tasks)

    return state


async def synthesize_results(state: OrchestratorState) -> OrchestratorState:
    """Step 3: Orchestrator combines results."""
    print("  → Synthesizing results...")

    combined = "\n".join([f"{i+1}. {r}" for i, r in enumerate(state.results)])
    state.synthesis = await llm(
        f"Synthesize these results for '{state.task}':\n{combined}"
    )

    return state


async def orchestrator_workers():
    """
    Orchestrator dynamically decomposes tasks and delegates to workers.

    Unlike parallelization (fixed tasks), the orchestrator determines
    subtasks based on the specific input.
    """
    workflow = Workflow[OrchestratorState](
        decompose_task,
        execute_workers,
        synthesize_results
    )

    result = await workflow.run(
        OrchestratorState(task="Research AI safety")
    )

    print(f"\nSynthesis:\n{result.synthesis[:200]}...\n")


# ============================================================================
# PATTERN 5: EVALUATOR-OPTIMIZER
# Generate → Evaluate → Refine loop for quality control
# ============================================================================

@dataclass
class OptimizerState(WorkflowState):
    """State for evaluator-optimizer pattern."""
    task: str = ""
    draft: str = ""
    score: int = 0
    feedback: str = ""
    iterations: int = 0
    final: str = ""
    should_continue: bool = True


class Evaluation(BaseModel):
    """Schema for evaluation."""
    score: int  # 1-10
    feedback: str


async def generate_or_refine(state: OptimizerState) -> OptimizerState:
    """Generate initial or refined version."""
    if state.iterations == 0:
        print("  → Generating initial draft...")
        state.draft = await llm(f"Generate: {state.task}")
    else:
        print(f"  → Refining (iteration {state.iterations})...")
        state.draft = await llm(
            f"Improve based on feedback: {state.feedback}\n\nCurrent: {state.draft}"
        )
    return state


async def evaluate_quality(state: OptimizerState) -> OptimizerState:
    """Evaluate and provide feedback."""
    print("  → Evaluating quality...")

    eval_result = await async_client.responses.parse(
        model="gpt-4o-mini",
        input=f"Score 1-10 and provide feedback:\n{state.draft}",
        text_format=Evaluation
    )

    state.score = eval_result.output_parsed.score
    state.feedback = eval_result.output_parsed.feedback
    state.iterations += 1

    print(f"  → Score: {state.score}/10")

    return state


async def check_quality(state: OptimizerState) -> OptimizerState:
    """Decide if quality is good enough."""
    if state.score >= 8 or state.iterations >= 3:
        state.final = state.draft
        state.should_continue = False
        print(f"  → Quality threshold met (score: {state.score}/10)")
    return state


async def evaluator_optimizer():
    """
    Generate-evaluate-refine loop with quality control.

    Continues iterating until quality threshold is met or max iterations reached.
    """
    # Create a loop workflow using should_continue flag
    async def optimize_loop(state: OptimizerState) -> OptimizerState:
        """Run generate-evaluate loop until quality threshold met."""
        while state.should_continue:
            state = await generate_or_refine(state)
            state = await evaluate_quality(state)
            state = await check_quality(state)
        return state

    workflow = Workflow[OptimizerState](optimize_loop)

    result = await workflow.run(
        OptimizerState(task="Write Python Fibonacci function")
    )

    print(f"\nFinal (after {result.iterations} iterations):")
    print(f"{result.final[:150]}...\n")


async def main():
    """Demonstrate all 5 workflow patterns using the Workflow framework."""
    print("=" * 70)
    print("WORKFLOW PATTERNS - Using the Workflow Framework")
    print("=" * 70)
    print("\nThis lesson demonstrates a reusable Workflow abstraction that")
    print("composes async steps with type-safe state management.\n")

    print("\n" + "=" * 70)
    print("PATTERN 1: PROMPT CHAINING")
    print("=" * 70)
    await prompt_chaining()

    print("\n" + "=" * 70)
    print("PATTERN 2: ROUTING")
    print("=" * 70)
    await routing()

    print("\n" + "=" * 70)
    print("PATTERN 3: PARALLELIZATION")
    print("=" * 70)
    await parallelization()

    print("\n" + "=" * 70)
    print("PATTERN 4: ORCHESTRATOR-WORKERS")
    print("=" * 70)
    await orchestrator_workers()

    print("\n" + "=" * 70)
    print("PATTERN 5: EVALUATOR-OPTIMIZER")
    print("=" * 70)
    await evaluator_optimizer()

    print("\n" + "=" * 70)
    print("KEY TAKEAWAYS")
    print("=" * 70)
    print("✓ The Workflow framework provides composable, type-safe orchestration")
    print("✓ WorkflowState base class enables control flow with should_continue")
    print("✓ Each pattern uses specialized state classes (ChainState, RoutingState, etc.)")
    print("✓ These 5 patterns are fundamental building blocks for AI agents")
    print("\nNext: Build autonomous agents that decide workflows dynamically!")
    print("=" * 70)


if __name__ == "__main__":
    asyncio.run(main())
