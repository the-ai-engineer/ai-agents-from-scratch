"""
Lesson 06: Workflow Patterns

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
"""

import os
import asyncio
import time
from typing import Literal, Optional
from dataclasses import dataclass
from openai import OpenAI, AsyncOpenAI
from pydantic import BaseModel, Field
from dotenv import load_dotenv

load_dotenv()

client = OpenAI(api_key=os.getenv("OPENAI_API_KEY"))
async_client = AsyncOpenAI(api_key=os.getenv("OPENAI_API_KEY"))


@dataclass
class WorkflowState:
    """
    State object for workflow patterns.

    Workflows have predefined flows and need to track intermediate results,
    not conversational context. Each field represents a stage in the workflow.
    """
    topic: str
    outline: Optional[str] = None
    draft: Optional[str] = None
    final: Optional[str] = None


# === PATTERN 1: PROMPT CHAINING ===

def prompt_chaining():
    """
    Chain multiple LLM calls sequentially using WorkflowState.

    Workflows have deterministic flows: you control the order of operations.
    State tracks intermediate results as they flow through the pipeline.
    """
    # Initialize workflow state
    state = WorkflowState(topic="The benefits of learning AI from first principles")

    # Step 1: Generate outline
    print("Step 1: Generating outline...")
    state.outline = client.responses.create(
        model="gpt-4o-mini",
        input=f"Create a 3-section outline for a blog post about: {state.topic}",
        temperature=0.7
    ).output_text
    print(f"Outline:\n{state.outline}\n")

    # Step 2: Write content (depends on outline)
    print("Step 2: Writing first section...")
    state.draft = client.responses.create(
        model="gpt-4o-mini",
        input=f"Using this outline, write the first section (150 words):\n{state.outline}",
        temperature=0.7
    ).output_text
    print(f"Draft:\n{state.draft}\n")

    # Step 3: Polish (depends on draft)
    print("Step 3: Polishing...")
    state.final = client.responses.create(
        model="gpt-4o-mini",
        input=f"Edit for clarity and impact:\n{state.draft}",
        temperature=0.3
    ).output_text
    print(f"Final:\n{state.final}\n")

    # At the end, state contains all intermediate results
    return state


# === PATTERN 2: ROUTING ===

class QueryClassification(BaseModel):
    category: Literal["technical", "billing", "general"]
    confidence: float = Field(ge=0.0, le=1.0)


def classify_query(query: str) -> QueryClassification:
    """Classify query to determine routing"""
    response = client.beta.chat.completions.parse(
        model="gpt-4o-mini",
        messages=[
            {
                "role": "system",
                "content": "Classify as: technical (bugs, how-to), billing (payments), or general (other)"
            },
            {"role": "user", "content": query}
        ],
        response_format=QueryClassification
    )
    return response.choices[0].message.parsed


def routing():
    """Route queries to specialized handlers"""
    queries = [
        "My app keeps crashing when I export data",
        "I was charged twice this month",
        "What are your office hours?"
    ]

    for query in queries:
        print(f"\nQuery: {query}")

        # Step 1: Classify
        classification = classify_query(query)
        print(f"Category: {classification.category} ({classification.confidence:.2f})")

        # Step 2: Route to specialized handler
        instructions = {
            "technical": "You are a technical support specialist. Provide step-by-step solutions.",
            "billing": "You are a billing specialist. Be empathetic about payment issues.",
            "general": "You are a friendly customer service rep."
        }

        response = client.responses.create(
            model="gpt-4o-mini",
            instructions=instructions[classification.category],
            input=query
        )
        print(f"Response: {response.output_text}\n")


# === PATTERN 3: PARALLELIZATION ===

async def classify_email_async(email: str) -> str:
    """Async email classification"""
    response = await async_client.responses.create(
        model="gpt-4o-mini",
        instructions="Classify as: spam, urgent, normal, or newsletter. Reply with just the category.",
        input=email,
        temperature=0.3
    )
    return response.output_text.strip()


async def parallelization():
    """Process multiple items concurrently"""
    emails = [
        "URGENT: Your account will be suspended!",
        "Weekly newsletter: Top 10 AI developments",
        "Meeting reminder: Team standup at 10am",
        "Congratulations! You've won a million dollars!",
        "Your invoice for March is attached"
    ]

    print("\nProcessing 5 emails...")

    # Sequential
    start = time.time()
    sequential_results = []
    for email in emails:
        result = await classify_email_async(email)
        sequential_results.append(result)
    sequential_time = time.time() - start

    # Parallel
    start = time.time()
    tasks = [classify_email_async(email) for email in emails]
    parallel_results = await asyncio.gather(*tasks)
    parallel_time = time.time() - start

    print(f"\nSequential: {sequential_time:.2f}s")
    print(f"Parallel: {parallel_time:.2f}s")
    print(f"Speedup: {sequential_time/parallel_time:.1f}x\n")

    for email, category in zip(emails, parallel_results):
        print(f"'{email[:40]}...' → {category}")


# === PATTERN 4: ORCHESTRATOR-WORKERS ===

@dataclass
class ResearchState:
    """State for orchestrator-workers pattern"""
    topic: str
    subtopics: list[str] = None
    research_results: dict[str, str] = None
    final_report: str = None


async def research_subtopic(subtopic: str) -> str:
    """Worker that researches a specific subtopic"""
    response = await async_client.responses.create(
        model="gpt-4o-mini",
        instructions="You are a research assistant. Provide a concise 2-3 sentence summary.",
        input=f"Research this subtopic: {subtopic}",
        temperature=0.7
    )
    return response.output_text


async def orchestrator_workers():
    """
    Central orchestrator decomposes task and delegates to workers.

    Unlike parallelization (fixed tasks), orchestrator dynamically determines
    subtasks based on the specific input.
    """
    state = ResearchState(topic="The impact of AI on software development")

    # Step 1: Orchestrator breaks down the task
    print(f"Researching: {state.topic}")
    print("\nStep 1: Orchestrator decomposing task...")

    decomposition = await async_client.responses.create(
        model="gpt-4o-mini",
        instructions="Break down the research topic into 3 specific subtopics. Return as a numbered list.",
        input=state.topic,
        temperature=0.7
    )

    # Parse subtopics (simplified parsing)
    state.subtopics = [
        line.split('. ', 1)[1] if '. ' in line else line.strip()
        for line in decomposition.output_text.strip().split('\n')
        if line.strip() and any(char.isdigit() for char in line[:3])
    ][:3]  # Take first 3

    print("Subtopics identified:")
    for i, subtopic in enumerate(state.subtopics, 1):
        print(f"  {i}. {subtopic}")

    # Step 2: Workers research subtopics in parallel
    print("\nStep 2: Workers researching subtopics in parallel...")
    tasks = [research_subtopic(subtopic) for subtopic in state.subtopics]
    results = await asyncio.gather(*tasks)

    state.research_results = dict(zip(state.subtopics, results))

    for subtopic, result in state.research_results.items():
        print(f"\n  {subtopic}:")
        print(f"  {result}")

    # Step 3: Orchestrator synthesizes final report
    print("\nStep 3: Synthesizing final report...")
    synthesis_input = f"Topic: {state.topic}\n\n"
    for subtopic, result in state.research_results.items():
        synthesis_input += f"{subtopic}:\n{result}\n\n"
    synthesis_input += "Synthesize these findings into a cohesive 2-paragraph summary."

    final = await async_client.responses.create(
        model="gpt-4o-mini",
        input=synthesis_input,
        temperature=0.7
    )
    state.final_report = final.output_text

    print(f"\nFinal Report:\n{state.final_report}\n")


# === PATTERN 5: EVALUATOR-OPTIMIZER ===

@dataclass
class OptimizationState:
    """State for evaluator-optimizer pattern"""
    task: str
    draft: str = None
    feedback: str = None
    final: str = None
    iterations: int = 0


def evaluator_optimizer():
    """
    Generate → Evaluate → Refine loop.

    Useful for tasks requiring quality control or iterative improvement.
    """
    state = OptimizationState(
        task="Write a Python function that calculates the Fibonacci sequence"
    )

    print(f"Task: {state.task}\n")

    # Step 1: Generator creates initial draft
    print("Step 1: Generating initial draft...")
    state.draft = client.responses.create(
        model="gpt-4o-mini",
        instructions="You are a Python coding assistant. Write clean, documented code.",
        input=state.task,
        temperature=0.7
    ).output_text
    print(f"Draft:\n{state.draft}\n")

    # Step 2: Evaluator provides feedback
    print("Step 2: Evaluating code quality...")
    state.feedback = client.responses.create(
        model="gpt-4o-mini",
        instructions="You are a code reviewer. Provide specific, actionable feedback on code quality, efficiency, and documentation.",
        input=f"Review this code:\n\n{state.draft}",
        temperature=0.3
    ).output_text
    print(f"Feedback:\n{state.feedback}\n")

    # Step 3: Generator refines based on feedback
    print("Step 3: Refining based on feedback...")
    state.final = client.responses.create(
        model="gpt-4o-mini",
        instructions="You are a Python coding assistant. Improve the code based on the feedback.",
        input=f"Original code:\n{state.draft}\n\nFeedback:\n{state.feedback}\n\nProvide improved code:",
        temperature=0.7
    ).output_text
    print(f"Final (Improved):\n{state.final}\n")

    state.iterations = 1
    print(f"Completed {state.iterations} iteration(s) of improvement")


async def main():
    print("="*60)
    print("WORKFLOW PATTERNS")
    print("="*60)

    print("\n\n=== PATTERN 1: PROMPT CHAINING ===")
    prompt_chaining()

    print("\n\n=== PATTERN 2: ROUTING ===")
    routing()

    print("\n\n=== PATTERN 3: PARALLELIZATION ===")
    await parallelization()

    print("\n\n=== PATTERN 4: ORCHESTRATOR-WORKERS ===")
    await orchestrator_workers()

    print("\n\n=== PATTERN 5: EVALUATOR-OPTIMIZER ===")
    evaluator_optimizer()

    print("\n\n" + "="*60)
    print("Key Insight: These 5 patterns give you control over workflows.")
    print("Next lesson: Agents that decide workflows dynamically!")
    print("="*60)


if __name__ == "__main__":
    asyncio.run(main())
