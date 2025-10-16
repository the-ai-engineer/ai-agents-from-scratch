"""
Lesson 05: Workflow Patterns

Learn to orchestrate multiple LLM calls using five fundamental patterns:
1. Prompt Chaining - Sequential workflows
2. Routing - Conditional branching
3. Parallelization - Concurrent execution
4. Orchestrator-Workers - Dynamic task decomposition
5. Evaluator-Optimizer - Generate-evaluate-refine loops

Run each pattern in separate Jupyter cells to see the results.
"""

import asyncio
import time

from typing import Literal, List
from openai import AsyncOpenAI
from pydantic import BaseModel
from dotenv import load_dotenv

load_dotenv()

client = AsyncOpenAI()

# ============================================================================
# HELPER FUNCTIONS
# ============================================================================


async def llm(prompt: str, instructions: str = "", model: str = "gpt-4o-mini") -> str:
    """Simple async LLM call."""
    response = await client.responses.create(
        model=model,
        instructions=instructions if instructions else None,
        input=prompt,
    )
    return response.output_text


# ============================================================================
# PATTERN 1: PROMPT CHAINING
# Sequential steps where each builds on the previous
# ============================================================================


async def prompt_chaining(topic: str = "AI agents"):
    """
    Chain multiple LLM calls sequentially.
    Each step depends on the previous step's output.
    """
    # Step 1: Create outline
    outline = await llm(f"Create a 3-point outline for: {topic}")

    # Step 2: Write draft based on outline
    draft = await llm(f"Write the first point (100 words):\n{outline}")

    # Step 3: Polish the draft
    final = await llm(f"Edit for clarity and conciseness:\n{draft}")

    return {"outline": outline, "draft": draft, "final": final}


# Run in Jupyter:
# result = await prompt_chaining("machine learning basics")
# print(result["final"])


# ============================================================================
# PATTERN 2: ROUTING
# Classify query type and route to specialized handlers
# ============================================================================


class SupportCategory(BaseModel):
    category: Literal["shipping", "refund", "technical", "general"]
    urgency: Literal["high", "medium", "low"]


async def routing(customer_query: str):
    """
    Route customer queries to specialized handlers.
    Different response strategies for different issue types.
    """
    # Classify the query
    classification = await client.responses.parse(
        model="gpt-4o-mini",
        input=f"Classify this customer support query: {customer_query}",
        text_format=SupportCategory,
    )
    category = classification.output_parsed.category
    urgency = classification.output_parsed.urgency

    # Route to appropriate specialist
    specialists = {
        "shipping": "You are a shipping specialist. Check tracking info and provide delivery estimates.",
        "refund": "You are a refund specialist. Be empathetic about payment issues and explain the process.",
        "technical": "You are a technical support expert. Provide clear troubleshooting steps.",
        "general": "You are a friendly customer service representative.",
    }

    # Add urgency context
    instructions = specialists[category]
    if urgency == "high":
        instructions += " This is urgent - prioritize immediate resolution."

    response = await llm(customer_query, instructions=instructions)

    return {
        "query": customer_query,
        "category": category,
        "urgency": urgency,
        "response": response,
    }


# await routing("Urgent! I need a refind on my order!!")

# ============================================================================
# PATTERN 3: PARALLELIZATION
# Process multiple items concurrently
# ============================================================================


async def parallelization(emails: List[str]):
    """
    Process multiple items in parallel for speed.
    Much faster than sequential processing.
    """

    async def classify_email(email: str) -> str:
        return await llm(
            f"Classify as spam/urgent/normal/newsletter (one word only): {email}"
        )

    # Process all emails concurrently
    start = time.time()
    results = await asyncio.gather(*[classify_email(email) for email in emails])
    elapsed = time.time() - start

    return {"emails": emails, "classifications": results, "time": elapsed}


# emails = [
#     "URGENT: Your account will be suspended!",
#     "Weekly AI newsletter",
#     "Meeting reminder: 3pm today"
# ]
# result = await parallelization(emails)
# for email, classification in zip(result["emails"], result["classifications"]):
#     print(f"{email[:30]}... → {classification}")


# ============================================================================
# PATTERN 4: ORCHESTRATOR-WORKERS
# Decompose task dynamically, execute in parallel
# ============================================================================


class TaskPlan(BaseModel):
    subtasks: List[str]


async def orchestrator_workers(task: str):
    """
    Orchestrator decomposes task, workers execute in parallel.
    Dynamic task breakdown based on the input.
    """
    # Orchestrator: decompose task
    plan = await client.responses.parse(
        model="gpt-4o-mini",
        input=f"Break this into 3 specific subtasks: {task}",
        text_format=TaskPlan,
    )
    subtasks = plan.output_parsed.subtasks

    # Workers: execute subtasks in parallel
    results = await asyncio.gather(
        *[llm(f"Complete this task: {subtask}") for subtask in subtasks]
    )

    # Orchestrator: synthesize results
    synthesis = await llm(
        f"Combine these results for '{task}':\n"
        + "\n".join([f"- {r}" for r in results])
    )

    return {"task": task, "subtasks": subtasks, "synthesis": synthesis}


# Run in Jupyter:
# result = await orchestrator_workers("Plan a product launch")
# print("Subtasks:", result["subtasks"])
# print("\nSynthesis:", result["synthesis"])


# ============================================================================
# PATTERN 5: EVALUATOR-OPTIMIZER
# Generate → Evaluate → Refine loop
# ============================================================================


class Evaluation(BaseModel):
    score: int  # 1-10
    feedback: str


async def evaluator_optimizer(
    task: str, target_score: int = 8, max_iterations: int = 3
):
    """
    Generate-evaluate-refine loop with quality control.
    Iterates until quality threshold is met.
    """
    draft = await llm(task)

    for iteration in range(max_iterations):
        # Evaluate current version
        evaluation = await client.responses.parse(
            model="gpt-4o-mini",
            input=f"Score 1-10 and give brief feedback:\n{draft}",
            text_format=Evaluation,
        )

        score = evaluation.output_parsed.score
        feedback = evaluation.output_parsed.feedback

        # Check if good enough
        if score >= target_score:
            return {"final": draft, "score": score, "iterations": iteration + 1}

        # Refine based on feedback
        draft = await llm(f"Improve based on: {feedback}\n\nCurrent:\n{draft}")

    return {"final": draft, "score": score, "iterations": max_iterations}


# Run in Jupyter:
# result = await evaluator_optimizer("Write a Python function for binary search")
# print(f"Score: {result['score']}/10 after {result['iterations']} iterations")
# print(f"\nFinal:\n{result['final']}")


# ============================================================================
# COMPARISON: Run all patterns
# ============================================================================


async def demo_all_patterns():
    """
    Demonstrate all 5 patterns with simple examples.
    Run this to see all patterns in action.
    """
    print("1. PROMPT CHAINING")
    chain_result = await prompt_chaining("neural networks")
    print(f"   Generated {len(chain_result['final'])} chars\n")

    print("2. ROUTING")
    route_result = await routing("How do I reset my password?")
    print(f"   Routed to: {route_result['category']}\n")

    print("3. PARALLELIZATION")
    emails = ["Meeting at 3pm", "50% off sale!", "Project update"]
    parallel_result = await parallelization(emails)
    print(f"   Processed {len(emails)} emails in {parallel_result['time']:.2f}s\n")

    print("4. ORCHESTRATOR-WORKERS")
    orchestrator_result = await orchestrator_workers("Write a blog post")
    print(f"   Created {len(orchestrator_result['subtasks'])} subtasks\n")

    print("5. EVALUATOR-OPTIMIZER")
    optimizer_result = await evaluator_optimizer("Hello world in Python")
    print(
        f"   Score: {optimizer_result['score']}/10 after {optimizer_result['iterations']} iterations"
    )


# Run in Jupyter:
# await demo_all_patterns()
