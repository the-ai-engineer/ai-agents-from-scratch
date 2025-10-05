"""
Lesson 06: Workflow Patterns

Learn to orchestrate multiple LLM calls using three fundamental patterns:
1. Prompt Chaining - Sequential workflows
2. Routing - Conditional branching
3. Parallelization - Concurrent execution
"""

import os
import asyncio
import time
from typing import Literal
from openai import OpenAI, AsyncOpenAI
from pydantic import BaseModel, Field
from dotenv import load_dotenv

load_dotenv()

client = OpenAI(api_key=os.getenv("OPENAI_API_KEY"))
async_client = AsyncOpenAI(api_key=os.getenv("OPENAI_API_KEY"))


# === PATTERN 1: PROMPT CHAINING ===

def prompt_chaining():
    """Chain multiple LLM calls sequentially"""
    topic = "The benefits of learning AI from first principles"

    # Step 1: Generate outline
    print("Step 1: Generating outline...")
    outline = client.responses.create(
        model="gpt-4o-mini",
        input=f"Create a 3-section outline for a blog post about: {topic}",
        temperature=0.7
    ).output_text
    print(f"Outline:\n{outline}\n")

    # Step 2: Write content
    print("Step 2: Writing first section...")
    draft = client.responses.create(
        model="gpt-4o-mini",
        input=f"Using this outline, write the first section (150 words):\n{outline}",
        temperature=0.7
    ).output_text
    print(f"Draft:\n{draft}\n")

    # Step 3: Polish
    print("Step 3: Polishing...")
    final = client.responses.create(
        model="gpt-4o-mini",
        input=f"Edit for clarity and impact:\n{draft}",
        temperature=0.3
    ).output_text
    print(f"Final:\n{final}\n")


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

    print("\n\n" + "="*60)
    print("Key Insight: These patterns give you control over workflows.")
    print("Next lesson: Agents that decide workflows dynamically!")
    print("="*60)


if __name__ == "__main__":
    asyncio.run(main())
