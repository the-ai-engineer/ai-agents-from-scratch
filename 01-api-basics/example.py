"""
Lesson 01: OpenAI API Basics

Learn how to make your first API call to OpenAI.
"""

from openai import OpenAI, AsyncOpenAI
from dotenv import load_dotenv
import asyncio

load_dotenv()

client = OpenAI()

# AsyncOpenAI: Use when you need to make multiple concurrent API calls or integrate
# with async frameworks (FastAPI, asyncio). The async client allows non-blocking I/O,
# enabling parallel requests for better performance. See Example 4 below for usage.
async_client = AsyncOpenAI()


## Example 1: Your First API Call


def basic_response():
    """Example 1: Basic API call using Responses API"""

    response = client.responses.create(
        model="gpt-4o-mini",
        instructions="You are a helpful Python programming assistant.",
        input="Explain what async/await does in Python in 2 sentences.",
    )

    print(f"\n{response.output_text}\n")


## Example 2: Streaming Responses


def streaming_response():
    """Example 2: Streaming responses for better UX"""

    stream = client.responses.create(
        model="gpt-4o-mini",
        input="Write a short haiku about coding.",
        stream=True,
    )

    for event in stream:
        if hasattr(event, "type") and "text.delta" in event.type:
            print(event.delta, end="", flush=True)


## Example 3: Temperature Control


def temperature_examples():
    """Example 3: Understanding temperature for creativity control"""

    prompt = "Complete this sentence: The future of AI is"

    # Low temperature (0.0) - Deterministic, focused
    print("\nTemperature 0.0 (Deterministic):")
    response_low = client.responses.create(
        model="gpt-4o-mini",
        input=prompt,
        temperature=0.0,
    )
    print(f"  {response_low.output_text}")

    # Medium temperature (0.7) - Balanced
    print("\nTemperature 0.7 (Balanced):")
    response_med = client.responses.create(
        model="gpt-4o-mini",
        input=prompt,
        temperature=0.5,
    )
    print(f"  {response_med.output_text}")

    # High temperature (1.5) - Creative, varied
    print("\nTemperature 1.5 (Creative):")
    response_high = client.responses.create(
        model="gpt-4o-mini",
        input=prompt,
        temperature=1,
    )
    print(f"  {response_high.output_text}\n")

    print("💡 Lower temperature = more focused and deterministic")
    print("💡 Higher temperature = more creative and varied\n")


## Example 4: Async Client for Concurrent Requests


async def async_concurrent_requests():
    """Example 4: Using AsyncOpenAI for parallel API calls"""

    prompts = ["What is Python?", "What is JavaScript?", "What is Rust?"]

    print("\nMaking 3 concurrent API calls with async client...")

    # Create multiple concurrent requests
    tasks = [
        async_client.responses.create(
            model="gpt-4o-mini",
            input=prompt,
            temperature=0.7,
        )
        for prompt in prompts
    ]

    # Execute all requests concurrently
    responses = await asyncio.gather(*tasks)

    for prompt, response in zip(prompts, responses):
        print(f"\n{prompt}")
        print(f"→ {response.output_text[:100]}...")

    print("\n💡 Async client is faster when making multiple requests!")
    print("💡 Use it with FastAPI, asyncio workflows, or parallel processing\n")


if __name__ == "__main__":
    # basic_response()
    streaming_response()
    # temperature_examples()
    # asyncio.run(async_concurrent_requests())
