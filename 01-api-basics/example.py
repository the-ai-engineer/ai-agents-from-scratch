"""
Lesson 01: OpenAI API Basics

Learn how to make your first API call to OpenAI.
"""

from openai import OpenAI
from dotenv import load_dotenv

load_dotenv()

client = OpenAI()


## Example 1: Your First API Call


def basic_response():
    """Example 1: Basic API call using Responses API"""
    print("=" * 60)
    print("Example 1: Basic API Call")
    print("=" * 60)

    response = client.responses.create(
        model="gpt-4o-mini",
        instructions="You are a helpful Python programming assistant.",
        input="Explain what async/await does in Python in 2 sentences.",
        temperature=0.7,
    )

    print(f"\n{response.output_text}\n")


## Example 2: Streaming Responses


def streaming_response():
    """Example 2: Streaming responses for better UX"""
    print("=" * 60)
    print("Example 2: Streaming Response")
    print("=" * 60)

    print("\n", end="")

    stream = client.responses.create(
        model="gpt-4o-mini",
        input="Write a short haiku about coding.",
        temperature=0.8,
        stream=True,
    )

    for event in stream:
        if event.type == "content.delta":
            print(event.delta, end="", flush=True)

    print("\n")


## Example 3: Temperature Control


def temperature_examples():
    """Example 3: Understanding temperature for creativity control"""
    print("=" * 60)
    print("Example 3: Temperature Control")
    print("=" * 60)

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
        temperature=0.7,
    )
    print(f"  {response_med.output_text}")

    # High temperature (1.5) - Creative, varied
    print("\nTemperature 1.5 (Creative):")
    response_high = client.responses.create(
        model="gpt-4o-mini",
        input=prompt,
        temperature=1.5,
    )
    print(f"  {response_high.output_text}\n")

    print("💡 Lower temperature = more focused and deterministic")
    print("💡 Higher temperature = more creative and varied\n")


if __name__ == "__main__":
    basic_response()
    streaming_response()
    temperature_examples()
