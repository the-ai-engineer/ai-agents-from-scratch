"""
Lesson 01: OpenAI API Basics with Responses API

Learn how to make your first API call using the new Responses API.
"""

from openai import OpenAI
from dotenv import load_dotenv

load_dotenv()

client = OpenAI()


def basic_response():
    """Example 1: Basic API call"""
    response = client.responses.create(
        model="gpt-4o-mini",
        instructions="You are a helpful Python programming assistant.",
        input="Explain what async/await does in Python in 2 sentences.",
        temperature=0.7,
    )

    print(response.output_text)


def streaming_response():
    """Example 2: Streaming responses for better UX"""
    stream = client.responses.create(
        model="gpt-4o-mini",
        input="Write a short haiku about coding.",
        temperature=0.8,
        stream=True,
    )

    print("\nStreaming response:")
    for event in stream:
        if event.type == "content.delta":
            print(event.delta, end="", flush=True)
    print("\n")


def stateful_conversation():
    """Example 3: Multi-turn conversation with history"""
    # First turn
    response1 = client.responses.create(
        model="gpt-4o-mini",
        input="What is 15 * 24?",
        temperature=0,
    )

    print("\nUser: What is 15 * 24?")
    print(f"Assistant: {response1.output_text}")

    # Second turn - pass conversation history
    response2 = client.responses.create(
        model="gpt-4o-mini",
        input=[
            {"role": "user", "content": "What is 15 * 24?"},
            {"role": "assistant", "content": response1.output_text},
            {"role": "user", "content": "Now multiply that result by 2"},
        ],
        temperature=0,
    )

    print("\nUser: Now multiply that result by 2")
    print(f"Assistant: {response2.output_text}")


if __name__ == "__main__":
    basic_response()
    streaming_response()
    stateful_conversation()
