"""
Lesson 02: Conversation Memory

Learn how to maintain context across multiple turns in a conversation.
"""

from openai import OpenAI
from openai import conversations
from dotenv import load_dotenv

load_dotenv()

client = OpenAI()


def stateful_conversation(messages: list[str]):
    """
    Have a multi-turn conversation with stateful context.

    Args:
        messages: List of user messages

    Returns:
        List of assistant responses
    """
    conversation = conversations.create()

    for i, message in enumerate(messages):
        print(f"\nTurn {i + 1}: {message}")
        response = client.responses.create(
            model="gpt-4o-mini",
            input=message,
            # Let open AI manage our memory
            store=True,
            conversation=conversation.id,
        )

        print(f"Response: {response.output_text}")


# Usage
conversation = [
    "My name is Owain?",
    "What is my name?",
    "Can you remind me of my name?",
]

stateful_conversation(conversation)
