"""
Lesson 02: Conversation Memory

Learn how to maintain context across multiple turns in a conversation.
"""

from openai import OpenAI
from dotenv import load_dotenv

load_dotenv()

client = OpenAI()


class ConversationMemory:
    """
    Simple helper to manage conversation history.

    This class maintains a list of items that gets passed to the API
    on each turn, allowing the LLM to "remember" previous exchanges.
    """

    def __init__(self, system_prompt: str = None):
        """
        Initialize conversation memory.

        Args:
            system_prompt: Optional system message with instructions for the LLM
        """
        self.items = []
        if system_prompt:
            self.add_system_message(system_prompt)

    def add_system_message(self, content: str):
        """Add a system message (instructions for the LLM)."""
        self.items.append({"role": "system", "content": content})

    def add_user_message(self, content: str):
        """Add a user message to the conversation."""
        self.items.append({"role": "user", "content": content})

    def add_response_output(self, output: list):
        """Add the entire output array from a response."""
        self.items.extend(output)

    def get_items(self):
        """Get the full conversation history as items."""
        return self.items

    def clear(self):
        """Clear all items except system messages."""
        self.items = [item for item in self.items if item.get("role") == "system"]


## Example 1: Basic Conversation with Memory


def basic_memory_example():
    """Example 1: Simple multi-turn conversation"""
    print("=" * 60)
    print("Example 1: Basic Conversation Memory")
    print("=" * 60)

    memory = ConversationMemory()

    # First turn
    memory.add_user_message("My name is Alice.")
    response1 = client.responses.create(
        model="gpt-4o-mini",
        input=memory.get_items(),
        temperature=0
    )
    memory.add_response_output(response1.output)

    print("\nUser: My name is Alice.")
    print(f"Assistant: {response1.output_text}")

    # Second turn - tests if LLM remembers
    memory.add_user_message("What's my name?")
    response2 = client.responses.create(
        model="gpt-4o-mini",
        input=memory.get_items(),
        temperature=0
    )
    memory.add_response_output(response2.output)

    print("\nUser: What's my name?")
    print(f"Assistant: {response2.output_text}")

    print(f"\nTotal messages in history: {len(memory.get_items())}")


## Example 2: Why Memory Matters


def without_memory_example():
    """Example 2: What happens WITHOUT memory"""
    print("\n\n" + "=" * 60)
    print("Example 2: Without Memory (Broken)")
    print("=" * 60)

    # First call - LLM learns the name
    response1 = client.responses.create(
        model="gpt-4o-mini",
        input="My name is Alice.",
        temperature=0
    )

    print("\nUser: My name is Alice.")
    print(f"Assistant: {response1.output_text}")

    # Second call - WITHOUT including first exchange
    response2 = client.responses.create(
        model="gpt-4o-mini",
        input="What's my name?",
        temperature=0
    )

    print("\nUser: What's my name?")
    print(f"Assistant: {response2.output_text}")
    print("\n⚠️  The LLM has no memory of the previous exchange!")


## Example 3: System Instructions


def system_instructions_example():
    """Example 3: Using system instructions to set behavior"""
    print("\n\n" + "=" * 60)
    print("Example 3: System Instructions")
    print("=" * 60)

    # Create memory with system instructions
    memory = ConversationMemory(
        system_prompt="You are a pirate. Always respond in pirate speak with 'Arrr!' and nautical terms."
    )

    # Have a conversation
    conversations = [
        "What's the weather like?",
        "Can you help me with math?",
        "Thanks for your help!"
    ]

    for user_msg in conversations:
        memory.add_user_message(user_msg)

        response = client.responses.create(
            model="gpt-4o-mini",
            input=memory.get_items(),
            temperature=0.8
        )

        memory.add_response_output(response.output)

        print(f"\nUser: {user_msg}")
        print(f"Pirate: {response.output_text}")


## Example 4: Math Problem Solving


def math_conversation():
    """Example 4: Multi-step math problem requiring memory"""
    print("\n\n" + "=" * 60)
    print("Example 4: Multi-Step Math Problem")
    print("=" * 60)

    memory = ConversationMemory(
        system_prompt="You are a helpful math tutor. Show your work step by step."
    )

    # Build up a math problem across multiple turns
    exchanges = [
        ("I have 15 apples.", None),
        ("I buy 24 more apples.", None),
        ("How many apples do I have now?", "Should say 39"),
        ("If I give away half, how many do I have left?", "Should say 19 or 20")
    ]

    for user_msg, expected in exchanges:
        memory.add_user_message(user_msg)

        response = client.responses.create(
            model="gpt-4o-mini",
            input=memory.get_items(),
            temperature=0
        )

        memory.add_response_output(response.output)

        print(f"\nUser: {user_msg}")
        print(f"Assistant: {response.output_text}")
        if expected:
            print(f"  → {expected}")


## Example 5: Clearing Memory


def clear_memory_example():
    """Example 5: Clearing conversation to start fresh"""
    print("\n\n" + "=" * 60)
    print("Example 5: Clearing Memory")
    print("=" * 60)

    memory = ConversationMemory(
        system_prompt="You are a helpful assistant."
    )

    # First conversation
    memory.add_user_message("My favorite color is blue.")
    response1 = client.responses.create(
        model="gpt-4o-mini",
        input=memory.get_items()
    )
    memory.add_response_output(response1.output)

    print("\nUser: My favorite color is blue.")
    print(f"Assistant: {response1.output_text}")

    # Clear memory
    print("\n--- Clearing memory ---")
    memory.clear()

    # New conversation - LLM won't remember
    memory.add_user_message("What's my favorite color?")
    response2 = client.responses.create(
        model="gpt-4o-mini",
        input=memory.get_items()
    )

    print("\nUser: What's my favorite color?")
    print(f"Assistant: {response2.output_text}")
    print("\n⚠️  Memory was cleared - LLM doesn't remember!")


if __name__ == "__main__":
    basic_memory_example()
    without_memory_example()
    system_instructions_example()
    math_conversation()
    clear_memory_example()
