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

    This class maintains a list of messages that gets passed to the API
    on each turn, allowing the LLM to "remember" previous exchanges.
    """

    def __init__(self, instructions: str = None):
        """
        Initialize conversation memory.

        Args:
            instructions: Optional system message with instructions for the LLM
        """
        self.messages = []
        if instructions:
            self.messages.append({"role": "system", "content": instructions})

    def add_message(self, role: str, content: str):
        """
        Add a message to the conversation history.

        Args:
            role: Either "user" or "assistant"
            content: The message content
        """
        self.messages.append({"role": role, "content": content})

    def get_history(self):
        """Get the full conversation history."""
        return self.messages

    def clear(self):
        """Clear all messages except system message."""
        system_messages = [msg for msg in self.messages if msg["role"] == "system"]
        self.messages = system_messages


## Example 1: Basic Conversation with Memory


def basic_memory_example():
    """Example 1: Simple multi-turn conversation"""
    print("=" * 60)
    print("Example 1: Basic Conversation Memory")
    print("=" * 60)

    memory = ConversationMemory()

    # First turn
    memory.add_message("user", "My name is Alice.")
    response1 = client.chat.completions.create(
        model="gpt-4o-mini",
        messages=memory.get_history(),
        temperature=0
    )
    assistant_reply1 = response1.choices[0].message.content
    memory.add_message("assistant", assistant_reply1)

    print("\nUser: My name is Alice.")
    print(f"Assistant: {assistant_reply1}")

    # Second turn - tests if LLM remembers
    memory.add_message("user", "What's my name?")
    response2 = client.chat.completions.create(
        model="gpt-4o-mini",
        messages=memory.get_history(),
        temperature=0
    )
    assistant_reply2 = response2.choices[0].message.content
    memory.add_message("assistant", assistant_reply2)

    print("\nUser: What's my name?")
    print(f"Assistant: {assistant_reply2}")

    print(f"\nTotal messages in history: {len(memory.get_history())}")


## Example 2: Why Memory Matters


def without_memory_example():
    """Example 2: What happens WITHOUT memory"""
    print("\n\n" + "=" * 60)
    print("Example 2: Without Memory (Broken)")
    print("=" * 60)

    # First call - LLM learns the name
    messages1 = [{"role": "user", "content": "My name is Alice."}]
    response1 = client.chat.completions.create(
        model="gpt-4o-mini",
        messages=messages1,
        temperature=0
    )

    print("\nUser: My name is Alice.")
    print(f"Assistant: {response1.choices[0].message.content}")

    # Second call - WITHOUT including first exchange
    messages2 = [{"role": "user", "content": "What's my name?"}]
    response2 = client.chat.completions.create(
        model="gpt-4o-mini",
        messages=messages2,
        temperature=0
    )

    print("\nUser: What's my name?")
    print(f"Assistant: {response2.choices[0].message.content}")
    print("\n⚠️  The LLM has no memory of the previous exchange!")


## Example 3: System Instructions


def system_instructions_example():
    """Example 3: Using system instructions to set behavior"""
    print("\n\n" + "=" * 60)
    print("Example 3: System Instructions")
    print("=" * 60)

    # Create memory with system instructions
    memory = ConversationMemory(
        instructions="You are a pirate. Always respond in pirate speak with 'Arrr!' and nautical terms."
    )

    # Have a conversation
    conversations = [
        "What's the weather like?",
        "Can you help me with math?",
        "Thanks for your help!"
    ]

    for user_msg in conversations:
        memory.add_message("user", user_msg)

        response = client.chat.completions.create(
            model="gpt-4o-mini",
            messages=memory.get_history(),
            temperature=0.8
        )

        assistant_reply = response.choices[0].message.content
        memory.add_message("assistant", assistant_reply)

        print(f"\nUser: {user_msg}")
        print(f"Pirate: {assistant_reply}")


## Example 4: Math Problem Solving


def math_conversation():
    """Example 4: Multi-step math problem requiring memory"""
    print("\n\n" + "=" * 60)
    print("Example 4: Multi-Step Math Problem")
    print("=" * 60)

    memory = ConversationMemory(
        instructions="You are a helpful math tutor. Show your work step by step."
    )

    # Build up a math problem across multiple turns
    exchanges = [
        ("I have 15 apples.", None),
        ("I buy 24 more apples.", None),
        ("How many apples do I have now?", "Should say 39"),
        ("If I give away half, how many do I have left?", "Should say 19 or 20")
    ]

    for user_msg, expected in exchanges:
        memory.add_message("user", user_msg)

        response = client.chat.completions.create(
            model="gpt-4o-mini",
            messages=memory.get_history(),
            temperature=0
        )

        assistant_reply = response.choices[0].message.content
        memory.add_message("assistant", assistant_reply)

        print(f"\nUser: {user_msg}")
        print(f"Assistant: {assistant_reply}")
        if expected:
            print(f"  → {expected}")


## Example 5: Clearing Memory


def clear_memory_example():
    """Example 5: Clearing conversation to start fresh"""
    print("\n\n" + "=" * 60)
    print("Example 5: Clearing Memory")
    print("=" * 60)

    memory = ConversationMemory(
        instructions="You are a helpful assistant."
    )

    # First conversation
    memory.add_message("user", "My favorite color is blue.")
    response1 = client.chat.completions.create(
        model="gpt-4o-mini",
        messages=memory.get_history()
    )
    memory.add_message("assistant", response1.choices[0].message.content)

    print("\nUser: My favorite color is blue.")
    print(f"Assistant: {response1.choices[0].message.content}")

    # Clear memory
    print("\n--- Clearing memory ---")
    memory.clear()

    # New conversation - LLM won't remember
    memory.add_message("user", "What's my favorite color?")
    response2 = client.chat.completions.create(
        model="gpt-4o-mini",
        messages=memory.get_history()
    )

    print("\nUser: What's my favorite color?")
    print(f"Assistant: {response2.choices[0].message.content}")
    print("\n⚠️  Memory was cleared - LLM doesn't remember!")


if __name__ == "__main__":
    basic_memory_example()
    without_memory_example()
    system_instructions_example()
    math_conversation()
    clear_memory_example()
