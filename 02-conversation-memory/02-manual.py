"""
Lesson 02: Conversation Memory

Learn how to maintain context across multiple turns in a conversation.
"""

from openai import OpenAI
from dotenv import load_dotenv

load_dotenv()

client = OpenAI()


def call_llm(input, temperature=0):
    response = client.responses.create(
        model="gpt-4o-mini", input=input, temperature=temperature
    )
    return response


##=================================================##
## Example 1: What happens WITHOUT memory
##=================================================##

# First call
response1 = call_llm("My name is Alice.")

print("User: My name is Alice.")
print(f"Assistant: {response1.output_text}\n")

# Second call - doesn't include previous exchange
response2 = call_llm("What's my name?")

print("User: What's my name?")
print(f"Assistant: {response2.output_text}")

##=================================================##
## Example 2: Basic conversation WITH memory
##=================================================##

# Keep track of the conversation history
messages = []

# First turn
messages.append({"role": "user", "content": "My name is Owain."})
response1 = call_llm(messages)
messages.append({"role": "assistant", "content": response1.output_text})

print("User: My name is Owain.")
print(f"Assistant: {response1.output_text}\n")

# Second turn - LLM remembers because we pass full history
messages.append({"role": "user", "content": "What's my name?"})
response2 = call_llm(messages)
messages.append({"role": "assistant", "content": response2.output_text})

print("User: What's my name?")
print(f"Assistant: {response2.output_text}\n")

##=================================================##
## Example 3: Using system instructions
##=================================================##

messages = [
    {"role": "system", "content": "You are a pirate. Always respond in pirate speak."}
]

conversations = [
    "What's the weather like?",
    "Can you help me with math?",
]

for user_msg in conversations:
    messages.append({"role": "user", "content": user_msg})
    response = call_llm(messages, temperature=0.8)
    messages.extend(response.output)

    print(f"User: {user_msg}")
    print(f"Pirate: {response.output_text}\n")


class ConversationMemory:
    """Simple helper to manage conversation history"""

    def __init__(self, instructions: str = None):
        """Initialize with optional system instructions"""
        self.messages = []
        if instructions:
            self.messages.append({"role": "system", "content": instructions})

    def add_message(self, role: str, content: str):
        """Add a message to history"""
        self.messages.append({"role": role, "content": content})

    def get_history(self):
        """Get full conversation history"""
        return self.messages

    def clear(self):
        """Clear all messages except system message"""
        system_messages = [msg for msg in self.messages if msg["role"] == "system"]
        self.messages = system_messages


def call_llm_with_memory(memory, user_message, temperature=0):
    """Helper to add message, call LLM, and store response"""
    memory.add_message("user", user_message)
    response = call_llm(memory.get_history(), temperature)
    memory.add_message("assistant", response.output_text)
    return response


##=================================================##
## Example 4: Using the helper class and function
##=================================================##

# Create conversation with system instructions
memory = ConversationMemory(instructions="You are a friendly math tutor.")

# Much cleaner with the helper function!
response1 = call_llm_with_memory(memory, "I have 10 apples.")
print("User: I have 10 apples.")
print(f"Assistant: {response1.output_text}\n")

response2 = call_llm_with_memory(memory, "I buy 5 more apples.")
print("User: I buy 5 more apples.")
print(f"Assistant: {response2.output_text}\n")

response3 = call_llm_with_memory(memory, "How many apples do I have now?")
print("User: How many apples do I have now?")
print(f"Assistant: {response3.output_text}\n")
