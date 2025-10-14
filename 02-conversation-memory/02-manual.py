"""
Lesson 02: Conversation Memory - Manual Management

Learn how to manage conversation history manually for full control.

Run from project root:
    uv run python 02-conversation-memory/02-manual.py
"""

from openai import OpenAI
from dotenv import load_dotenv

load_dotenv()
client = OpenAI()


##=================================================##
## Example 1: What happens WITHOUT memory
##=================================================##

# First call
response1 = client.responses.create(
    model="gpt-4o-mini",
    input="My name is Alice.",
    temperature=0
)

# response1.output_text

# Second call - doesn't include previous exchange
response2 = client.responses.create(
    model="gpt-4o-mini",
    input="What's my name?",
    temperature=0
)

# response2.output_text  # LLM doesn't know - no memory!


##=================================================##
## Example 2: Basic conversation WITH manual memory
##=================================================##

# Keep track of the conversation history manually
messages = []

# First turn
messages.append({"role": "user", "content": "My name is Alice."})
response1 = client.responses.create(
    model="gpt-4o-mini",
    input=messages,
    temperature=0
)
messages.append({"role": "assistant", "content": response1.output_text})

# response1.output_text

# Second turn - LLM remembers because we pass full history
messages.append({"role": "user", "content": "What's my name?"})
response2 = client.responses.create(
    model="gpt-4o-mini",
    input=messages,
    temperature=0
)
messages.append({"role": "assistant", "content": response2.output_text})

# response2.output_text  # "Your name is Alice."


##=================================================##
## Example 3: Using system instructions
##=================================================##

messages = [
    {"role": "system", "content": "You are a pirate. Always respond in pirate speak."}
]

# First turn
messages.append({"role": "user", "content": "What's the weather like?"})
response = client.responses.create(
    model="gpt-4o-mini",
    input=messages,
    temperature=0.8
)
messages.extend(response.output)

# response.output_text

# Second turn - maintains pirate personality
messages.append({"role": "user", "content": "Can you help me with math?"})
response = client.responses.create(
    model="gpt-4o-mini",
    input=messages,
    temperature=0.8
)
messages.extend(response.output)

# response.output_text


##=================================================##
## Example 4: ConversationMemory helper class
##=================================================##

class ConversationMemory:
    """Simple helper to manage conversation history"""

    def __init__(self, instructions: str = None):
        self.messages = []
        if instructions:
            self.messages.append({"role": "system", "content": instructions})

    def add_message(self, role: str, content: str):
        self.messages.append({"role": role, "content": content})

    def get_history(self):
        return self.messages

    def clear(self):
        """Clear all messages except system message"""
        system_messages = [msg for msg in self.messages if msg["role"] == "system"]
        self.messages = system_messages


# Create conversation with system instructions
memory = ConversationMemory(instructions="You are a friendly math tutor.")

# Turn 1
memory.add_message("user", "I have 10 apples.")
response = client.responses.create(
    model="gpt-4o-mini",
    input=memory.get_history(),
    temperature=0
)
memory.add_message("assistant", response.output_text)

# response.output_text

# Turn 2
memory.add_message("user", "I buy 5 more apples.")
response = client.responses.create(
    model="gpt-4o-mini",
    input=memory.get_history(),
    temperature=0
)
memory.add_message("assistant", response.output_text)

# response.output_text

# Turn 3 - remembers both previous turns
memory.add_message("user", "How many apples do I have now?")
response = client.responses.create(
    model="gpt-4o-mini",
    input=memory.get_history(),
    temperature=0
)
memory.add_message("assistant", response.output_text)

# response.output_text  # "15 apples"
