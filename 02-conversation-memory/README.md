# Lesson 02: Conversation Memory

## What You'll Learn

LLMs are **stateless**—they don't remember previous interactions. Every API call is independent. To build conversational agents that remember context, you need to manage conversation history yourself.

In this lesson, you'll learn how to maintain context across multiple turns using a simple `ConversationMemory` helper class. This is the foundation for building chatbots, assistants, and any agent that needs to maintain dialogue context.

## The Problem: Stateless LLMs

```python
# First call
messages1 = [{"role": "user", "content": "My name is Alice."}]
response1 = client.chat.completions.create(model="gpt-4o-mini", messages=messages1)
# Assistant: "Nice to meet you, Alice!"

# Second call - WITHOUT including first exchange
messages2 = [{"role": "user", "content": "What's my name?"}]
response2 = client.chat.completions.create(model="gpt-4o-mini", messages=messages2)
# Assistant: "I don't know your name."
```

**The LLM forgot!** Each API call is independent. The second call has no knowledge of the first.

## The Solution: Conversation History

The Chat Completions API accepts a `messages` array. By including the entire conversation history in each call, the LLM "remembers" previous exchanges.

```python
# Include FULL history in second call
messages = [
    {"role": "user", "content": "My name is Alice."},
    {"role": "assistant", "content": "Nice to meet you, Alice!"},
    {"role": "user", "content": "What's my name?"}
]
response = client.chat.completions.create(model="gpt-4o-mini", messages=messages)
# Assistant: "Your name is Alice."
```

The LLM sees the full context and can respond appropriately.

## ConversationMemory Helper

Managing message arrays manually is error-prone. The `ConversationMemory` class provides a clean interface:

```python
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
```

## Usage Pattern

```python
memory = ConversationMemory()

# Turn 1
memory.add_message("user", "My name is Alice.")
response = client.chat.completions.create(
    model="gpt-4o-mini",
    messages=memory.get_history()
)
memory.add_message("assistant", response.choices[0].message.content)

# Turn 2 - LLM has full context
memory.add_message("user", "What's my name?")
response = client.chat.completions.create(
    model="gpt-4o-mini",
    messages=memory.get_history()  # Includes all previous messages
)
memory.add_message("assistant", response.choices[0].message.content)
```

## Message Roles

The `messages` array uses three roles:

### 1. System
Sets the assistant's behavior and instructions:

```python
memory = ConversationMemory(
    instructions="You are a pirate. Always respond in pirate speak."
)
```

System messages are optional but powerful for controlling tone, personality, and capabilities.

### 2. User
Represents user input:

```python
memory.add_message("user", "What's the weather?")
```

### 3. Assistant
Represents LLM responses:

```python
memory.add_message("assistant", response.choices[0].message.content)
```

**Important:** Always add the assistant's response to history after each API call.

## System Instructions

System messages set behavior for the entire conversation:

```python
memory = ConversationMemory(
    instructions="You are a helpful math tutor. Show your work step by step."
)

memory.add_message("user", "What is 15 + 24?")
response = client.chat.completions.create(
    model="gpt-4o-mini",
    messages=memory.get_history()
)
# Assistant will explain the math step-by-step
```

**Best Practices:**
- Use system messages for personality, tone, and capabilities
- Keep system messages concise and clear
- System messages persist across `clear()` calls

## Multi-Turn Conversations

The power of conversation memory is multi-turn reasoning:

```python
memory = ConversationMemory()

# Turn 1: Set context
memory.add_message("user", "I have 15 apples.")
response = client.chat.completions.create(model="gpt-4o-mini", messages=memory.get_history())
memory.add_message("assistant", response.choices[0].message.content)

# Turn 2: Add information
memory.add_message("user", "I buy 24 more apples.")
response = client.chat.completions.create(model="gpt-4o-mini", messages=memory.get_history())
memory.add_message("assistant", response.choices[0].message.content)

# Turn 3: Ask question that requires full context
memory.add_message("user", "How many apples do I have now?")
response = client.chat.completions.create(model="gpt-4o-mini", messages=memory.get_history())
# Assistant: "You have 39 apples (15 + 24 = 39)."
```

The LLM sees the full conversation and can track state across multiple turns.

## Clearing Memory

Use `clear()` to start a fresh conversation:

```python
memory = ConversationMemory(instructions="You are a helpful assistant.")

# First conversation
memory.add_message("user", "My favorite color is blue.")
# ... conversation continues ...

# Start fresh
memory.clear()  # Keeps system message, removes everything else

# New conversation
memory.add_message("user", "What's my favorite color?")
# Assistant won't remember - memory was cleared
```

## Key Concepts

1. **LLMs are stateless** - They don't remember previous calls
2. **Pass full history** - Include all previous messages in each API call
3. **Always add assistant responses** - Don't forget to add the LLM's reply to history
4. **System messages persist** - They set behavior for the entire conversation
5. **Memory costs tokens** - Each message in history counts toward token limits (we'll handle this in Lesson 10)

## When to Use Conversation Memory

✅ **Use conversation memory when:**
- Building chatbots or conversational interfaces
- Multi-turn problem solving (math, coding, analysis)
- Tracking context across user interactions
- Building agents that need dialogue history

❌ **Don't use conversation memory when:**
- Single-shot tasks (one question, one answer)
- Independent batch processing
- Tasks where context doesn't matter

## Common Mistakes

### Mistake 1: Forgetting to add assistant response

```python
# ❌ Wrong
memory.add_message("user", "Hello")
response = client.chat.completions.create(messages=memory.get_history())
# Forgot to add assistant response!

memory.add_message("user", "How are you?")
# LLM won't see its previous response

# ✅ Correct
memory.add_message("user", "Hello")
response = client.chat.completions.create(messages=memory.get_history())
memory.add_message("assistant", response.choices[0].message.content)  # Add response!

memory.add_message("user", "How are you?")
```

### Mistake 2: Creating new memory each time

```python
# ❌ Wrong
def chat(user_msg):
    memory = ConversationMemory()  # Creates fresh memory each time!
    memory.add_message("user", user_msg)
    return client.chat.completions.create(messages=memory.get_history())

# ✅ Correct
memory = ConversationMemory()  # Create once

def chat(user_msg):
    memory.add_message("user", user_msg)
    response = client.chat.completions.create(messages=memory.get_history())
    memory.add_message("assistant", response.choices[0].message.content)
    return response
```

### Mistake 3: Not handling None content

```python
# ❌ Wrong
memory.add_message("assistant", response.choices[0].message.content)
# Can crash if content is None (e.g., when only tool calls are present)

# ✅ Correct
content = response.choices[0].message.content or ""
memory.add_message("assistant", content)
```

## Running the Examples

```bash
cd 02-conversation-memory
uv run example.py
```

The examples demonstrate:
1. Basic conversation memory
2. What happens WITHOUT memory (comparison)
3. Using system instructions
4. Multi-step math problems
5. Clearing memory

## What's Next?

Now that you understand conversation memory, you can:
- **Lesson 03**: Learn prompting techniques to improve response quality
- **Lesson 04**: Add structured outputs with Pydantic
- **Lesson 10**: Learn advanced memory strategies (token limits, trimming, persistence)

## Key Takeaway

> **Conversation memory is the foundation of conversational AI.** By maintaining message history and passing it to each API call, you enable LLMs to maintain context across multiple turns. This simple pattern unlocks chatbots, assistants, and agents that can engage in meaningful dialogue.

---

**Next:** [Lesson 03 - Prompting Techniques →](../03-prompting)
