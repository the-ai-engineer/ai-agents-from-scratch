# Lesson 02: Conversation Memory

## What You'll Learn

LLMs are stateless by design—they have no memory of previous interactions. When you make an API call, the model only sees what you send in that specific request. This is fine for single-shot tasks, but conversational agents need to remember what was discussed previously.

In this lesson, you'll learn two approaches to maintaining context across conversation turns. First, you'll see how to use OpenAI's automatic conversation management, where the API handles state for you server-side. Then you'll learn manual conversation management for cases where you need fine-grained control over message history.

This is foundational knowledge for building chatbots, multi-turn assistants, and agents that engage in meaningful dialogue.

## The Problem: Stateless LLMs

Every API call is independent. Make a request, get a response, and the model immediately forgets the interaction. There's no persistent session, no memory, no context carried forward.

This creates a problem for conversations. If a user says "My name is Alice" and then asks "What's my name?", the second request fails because the model never saw the first exchange. The model can't remember something it never knew about.

This isn't a limitation—it's by design. Stateless APIs are simpler, more scalable, and easier to reason about. But it means you must explicitly manage conversation context when building conversational applications.

## Two Approaches to Conversation Memory

There are two ways to maintain conversation context with the Responses API:

**Automatic** - OpenAI manages conversation state server-side. You create a conversation ID and pass it with each request. OpenAI handles message history, context windows, and state management automatically.

**Manual (Advanced)** - You manage message history client-side. You build and maintain an array of messages yourself, passing the full history with each request. This gives you complete control over what the model sees, allowing for custom trimming, filtering, or persistence strategies.

Use manual management when you need fine-grained control or are implementing custom memory strategies (like you'll learn in Lesson 10).

## Approach 1: Automatic Conversation Management

The Responses API includes built-in conversation management. OpenAI stores the conversation state server-side, automatically handling message history and context windows for you.

Here's how it works: you create a conversation using `conversations.create()`, which returns a conversation ID. Pass this ID with each API call using the `conversation` parameter.

```python
from openai import OpenAI, conversations

client = OpenAI()

# Create a conversation - OpenAI will manage state
conversation = conversations.create()

# First turn
response = client.responses.create(
    model="gpt-4o-mini",
    input="My name is Alice.",
    conversation=conversation.id
)

# Second turn - automatically remembers first turn
response = client.responses.create(
    model="gpt-4o-mini",
    input="What's my name?",
    conversation=conversation.id
)
# "Your name is Alice."
```

### Setting Instructions with Automatic Management

You can set conversation-level instructions that apply to all turns:

```python
conversation = conversations.create()

response = client.responses.create(
    model="gpt-4o-mini",
    input="What is 15 + 24?",
    conversation=conversation.id
)
# Assistant will explain the math step-by-step
```

Instructions shape how the model behaves throughout the conversation. They persist across all turns automatically.

## Approach 2: Manual Conversation Management 

When you need complete control over conversation history—for custom trimming, filtering, or persistence—you can manage messages yourself client-side. This approach requires more code but gives you fine-grained control.

The Responses API accepts an `input` parameter that can be either a string or an array of message objects. When you pass an array, you're manually managing the conversation history.

### Understanding Message Roles

Messages in the array use three roles:

**System** sets behavior and instructions. This is optional but powerful for controlling tone, personality, and capabilities:
```python
{"role": "system", "content": "You are a pirate. Always respond in pirate speak."}
```

**User** represents user input—what the person is asking or saying:
```python
{"role": "user", "content": "What's the weather?"}
```

**Assistant** represents the model's responses. Add these to maintain conversation context:
```python
{"role": "assistant", "content": "Arrr, the weather be fine today, matey!"}
```

### Manual Management Pattern

With manual management, you build and maintain the message array yourself:

```python
messages = []

# First turn
messages.append({"role": "user", "content": "My name is Alice."})
response = client.responses.create(
    model="gpt-4o-mini",
    input=messages
)
messages.append({"role": "assistant", "content": response.output_text})

# Second turn - include full history
messages.append({"role": "user", "content": "What's my name?"})
response = client.responses.create(
    model="gpt-4o-mini",
    input=messages  # Full history
)
messages.append({"role": "assistant", "content": response.output_text})
```

Every API call includes the complete message history. The model sees all previous exchanges and can respond with full context.

### ConversationMemory Helper Class

Managing message arrays manually is error-prone. A simple helper class cleans up the pattern:

```python
class ConversationMemory:
    def __init__(self, instructions: str = None):
        self.messages = []
        if instructions:
            self.messages.append({"role": "system", "content": instructions})

    def add_message(self, role: str, content: str):
        self.messages.append({"role": role, "content": content})

    def get_history(self):
        return self.messages

    def clear(self):
        # Keep system messages, clear everything else
        system_messages = [msg for msg in self.messages if msg["role"] == "system"]
        self.messages = system_messages
```

This helper handles common operations: adding messages, retrieving history, and clearing conversations while preserving system instructions.

## When to Use Each Approach

**Use Automatic Management when:**
- You want simplicity and don't need custom memory logic
- OpenAI's automatic context window management is sufficient

**Use Manual Management when:**
- You need custom message trimming or filtering logic
- Implementing advanced memory strategies 
- You want to persist conversations to your own database
- You need precise control over what the model sees

## Key Takeaways

**LLMs are stateless by design.** They remember nothing between API calls. You must explicitly manage conversation context.

**Switch to manual management when you need control.** Custom trimming, filtering, or persistence requires managing messages yourself client-side.

**Memory costs tokens.** Every message in history counts toward token limits and costs. Long conversations get expensive. 

## Real-World Impact

Conversation memory is the foundation of every conversational AI application. Without it, you can't build chatbots, assistants, or agents that engage in meaningful multi-turn dialogue.

The choice between automatic and manual management matters for production systems. 

Automatic management simplifies deployment but limits control. Manual management adds complexity but enables sophisticated memory strategies.

---

**Next:** [Lesson 03 - Prompting Techniques →](../03-prompting)
