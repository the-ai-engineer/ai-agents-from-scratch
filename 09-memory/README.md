# Lesson 09: Memory and State Management for Long-Running Agents

## What You'll Learn

In this lesson, you'll master conversation history management, handle token limits gracefully, and persist agent state for long-running applications.

Every agent you've built so far keeps conversation history in memory. That works for demos. But production agents face real constraints: conversation history grows unbounded, token limits get exceeded, servers restart and lose state, and users expect multi-session continuity.

By the end of this lesson, you'll understand token counting and model limits, implement history trimming strategies, build conversation summarization systems, persist state to disk or databases, and manage context windows intelligently.

This is what separates toy agents from production systems. Real agents must handle memory management.

## The Problem

Your agent works great for the first 10 messages. Then users keep chatting. After 50 messages, your API calls start failing with cryptic token limit errors. After 100 messages, you're burning $2 per conversation on wasted context.

Real scenarios that break naive implementations:
- Customer support sessions lasting 30+ minutes
- Research agents that iterate through dozens of steps
- Code review agents analyzing multiple files
- Long debugging conversations with complex state

Without memory management, your agent either crashes or becomes prohibitively expensive.

## API Note: Tool Calling with Chat Completions

**Important:** This lesson uses the Chat Completions API (`client.chat.completions.create()`) because the examples include tool calling, which requires the `tools` parameter and message-based conversation format. The Responses API does not yet support tool calling in the same way.

When OpenAI adds tool calling support to the Responses API, the Agent class can be updated to use the simpler `client.responses.create()` interface. For now, agents with tool calling must use Chat Completions.

## How Memory Management Works

Every LLM has a context window—the maximum tokens it can process in a single request. That includes:
- System prompt
- All conversation history
- Tool schemas
- Current user message
- Response tokens

When you exceed this limit, the API rejects your request.

### Token Limits by Model (as of 2024)

| Model | Context Window | Input Cost (per 1M) | Output Cost (per 1M) |
|-------|----------------|---------------------|----------------------|
| GPT-4o | 128,000 tokens | $2.50 | $10.00 |
| GPT-4o-mini | 128,000 tokens | $0.15 | $0.60 |
| GPT-3.5-turbo | 16,385 tokens | $0.50 | $1.50 |

A typical conversation message is 50-200 tokens. Tool calls add 100-500 tokens each. Long research tasks easily hit 10,000+ tokens.

### The Core Challenge

You must keep context under the limit while preserving enough history for coherent conversation. Too aggressive trimming makes the agent forget important context. Too conservative trimming hits limits and fails.

## Strategy 1: Token Counting

Before implementing any trimming strategy, you need accurate token counting.

### Using tiktoken

OpenAI uses byte-pair encoding (BPE) for tokenization. The `tiktoken` library provides exact token counts:

```python
import tiktoken


def count_tokens(messages: list[dict], model: str = "gpt-4o-mini") -> int:
    """
    Count tokens in a conversation for a specific model.

    Args:
        messages: List of message dictionaries
        model: Model name for tokenizer selection

    Returns:
        Total token count
    """
    try:
        encoding = tiktoken.encoding_for_model(model)
    except KeyError:
        # Fall back to cl100k_base for newer models
        encoding = tiktoken.get_encoding("cl100k_base")

    num_tokens = 0

    for message in messages:
        # Every message follows <|im_start|>{role}\n{content}<|im_end|>\n
        num_tokens += 4  # Message formatting overhead

        for key, value in message.items():
            if isinstance(value, str):
                num_tokens += len(encoding.encode(value))
            elif key == "tool_calls" and value:
                # Tool calls add significant tokens
                import json
                num_tokens += len(encoding.encode(json.dumps(value)))

    num_tokens += 2  # Every reply is primed with <|im_start|>assistant

    return num_tokens
```

### Integrating Token Counting into Agent

Add token tracking to your Agent class:

```python
class Agent:
    def __init__(self, model="gpt-4o-mini", max_context_tokens=100000):
        self.client = OpenAI()
        self.model = model
        self.max_context_tokens = max_context_tokens  # Leave buffer for response
        self.conversation_history = []

    def _count_tokens(self) -> int:
        """Count tokens in current conversation history."""
        return count_tokens(self.conversation_history, self.model)

    def chat(self, message: str) -> str:
        self.conversation_history.append({
            "role": "user",
            "content": message
        })

        # Check token count before API call
        current_tokens = self._count_tokens()
        if current_tokens > self.max_context_tokens:
            self._trim_history()

        # Rest of agent loop...
        # Note: Uses chat.completions.create for tool calling support
```

Always count tokens BEFORE making API calls. Catching errors after is too late.

## Strategy 2: Simple History Trimming

The simplest approach: remove oldest messages when you hit a limit.

```python
def _trim_history(self):
    """
    Remove oldest messages to stay under token limit.
    Always preserve system prompt.
    """
    # Keep system messages (usually first message)
    system_messages = [
        msg for msg in self.conversation_history
        if msg["role"] == "system"
    ]

    # Get all other messages
    other_messages = [
        msg for msg in self.conversation_history
        if msg["role"] != "system"
    ]

    # Remove oldest messages until under limit
    while self._count_tokens() > self.max_context_tokens and len(other_messages) > 1:
        # Remove oldest non-system message
        other_messages.pop(0)

        # Rebuild conversation history
        self.conversation_history = system_messages + other_messages

    if self._count_tokens() > self.max_context_tokens:
        raise ValueError(
            f"Cannot trim conversation below {self.max_context_tokens} tokens. "
            "Consider using summarization or reducing max_context_tokens."
        )
```

This works for most cases but loses potentially important early context.

## Strategy 3: Sliding Window with Important Message Retention

Better approach: keep recent messages plus important ones.

```python
def _trim_history_smart(self, keep_recent: int = 10):
    """
    Keep recent messages and important ones (system prompts, tool results).

    Args:
        keep_recent: Number of recent message pairs to always keep
    """
    # Always keep system messages
    system_messages = [
        msg for msg in self.conversation_history
        if msg["role"] == "system"
    ]

    # Separate messages by type
    other_messages = [
        msg for msg in self.conversation_history
        if msg["role"] != "system"
    ]

    # Keep last N messages regardless of tokens
    recent_messages = other_messages[-keep_recent*2:] if len(other_messages) > keep_recent*2 else other_messages

    # Rebuild with system + recent
    self.conversation_history = system_messages + recent_messages

    # If still over limit, remove oldest from recent messages
    while self._count_tokens() > self.max_context_tokens and len(recent_messages) > 2:
        recent_messages.pop(0)
        self.conversation_history = system_messages + recent_messages
```

This preserves recent context while dropping old history.

## Strategy 4: Conversation Summarization

For very long conversations, summarize old messages instead of deleting them.

```python
def _summarize_and_trim(self):
    """
    Summarize old messages and replace them with a condensed version.
    """
    # Keep system messages
    system_messages = [
        msg for msg in self.conversation_history
        if msg["role"] == "system"
    ]

    # Get messages to summarize (everything except recent N)
    keep_recent = 6  # Keep last 3 exchanges
    all_messages = [
        msg for msg in self.conversation_history
        if msg["role"] != "system"
    ]

    if len(all_messages) <= keep_recent:
        return  # Not enough history to summarize

    messages_to_summarize = all_messages[:-keep_recent]
    recent_messages = all_messages[-keep_recent:]

    # Generate summary
    summary_prompt = self._build_summary_prompt(messages_to_summarize)

    # Note: Using chat.completions for consistency with tool calling agents
    summary_response = self.client.chat.completions.create(
        model="gpt-4o-mini",  # Use cheaper model for summarization
        messages=[{"role": "user", "content": summary_prompt}],
        temperature=0
    )

    summary = summary_response.choices[0].message.content

    # Replace old messages with summary
    summary_message = {
        "role": "system",
        "content": f"Previous conversation summary:\n{summary}"
    }

    self.conversation_history = system_messages + [summary_message] + recent_messages


def _build_summary_prompt(self, messages: list[dict]) -> str:
    """Create prompt for summarizing conversation history."""
    conversation_text = "\n".join([
        f"{msg['role']}: {msg.get('content', '[tool call]')}"
        for msg in messages
    ])

    return f"""Summarize this conversation concisely. Focus on:
- Key topics discussed
- Important decisions or conclusions
- Relevant context for future messages

Keep it under 200 tokens.

Conversation:
{conversation_text}

Summary:"""
```

Summarization costs an extra API call but preserves more context than simple trimming.

### When to Use Each Strategy

| Strategy | Best For | Pros | Cons |
|----------|----------|------|------|
| **Simple Trimming** | Short sessions, quick prototypes | Fast, no extra API calls | Loses context |
| **Sliding Window** | Most production use cases | Balances recency and performance | May lose important early context |
| **Summarization** | Long research sessions, debugging | Preserves more information | Extra API calls, slight latency |

Default to sliding window for most applications.

## Strategy 5: State Persistence

For multi-session agents, persist conversation history to disk or database.

```python
import json
from pathlib import Path


class PersistentAgent(Agent):
    """Agent that can save and load conversation state."""

    def save_conversation(self, filepath: str | Path):
        """
        Save conversation history to a JSON file.

        Args:
            filepath: Path to save conversation
        """
        filepath = Path(filepath)

        # Ensure directory exists
        filepath.parent.mkdir(parents=True, exist_ok=True)

        # Convert conversation to JSON-serializable format
        serializable_history = []
        for msg in self.conversation_history:
            # Handle OpenAI message objects
            if hasattr(msg, 'model_dump'):
                serializable_history.append(msg.model_dump())
            else:
                serializable_history.append(msg)

        with open(filepath, 'w') as f:
            json.dump({
                'model': self.model,
                'conversation_history': serializable_history,
                'tool_schemas': self.tool_schemas
            }, f, indent=2)

    def load_conversation(self, filepath: str | Path):
        """
        Load conversation history from a JSON file.

        Args:
            filepath: Path to load conversation from
        """
        filepath = Path(filepath)

        if not filepath.exists():
            raise FileNotFoundError(f"No conversation found at {filepath}")

        with open(filepath, 'r') as f:
            data = json.load(f)

        self.model = data.get('model', self.model)
        self.conversation_history = data['conversation_history']
        self.tool_schemas = data.get('tool_schemas', [])

    def auto_save(self, save_dir: str | Path):
        """
        Enable automatic saving after each message.

        Args:
            save_dir: Directory to save conversations
        """
        self.save_dir = Path(save_dir)
        self.save_dir.mkdir(parents=True, exist_ok=True)
        self.auto_save_enabled = True

    def chat(self, message: str) -> str:
        """Override chat to add auto-save functionality."""
        response = super().chat(message)

        # Auto-save if enabled
        if hasattr(self, 'auto_save_enabled') and self.auto_save_enabled:
            save_path = self.save_dir / f"conversation_{len(self.conversation_history)}.json"
            self.save_conversation(save_path)

        return response
```

### Database Persistence for Production

For production systems, use a database instead of JSON files:

```python
# Pseudocode for database persistence
class DatabaseAgent(Agent):
    def __init__(self, session_id: str, db_connection):
        super().__init__()
        self.session_id = session_id
        self.db = db_connection

        # Load existing conversation if available
        self._load_from_db()

    def _load_from_db(self):
        """Load conversation history from database."""
        result = self.db.query(
            "SELECT conversation_history FROM agent_sessions WHERE session_id = ?",
            (self.session_id,)
        )
        if result:
            self.conversation_history = json.loads(result[0]['conversation_history'])

    def _save_to_db(self):
        """Save conversation history to database."""
        self.db.execute(
            "INSERT OR REPLACE INTO agent_sessions (session_id, conversation_history, updated_at) VALUES (?, ?, ?)",
            (self.session_id, json.dumps(self.conversation_history), datetime.now())
        )

    def chat(self, message: str) -> str:
        response = super().chat(message)
        self._save_to_db()
        return response
```

## Running the Example

This lesson includes implementations of all memory strategies:

```bash
cd 09-memory
uv run example.py
```

The example demonstrates:
- Token counting with tiktoken
- Simple history trimming
- Sliding window with recent message retention
- Conversation summarization
- Saving and loading conversations

Try simulating long conversations to see each strategy in action.

## Key Takeaways

1. **Always monitor token usage.** Track tokens with every API call. Log warnings when approaching limits.

2. **Trim proactively, not reactively.** Don't wait for API errors. Trim when you hit 80% of your limit.

3. **Never trim system prompts.** System messages define agent behavior. Removing them breaks everything.

4. **Keep recent context.** Sliding window strategies work well for most use cases. Recent messages matter most.

5. **Test with long conversations.** Most bugs appear after 20+ turns. Don't just test happy paths with 3 messages.

## Common Pitfalls

1. **Not counting tokens**: Guessing at token usage leads to API errors. Always measure with tiktoken.

2. **Trimming system messages**: Accidentally removing system prompts breaks agent behavior. Always preserve them.

3. **No buffer for responses**: Setting max_context_tokens = model limit doesn't leave room for the response. Use 80-90% of the limit.

4. **Over-aggressive trimming**: Removing too much context makes the agent forget what you're talking about.

5. **Not testing long sessions**: Five-message demos work fine. Fifty-message sessions reveal memory issues.

6. **Forgetting tool schemas**: Tool schemas consume tokens too. Count them when calculating limits.

7. **Synchronous saves**: Saving to disk/database on every message blocks the response. Use async saves in production.

## Real-World Impact

Memory management is critical for production agents. Without it, you face:

**Cost overruns**: One company's RAG chatbot was sending 50,000 tokens per request (including full conversation history every time). They were paying $500/day unnecessarily. After implementing sliding window trimming, costs dropped to $50/day.

**API failures**: Agents that hit token limits fail silently or crash. Users see cryptic errors. Sessions end abruptly.

**Poor UX**: Agents that can't remember earlier conversation context frustrate users. "I already told you my order number!"

**Scaling issues**: Without persistence, every server restart loses all agent state. Users can't resume sessions.

With proper memory management:
- 80-90% reduction in token usage
- No API limit errors
- Seamless multi-session experiences
- Predictable costs at scale

## Assignment

Enhance your Agent class with comprehensive memory management:

1. **Add token counting** using tiktoken. Log token usage with each API call.

2. **Implement sliding window trimming** that keeps the last 10 messages plus system prompts.

3. **Add conversation persistence** with save/load methods.

4. **Create a test script** that simulates a 30-message conversation and logs:
   - Token count after each message
   - When trimming happens
   - What gets removed

5. **Test all strategies**:
   - Simple trimming
   - Sliding window
   - Summarization

Compare token usage and response quality across strategies.

## Next Steps

You now have a complete, production-ready agent architecture with memory management. Next, explore [Lesson 10 - RAG Integration](../10-rag-integration) to add knowledge retrieval capabilities to your agents.

## Resources

- [tiktoken Documentation](https://github.com/openai/tiktoken)
- [OpenAI Token Limits](https://platform.openai.com/docs/models/gpt-4-and-gpt-4-turbo)
- [Understanding Context Windows](https://help.openai.com/en/articles/4936856-what-are-tokens-and-how-to-count-them)
- [Conversation Memory Patterns](https://python.langchain.com/docs/modules/memory/) - LangChain's memory implementations
- [Redis for Session Storage](https://redis.io/docs/manual/persistence/) - Production persistence strategies
