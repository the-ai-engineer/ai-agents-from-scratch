# Lesson 10: Advanced Memory - Token Limits and Persistence

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

## API Note: Using the Responses API

This lesson uses the Responses API (`client.responses.create()`) which provides a simpler interface and supports tool calling. The Responses API handles the complexity of managing tool calls and iterating until completion.

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

### Redis Persistence for Production

For production systems, Redis provides fast in-memory persistence with session management:

```python
import redis
import json

class RedisAgent(Agent):
    """Production-ready agent with Redis persistence"""

    def __init__(
        self,
        session_id: str,
        redis_url: str = "redis://localhost:6379",
        ttl: int = 86400,  # 24 hours
        **kwargs
    ):
        super().__init__(**kwargs)
        self.session_id = session_id
        self.redis = redis.from_url(redis_url, decode_responses=True)
        self.ttl = ttl
        self.key = f"agent:session:{session_id}"

        # Load existing conversation if available
        self._load_from_redis()

    def _load_from_redis(self):
        """Load conversation history from Redis"""
        data = self.redis.get(self.key)
        if data:
            self.conversation_history = json.loads(data)
            print(f"Loaded session {self.session_id} from Redis")

    def _save_to_redis(self):
        """Save conversation history to Redis with TTL"""
        self.redis.setex(
            self.key,
            self.ttl,
            json.dumps(self.conversation_history)
        )

    def chat(self, message: str) -> str:
        """Override chat to add Redis persistence"""
        response = super().chat(message)

        # Auto-save after each turn
        self._save_to_redis()

        return response
```

**Why Redis for production:**
- **Fast**: In-memory performance (microsecond latency)
- **Scalable**: Works across multiple application servers
- **Automatic expiration**: TTL handles cleanup automatically
- **Session management**: Perfect for stateless APIs (e.g., FastAPI)
- **Crash recovery**: Conversations survive server restarts

**Setup Redis:**
```bash
# Local development
docker run -d -p 6379:6379 redis:latest

# Or install locally
brew install redis  # macOS
redis-server
```

**Usage:**
```python
# Agent state persists across restarts
agent1 = RedisAgent(session_id="user_123")
agent1.chat("My name is Alice")

# Different instance, same session
agent2 = RedisAgent(session_id="user_123")
agent2.chat("What's my name?")  # Remembers "Alice"
```

### Database Persistence Alternative

For SQL/NoSQL databases, the pattern is similar:

```python
# Pseudocode for database persistence
class DatabaseAgent(Agent):
    def __init__(self, session_id: str, db_connection):
        super().__init__()
        self.session_id = session_id
        self.db = db_connection
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

## Production Alternatives: Managed Memory Services

Once you understand how memory persistence works, you can choose to use managed services that handle the infrastructure for you.

### Mem0 (https://mem0.ai/)

Managed memory layer specifically designed for AI agents with semantic search capabilities:

```python
from mem0 import Memory

# Initialize Mem0 (uses local vector store by default, or configure cloud storage)
memory = Memory()

# Add memories with user context
memory.add("User prefers Python over JavaScript", user_id="john")
memory.add("User is working on an e-commerce project", user_id="john")
memory.add("User's budget is $50,000", user_id="john")

# Retrieve relevant memories with semantic search
relevant = memory.search("What technology should I use?", user_id="john")
# Returns: ["User prefers Python over JavaScript", "User is working on an e-commerce project"]

# Get all memories for a user
all_memories = memory.get_all(user_id="john")
```

**Integrating Mem0 with Your Agent:**

```python
from mem0 import Memory
from openai import OpenAI

class AgentWithMem0:
    """Agent with semantic memory using Mem0"""

    def __init__(self, user_id: str):
        self.client = OpenAI()
        self.memory = Memory()
        self.user_id = user_id
        self.conversation_history = []

    def chat(self, message: str) -> str:
        """Chat with semantic memory retrieval"""

        # Search for relevant memories based on current message
        relevant_memories = self.memory.search(message, user_id=self.user_id)

        # Build context from relevant memories
        memory_context = "\n".join([
            f"- {mem['memory']}"
            for mem in relevant_memories[:3]  # Top 3 most relevant
        ])

        # Create system prompt with memory context
        system_prompt = f"""You are a helpful assistant with access to user's memory.

Relevant memories about this user:
{memory_context if memory_context else "No relevant memories yet."}

Use these memories to provide personalized responses."""

        # Build messages with memory context
        messages = [{"role": "system", "content": system_prompt}]
        messages.extend(self.conversation_history[-6:])  # Keep recent history
        messages.append({"role": "user", "content": message})

        # Get response
        response = self.client.chat.completions.create(
            model="gpt-4o-mini",
            messages=messages
        )

        answer = response.choices[0].message.content

        # Store this interaction in conversation history
        self.conversation_history.append({"role": "user", "content": message})
        self.conversation_history.append({"role": "assistant", "content": answer})

        # Extract and store important facts from conversation
        # (In production, you'd use a more sophisticated extraction method)
        self.memory.add(
            f"User asked: {message}. Response: {answer}",
            user_id=self.user_id
        )

        return answer

# Usage
agent = AgentWithMem0(user_id="alice")
agent.chat("I prefer Python for backend work")
agent.chat("I'm building an e-commerce site")
agent.chat("What technology should I use?")  # Retrieves relevant memories!
```

**Key Features:**
- **Semantic search**: Retrieves relevant memories, not just recent ones
- **Multi-user isolation**: Separate memory per user automatically
- **Long-term memory**: Persists across sessions indefinitely
- **Automatic relevance**: Ranks memories by importance to current context
- **No infrastructure**: Managed service (cloud) or local vector store

**Installation:**
```bash
# Install mem0
uv add mem0ai

# Run the example
uv run example.py
```

### Comparison: DIY vs Managed

| Approach | Setup | Cost | Control | Semantic Search | Best For |
|----------|-------|------|---------|-----------------|----------|
| **File-based (JSON)** | None | Free | Full | ❌ | Development, testing |
| **Redis (self-hosted)** | Medium | Server costs | Full | ❌ | Production, custom needs, full control |
| **Redis Cloud** | Low | $5-50/mo | Medium | ❌ | Production, less ops work |
| **Mem0** | None | Usage-based | Limited | ✅ | Fast MVP, semantic memory features |
| **LangChain Memory** | Medium | Free + compute | Medium | ⚠️ | If using LangChain already |

**When to use what:**

- **Learning/Prototyping:** File-based (what you built in this lesson - understand how it works)
- **Production (control):** Self-hosted Redis (you control everything, predictable costs)
- **Production (speed):** Managed Redis or Mem0 (less operational overhead)
- **Semantic features:** Mem0 or build your own with vector DB (covered in separate RAG module)
- **Enterprise:** Custom solution with your existing database infrastructure

**The Trade-off:**

After completing this lesson, you understand exactly how memory persistence works. You can then make informed decisions:
- **Build**: Use Redis/database when you need full control, custom features, or have existing infrastructure
- **Buy**: Use Mem0 when you need semantic search or want to move fast without managing infrastructure

Understanding the fundamentals means you're never locked into a vendor and can always switch approaches.

## Running the Example

This lesson includes implementations of all memory strategies:

```bash
cd 10-advanced-memory
uv run example.py
```

The example demonstrates:
- Token counting with tiktoken
- Simple history trimming
- Sliding window with recent message retention
- Conversation summarization
- Saving and loading conversations to files
- (Optional) Redis persistence if you have Redis running
- (Optional) Mem0 semantic memory for intelligent retrieval

Try simulating long conversations to see each strategy in action.

**To test Redis persistence:**
```bash
# Start Redis in Docker
docker run -d -p 6379:6379 redis:latest

# Install redis-py
uv add redis

# Run Redis examples in example.py
uv run example.py
```

**To test Mem0 semantic memory:**
```bash
# Install mem0
uv add mem0ai

# Run Mem0 examples in example.py
uv run example.py
```

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

You now have a complete, production-ready agent architecture with memory management. This completes the core fundamentals course!

For advanced topics like RAG (Retrieval-Augmented Generation), vector databases, and production deployment, check out the separate advanced modules.

## Resources

**Token Management:**
- [tiktoken Documentation](https://github.com/openai/tiktoken)
- [OpenAI Token Limits](https://platform.openai.com/docs/models/gpt-4-and-gpt-4-turbo)
- [Understanding Context Windows](https://help.openai.com/en/articles/4936856-what-are-tokens-and-how-to-count-them)

**Persistence Solutions:**
- [Redis Documentation](https://redis.io/docs/) - Fast in-memory persistence
- [Redis for Session Storage](https://redis.io/docs/manual/persistence/) - Production patterns
- [redis-py Library](https://redis-py.readthedocs.io/) - Python Redis client

**Managed Memory Services:**
- [Mem0.ai](https://mem0.ai/) - Managed memory layer for AI agents
- [Mem0 Documentation](https://docs.mem0.ai/) - API and integration guides
- [LangChain Memory](https://python.langchain.com/docs/modules/memory/) - Alternative memory implementations
