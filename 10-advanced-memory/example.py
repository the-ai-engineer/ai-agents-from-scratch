"""
Lesson 07: Memory and State Management

Learn how to manage conversation history and handle token limits.
Focus on practical patterns for production agents.
"""

import json
import tiktoken
import redis
import json
import numpy as np
from typing import Tuple

from typing import List, Dict, Any
from openai import OpenAI
from dotenv import load_dotenv


load_dotenv()

client = OpenAI()


##=================================================##
## Token Management
##=================================================##


class ConversationMemory:
    """Manage conversation history with token limits."""

    def __init__(self, max_tokens: int = 3000, model: str = "gpt-4o-mini"):
        self.max_tokens = max_tokens
        self.messages = []
        self.encoder = tiktoken.encoding_for_model(model)

    def count_tokens(self, messages: List[Dict] = None) -> int:
        """Count tokens in messages."""
        if messages is None:
            messages = self.messages

        tokens = 0
        for msg in messages:
            if "content" in msg:
                tokens += len(self.encoder.encode(msg["content"]))
            tokens += 4  # Message overhead
        return tokens

    def add(self, role: str, content: str):
        """Add message and trim if needed."""
        self.messages.append({"role": role, "content": content})
        self._trim_if_needed()

    def _trim_if_needed(self):
        """Remove old messages to stay under token limit."""
        while self.count_tokens() > self.max_tokens and len(self.messages) > 1:
            # Keep system message if present, remove oldest user/assistant message
            if self.messages[0].get("role") == "system":
                self.messages.pop(1)
            else:
                self.messages.pop(0)

    def get_messages(self) -> List[Dict]:
        """Get all messages for API call."""
        return self.messages


# Example usage:
memory = ConversationMemory(max_tokens=500)
memory.add("system", "You are a helpful assistant.")
memory.add("user", "Tell me about Paris")
memory.add("assistant", "Paris is the capital of France...")

# Check token usage
# memory.count_tokens()
# memory.get_messages()


##=================================================##
## Conversation Persistence
##=================================================##


class PersistentAgent:
    """Agent with save/load capabilities."""

    def __init__(self, session_id: str = "default"):
        self.session_id = session_id
        self.memory = ConversationMemory()
        self.client = OpenAI()

    def chat(self, message: str) -> str:
        """Send message and get response."""
        self.memory.add("user", message)

        response = self.client.chat.completions.create(
            model="gpt-4o-mini", messages=self.memory.get_messages()
        )

        reply = response.choices[0].message.content
        self.memory.add("assistant", reply)
        return reply

    def save(self):
        """Save conversation to file."""
        filename = f"session_{self.session_id}.json"
        with open(filename, "w") as f:
            json.dump(self.memory.messages, f)
        return filename

    def load(self):
        """Load conversation from file."""
        filename = f"session_{self.session_id}.json"
        try:
            with open(filename, "r") as f:
                self.memory.messages = json.load(f)
            return True
        except FileNotFoundError:
            return False


# Example: Conversation across sessions
agent = PersistentAgent("user_123")
agent.chat("My name is Alice")
agent.chat("I'm learning Python")
agent.save()

# Later (new session)
agent2 = PersistentAgent("user_123")
agent2.load()
# agent2.chat("What's my name?")  # Will remember Alice


##=================================================##
## Redis for Production (Optional)
##=================================================##


class RedisMemory:
    """Production-ready memory with Redis backend."""

    def __init__(self, session_id: str, ttl: int = 86400):
        self.session_id = session_id
        self.ttl = ttl  # Time to live in seconds (default 24h)
        self.redis = redis.Redis(decode_responses=True)
        self.key = f"chat:{session_id}"

    def save(self, messages: List[Dict]):
        """Save messages to Redis with TTL."""
        self.redis.setex(self.key, self.ttl, json.dumps(messages))

    def load(self) -> List[Dict]:
        """Load messages from Redis."""
        data = self.redis.get(self.key)
        return json.loads(data) if data else []

    def extend_ttl(self):
        """Extend session TTL on activity."""
        self.redis.expire(self.key, self.ttl)


# Usage (requires Redis running):
# memory = RedisMemory("user_123")
# memory.save([{"role": "user", "content": "Hello"}])
# messages = memory.load()


##=================================================##
## Semantic Memory with Embeddings
##=================================================##


class SemanticMemory:
    """
    Store and retrieve memories by semantic similarity.
    Simplified version - production would use vector DB.
    """

    def __init__(self):
        self.memories = []
        self.embeddings = []
        self.client = OpenAI()

    def _get_embedding(self, text: str) -> List[float]:
        """Get embedding for text."""
        response = self.client.embeddings.create(
            model="text-embedding-3-small", input=text
        )
        return response.data[0].embedding

    def add(self, memory: str):
        """Add memory with its embedding."""
        embedding = self._get_embedding(memory)
        self.memories.append(memory)
        self.embeddings.append(embedding)

    def search(self, query: str, top_k: int = 3) -> List[Tuple[str, float]]:
        """Find most relevant memories for query."""
        if not self.memories:
            return []

        # Get query embedding
        query_embedding = np.array(self._get_embedding(query))

        # Calculate similarities
        similarities = []
        for i, emb in enumerate(self.embeddings):
            similarity = np.dot(query_embedding, emb)
            similarities.append((self.memories[i], similarity))

        # Return top results
        similarities.sort(key=lambda x: x[1], reverse=True)
        return similarities[:top_k]


# Example: Context-aware retrieval
semantic = SemanticMemory()
semantic.add("User prefers Python for data science")
semantic.add("User is working on a recommendation system")
semantic.add("User's budget is $10,000")

# Query relevant context
# results = semantic.search("What programming language should I use?")
# for memory, score in results:
#     print(f"{score:.3f}: {memory}")


##=================================================##
## Practical Example: Customer Support Bot
##=================================================##


class SupportBot:
    """Customer support bot with memory management."""

    def __init__(self, customer_id: str):
        self.customer_id = customer_id
        self.memory = ConversationMemory(max_tokens=1000)
        self.context = SemanticMemory()
        self.client = OpenAI()

        # Load customer context
        self._load_customer_context()

    def _load_customer_context(self):
        """Load relevant customer data."""
        # In production, load from database
        self.context.add(f"Customer {self.customer_id} has premium account")
        self.context.add(f"Customer previously had shipping issues")
        self.context.add(f"Customer prefers email communication")

    def handle_query(self, query: str) -> str:
        """Handle customer query with context."""
        # Find relevant context
        relevant_context = self.context.search(query, top_k=2)
        context_str = "\n".join([m for m, _ in relevant_context])

        # Build prompt with context
        prompt = f"""Customer context:
{context_str}

Current query: {query}

Provide helpful response."""

        self.memory.add("user", prompt)

        response = self.client.chat.completions.create(
            model="gpt-4o-mini", messages=self.memory.get_messages()
        )

        reply = response.choices[0].message.content
        self.memory.add("assistant", reply)

        return reply


# Example:
# bot = SupportBot("customer_123")
# bot.handle_query("My order hasn't arrived yet")
# bot.handle_query("Can you send me an update?")
