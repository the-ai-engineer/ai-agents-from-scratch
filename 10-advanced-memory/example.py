"""
Lesson 07: Advanced Memory and State Management

Learn how to manage conversation history, handle token limits, and persist state.

This builds on the Agent class from Lesson 06, adding:
- Token counting with tiktoken
- Automatic history trimming
- Conversation persistence (save/load)
"""

import json
import sys
import tiktoken
from pathlib import Path
from typing import Callable, Optional
from dotenv import load_dotenv

# Add parent directory to path to import agents framework
sys.path.insert(0, str(Path(__file__).parent.parent))
from src.agent import Agent as BaseAgent

load_dotenv()


class AgentWithMemory(BaseAgent):
    """
    Agent with advanced memory management.

    Extends the base Agent class with:
    - Token counting
    - Automatic history trimming
    - Conversation persistence
    """

    def __init__(
        self,
        model: str = "gpt-4o-mini",
        max_iterations: int = 5,
        max_history_tokens: int = 4000,
        system_prompt: Optional[str] = None,
        tools: Optional[list[Callable]] = None,
    ):
        """
        Initialize agent with memory management.

        Args:
            model: Which OpenAI model to use
            max_iterations: Maximum number of tool-calling loops
            max_history_tokens: Maximum tokens to keep in history
            system_prompt: System instructions for the agent
            tools: List of functions decorated with @tool
        """
        super().__init__(model, max_iterations, system_prompt, tools)
        self.max_history_tokens = max_history_tokens
        self.tokenizer = tiktoken.encoding_for_model(model)

    def _count_tokens(self, messages: list = None) -> int:
        """Count tokens in message history"""
        if messages is None:
            messages = self.memory.get_items()

        token_count = 0
        for message in messages:
            # Count content tokens
            if isinstance(message.get("content"), str):
                token_count += len(self.tokenizer.encode(message["content"]))
            elif isinstance(message.get("output"), str):
                # Count function output tokens
                token_count += len(self.tokenizer.encode(message["output"]))
            token_count += 4  # Overhead per message
        return token_count

    def _trim_history(self):
        """Trim old messages to stay under token limit"""
        current_tokens = self._count_tokens()

        if current_tokens <= self.max_history_tokens:
            return

        print(
            f"Trimming history: {current_tokens} tokens > {self.max_history_tokens} limit"
        )

        # Keep system messages, remove oldest user/assistant messages
        items = self.memory.get_items()
        system_messages = [msg for msg in items if msg.get("role") == "system"]
        other_messages = [msg for msg in items if msg.get("role") != "system"]

        # Remove oldest messages until under limit
        while (
            self._count_tokens(system_messages + other_messages)
            > self.max_history_tokens
            and other_messages
        ):
            other_messages.pop(0)

        # Rebuild memory with trimmed history
        self.memory.items = system_messages + other_messages

    def chat(self, message: str) -> str:
        """Send a message and get a response (with automatic trimming)"""
        # Trim history before adding new message
        self._trim_history()

        # Call parent chat method
        return super().chat(message)

    def save_conversation(self, filepath: str):
        """Save conversation to JSON file"""
        with open(filepath, "w") as f:
            json.dump(self.memory.get_items(), f, indent=2)
        print(f"Saved to {filepath}")

    def load_conversation(self, filepath: str):
        """Load conversation from JSON file"""
        with open(filepath, "r") as f:
            self.memory.items = json.load(f)
        print(f"Loaded from {filepath}")

    def get_token_count(self) -> int:
        """Get current token count"""
        return self._count_tokens()


# ============================================================================
# Usage Examples
# ============================================================================


def token_management_example():
    """Example 1: Token counting and automatic trimming"""
    print("=" * 60)
    print("Example 1: Token Management")
    print("=" * 60 + "\n")

    # Create agent with low token limit to trigger trimming
    agent = AgentWithMemory(
        max_history_tokens=500, system_prompt="You are a helpful assistant."
    )

    # Have a long conversation to demonstrate trimming
    for i in range(8):
        question = f"Tell me a fun fact about the number {i}"
        print(f"Turn {i + 1}: {question}")
        agent.chat(question)
        print(f"Tokens: {agent.get_token_count()}/{agent.max_history_tokens}\n")


def persistence_example():
    """Example 2: Save and load conversation state"""
    print("\n" + "=" * 60)
    print("Example 2: Conversation Persistence")
    print("=" * 60 + "\n")

    # Session 1: Have a conversation
    print("Session 1 - Building context:")
    agent1 = AgentWithMemory(system_prompt="You are a helpful assistant.")
    agent1.chat("My name is John")
    agent1.chat("I live in Paris")
    agent1.chat("I'm a software engineer")

    # Save conversation state
    agent1.save_conversation("/tmp/conversation.json")

    # Session 2: Load in new agent (simulating server restart)
    print("\nSession 2 - After restart:")
    agent2 = AgentWithMemory(system_prompt="You are a helpful assistant.")
    agent2.load_conversation("/tmp/conversation.json")

    # Agent remembers context from previous session
    print("User: What's my name and where do I live?")
    answer = agent2.chat("What's my name and where do I live?")
    print(f"Assistant: {answer}")


def redis_persistence_example():
    """Example 3: Redis persistence for production (optional)"""
    try:
        import redis
    except ImportError:
        print("\n=== Redis Persistence ===")
        print("Redis not installed. Run: uv add redis")
        print("Skipping Redis example.\n")
        return

    print("\n=== Redis Persistence (Production) ===\n")

    try:
        # Try to connect to Redis
        r = redis.Redis(host="localhost", port=6379, decode_responses=True)
        r.ping()
    except (redis.ConnectionError, Exception) as e:
        print(f"Redis not available: {e}")
        print("Start Redis with: docker run -d -p 6379:6379 redis:latest")
        print("Skipping Redis example.\n")
        return

    # Create a simple Redis-backed memory store
    session_id = "user_alice"
    key = f"agent:session:{session_id}"

    print(f"Session 1: Creating conversation for {session_id}")

    # Simulate saving messages
    conversation = [
        {"role": "user", "content": "My favorite color is blue"},
        {
            "role": "assistant",
            "content": "I'll remember that your favorite color is blue!",
        },
        {"role": "user", "content": "I live in Tokyo"},
        {"role": "assistant", "content": "Got it! You live in Tokyo."},
    ]

    r.setex(key, 86400, json.dumps(conversation))  # TTL: 24 hours
    print(f"Saved {len(conversation)} messages to Redis with 24-hour TTL\n")

    print("Session 2: Loading conversation (simulating server restart)")

    # Load from Redis
    data = r.get(key)
    if data:
        loaded_conversation = json.loads(data)
        print(f"Loaded {len(loaded_conversation)} messages from Redis")
        print("\nConversation history:")
        for msg in loaded_conversation:
            print(f"  {msg['role']}: {msg['content']}")

    print("\n✓ Redis persistence working! Session survives restarts.\n")

    # Cleanup
    r.delete(key)


def mem0_semantic_memory_example():
    """Example 4: Mem0 for semantic, long-term memory (optional)"""
    try:
        from mem0 import Memory
    except ImportError:
        print("\n=== Mem0 Semantic Memory ===")
        print("Mem0 not installed. Run: uv add mem0ai")
        print("Skipping Mem0 example.\n")
        return

    print("\n=== Mem0 Semantic Memory ===\n")
    print(
        "Mem0 provides semantic search over memories, not just chronological storage.\n"
    )

    try:
        # Initialize Mem0 (uses local vector store by default)
        memory = Memory()

        # User context
        user_id = "john_doe"

        print("Step 1: Adding memories for user")
        print("-" * 50)

        # Add various memories with context
        memories_to_add = [
            "User prefers Python over JavaScript for backend development",
            "User is working on an e-commerce project with a $50,000 budget",
            "User's team uses React for frontend",
            "User mentioned they have a tight deadline - project due in 3 weeks",
            "User is interested in implementing AI features",
        ]

        for mem in memories_to_add:
            memory.add(mem, user_id=user_id)
            print(f"  ✓ Added: {mem}")

        print("\nStep 2: Semantic search for relevant memories")
        print("-" * 50)

        # Query 1: Technology recommendations
        query1 = "What technology stack should I use for my project?"
        print(f"\nQuery: '{query1}'")
        print("Relevant memories:")

        results = memory.search(query1, user_id=user_id)
        for i, result in enumerate(results[:3], 1):
            print(f"  {i}. {result['memory']}")

        # Query 2: Project constraints
        query2 = "What are my project constraints?"
        print(f"\nQuery: '{query2}'")
        print("Relevant memories:")

        results = memory.search(query2, user_id=user_id)
        for i, result in enumerate(results[:3], 1):
            print(f"  {i}. {result['memory']}")

        print("\nStep 3: Multi-user isolation")
        print("-" * 50)

        # Add memory for different user
        other_user_id = "jane_smith"
        memory.add("User is learning Rust programming", user_id=other_user_id)

        # Search only returns memories for john_doe, not jane_smith
        results = memory.search("What is the user learning?", user_id=user_id)
        print(f"Searching for john_doe's learning activities: {len(results)} results")
        print("(Correctly isolated from jane_smith's memories)")

        print("\n✓ Mem0 semantic memory working!")
        print("Key benefit: Retrieves relevant memories, not just recent ones\n")

    except Exception as e:
        print(f"Error with Mem0: {e}")
        print("Note: Mem0 requires additional setup for production use")
        print("See: https://docs.mem0.ai/quickstart\n")


if __name__ == "__main__":
    token_management_example()
    persistence_example()
    redis_persistence_example()
    mem0_semantic_memory_example()
