# Core Primitives - src/

This directory contains the foundational building blocks for creating AI agents from scratch. These primitives are designed to be **simple enough for learning** while being **robust enough for real use**.

## Philosophy

The code here demonstrates the core patterns that power production agent systems:

- **No frameworks** - Pure Python with minimal dependencies
- **Transparent** - Every line is readable and understandable
- **Educational** - Extensive documentation explains the "why", not just the "how"
- **Production-ready patterns** - Real error handling, logging, and best practices

## Components

### 1. Tool (`tool.py`)

Converts Python functions into LLM-compatible tool schemas.

```python
from src import Tool

def get_weather(city: str) -> str:
    """Get the current weather for a city."""
    return f"Weather in {city}: Sunny, 72F"

# Convert to tool
tool = Tool.from_function(get_weather)

# Convert to OpenAI format
schema = tool.to_dict()
```

**Key Features:**
- Automatic schema generation from function signatures
- Extracts docstrings for descriptions
- Identifies required vs optional parameters
- OpenAI-compatible format

**When to use:**
- You want to give agents capabilities
- You're building custom tools
- You need to convert functions to schemas

### 2. ConversationMemory (`agent.py`)

Manages conversation history for agents.

```python
from src import ConversationMemory

memory = ConversationMemory(system_prompt="You are helpful.")

memory.add_message("user", "Hello!")
memory.add_message("assistant", "Hi there!")

history = memory.get_history()  # Get full conversation
memory.clear()  # Reset conversation (keeps system prompt)
```

**Key Features:**
- Maintains user/assistant/tool message history
- System prompts persist across clears
- Simple append-only interface
- Foundation for advanced memory strategies

**When to use:**
- Building conversational agents
- Need to track dialogue context
- Implementing custom memory patterns

### 3. Agent (`agent.py`)

Autonomous agent with tool-calling capabilities.

```python
from src import Agent

# Create agent
agent = Agent(
    system_prompt="You are a helpful assistant.",
    max_iterations=5
)

# Register tools
agent.register_tool(get_weather)
agent.register_tools(calculate, search_web)

# Chat
response = agent.chat("What's the weather in Tokyo?")
```

**Key Features:**
- Implements the agent loop pattern
- Tool execution with error handling
- Conversation memory management
- Max iterations to prevent infinite loops
- Comprehensive logging

**When to use:**
- Building autonomous agents
- Need multi-step reasoning
- Want tool-calling capabilities
- Production agent systems

## The Agent Loop Pattern

The core pattern implemented here is: **"A model using tools in a loop"**

```
User Query
    ↓
[Agent Loop: max N iterations]
    ↓
Send history + tools to LLM
    ↓
LLM Decision: Call tools OR Final answer?
    ↓
If tools → Execute → Add results → Loop
If answer → Return to user
```

This simple pattern powers every autonomous agent system.

## Architecture Diagram

```
┌─────────────────────────────────────────────────┐
│                    Agent                        │
│  ┌───────────────────────────────────────────┐ │
│  │        ConversationMemory                 │ │
│  │  - System prompt                          │ │
│  │  - User/Assistant messages                │ │
│  │  - Tool results                           │ │
│  └───────────────────────────────────────────┘ │
│                                                 │
│  ┌───────────────────────────────────────────┐ │
│  │           Tool Registry                   │ │
│  │  - get_weather → function                 │ │
│  │  - calculate → function                   │ │
│  │  - search_web → function                  │ │
│  └───────────────────────────────────────────┘ │
│                                                 │
│  ┌───────────────────────────────────────────┐ │
│  │          Agent Loop                       │ │
│  │  1. Call LLM with history + tools         │ │
│  │  2. LLM decides: tools or answer?         │ │
│  │  3. Execute tools if needed               │ │
│  │  4. Add results to memory                 │ │
│  │  5. Repeat until final answer             │ │
│  └───────────────────────────────────────────┘ │
└─────────────────────────────────────────────────┘
```

## Logging

All components include comprehensive logging:

```python
import logging

# Enable INFO logs to see agent decisions
logging.basicConfig(level=logging.INFO)

# Enable DEBUG logs to see everything
logging.getLogger('src').setLevel(logging.DEBUG)

agent = Agent()
response = agent.chat("Hello")
# You'll see:
# - Tool registrations
# - Agent loop iterations
# - Tool executions
# - Errors and warnings
```

**Log Levels:**
- **DEBUG**: Everything (tool schemas, full messages, detailed execution)
- **INFO**: Important events (tool registration, loop iterations, completions)
- **WARNING**: Potential issues
- **ERROR**: Failures (tool errors, API errors, max iterations)

## Error Handling

All components handle errors gracefully:

### Tool Execution Errors

```python
agent.register_tool(broken_function)
response = agent.chat("Use the broken tool")
# Agent receives error message in JSON format
# LLM can see the error and handle it or report to user
```

### Max Iterations

```python
agent = Agent(max_iterations=3)
# If agent can't solve problem in 3 iterations:
# Raises RuntimeError with clear message
```

### Missing Tools

```python
# If LLM requests non-existent tool:
# Error returned to LLM, logged, and handled gracefully
```

## Testing

Comprehensive test suite with 36 tests covering:

- Tool creation from various function types
- Agent initialization and configuration
- Tool registration
- Conversation memory
- Tool execution (success and failure)
- Agent loop with mocked API calls

Run tests:
```bash
# All tests
uv run pytest tests/ -v

# With coverage
uv run pytest tests/ --cov=src --cov-report=term-missing
```

See `tests/README.md` for details.

## Examples

### Example 1: Simple Agent

```python
from src import Agent

agent = Agent(system_prompt="You are concise.")
response = agent.chat("What is 2+2?")
```

### Example 2: Agent with Tools

```python
from src import Agent

def search_web(query: str) -> str:
    """Search the web for information."""
    return f"Results for: {query}"

agent = Agent()
agent.register_tool(search_web)
response = agent.chat("Search for Python tutorials")
```

### Example 3: Multi-Step Reasoning

```python
from src import Agent

def get_weather(city: str) -> str:
    """Get weather for a city."""
    return f"Weather in {city}: Sunny"

def calculate(operation: str, a: str, b: str) -> str:
    """Perform math operation."""
    # Implementation

agent = Agent()
agent.register_tools(get_weather, calculate)

# Agent will call both tools to answer
response = agent.chat(
    "What's the weather in Tokyo and Paris? "
    "Calculate the temperature difference."
)
```

### Example 4: Conversation Memory

```python
from src import Agent

agent = Agent()

# Conversation 1
agent.chat("My name is Alice")
agent.chat("What's my name?")  # Agent remembers: "Alice"

# Reset
agent.reset()
agent.chat("What's my name?")  # Agent doesn't remember
```

## Running the Demo

A complete example demonstrating all features:

```bash
uv run src/example.py
```

This shows:
- Tool creation
- Agent with no tools
- Agent with single tool
- Agent with multiple tools
- Multi-step reasoning
- Conversation memory

## Design Decisions

### Why separate Tool class?

**Reason:** Encapsulates schema generation logic. Functions are converted once at registration, not on every API call.

### Why ConversationMemory class?

**Reason:** Separates state management from agent logic. Makes it easy to implement advanced memory strategies (trimming, persistence, etc.) without changing agent code.

### Why max_iterations parameter?

**Reason:** Prevents infinite loops. In production, LLMs can occasionally get stuck calling tools repeatedly. This provides a safety mechanism.

### Why JSON for tool results?

**Reason:** Structured format makes it easy to include both results and errors. LLM can parse and handle both cases.

### Why logging instead of print?

**Reason:** Production code needs proper logging. Users can configure log levels, filter messages, and integrate with logging infrastructure.

## Extending the Primitives

### Custom Memory Strategy

```python
class SlidingWindowMemory(ConversationMemory):
    def __init__(self, window_size: int = 10):
        super().__init__()
        self.window_size = window_size

    def add_message(self, role: str, content: str):
        super().add_message(role, content)
        # Keep only last N messages
        system_msgs = [m for m in self.messages if m["role"] == "system"]
        recent_msgs = [m for m in self.messages if m["role"] != "system"]
        self.messages = system_msgs + recent_msgs[-self.window_size:]
```

### Custom Tool Validation

```python
class ValidatedAgent(Agent):
    def register_tool(self, func: Callable):
        # Add custom validation
        if not func.__doc__:
            raise ValueError("Tools must have docstrings")
        super().register_tool(func)
```

### Streaming Responses

```python
class StreamingAgent(Agent):
    def chat_stream(self, message: str):
        # Implement streaming using OpenAI's stream parameter
        # Yield chunks as they arrive
        pass
```

## Best Practices

### 1. Always Provide Good Docstrings

```python
# Bad
def search(q: str) -> str:
    return "results"

# Good
def search(query: str) -> str:
    """Search the web for information about a topic."""
    return "results"
```

### 2. Handle Tool Errors

```python
def risky_tool(param: str) -> str:
    """Tool that might fail."""
    try:
        # Risky operation
        return result
    except Exception as e:
        return f"Error: {str(e)}"
```

### 3. Set Appropriate max_iterations

```python
# Simple tasks
agent = Agent(max_iterations=3)

# Complex multi-step tasks
agent = Agent(max_iterations=10)
```

### 4. Use Specific System Prompts

```python
# Bad
agent = Agent(system_prompt="Be helpful")

# Good
agent = Agent(
    system_prompt="You are a customer service agent. "
    "Be polite, concise, and use tools to look up information. "
    "Always verify information before responding."
)
```

### 5. Enable Logging in Development

```python
import logging

logging.basicConfig(
    level=logging.DEBUG,
    format='%(name)s - %(levelname)s - %(message)s'
)
```

## Performance Considerations

### Token Usage

Every message in history counts toward token limits and costs:
- Keep conversations focused
- Implement memory trimming for long conversations
- Clear history when starting new topics

### API Calls

Each agent loop iteration = 1 API call:
- Set appropriate `max_iterations`
- Design tools to minimize back-and-forth
- Consider batching operations

### Tool Execution

Tools execute synchronously:
- Keep tool functions fast
- Use async tools for I/O-bound operations (future enhancement)
- Handle timeouts in tool code

## Common Pitfalls

### 1. Forgetting max_iterations

**Problem:** Agent gets stuck in infinite loop
**Solution:** Always set `max_iterations` with appropriate value

### 2. Poor tool docstrings

**Problem:** LLM doesn't understand when to use tools
**Solution:** Write clear, specific docstrings

### 3. Not handling tool errors

**Problem:** Agent crashes on tool failures
**Solution:** Wrap tool code in try-except

### 4. Losing conversation context

**Problem:** Agent forgets previous exchanges
**Solution:** Don't clear memory unless intentional

### 5. Exposing API keys

**Problem:** Hardcoded keys in code
**Solution:** Use environment variables with python-dotenv

## FAQ

**Q: Can I use models other than OpenAI?**
A: Yes, but you'll need to modify the agent to use a different client. The patterns are model-agnostic.

**Q: How do I add async tool execution?**
A: Modify `_execute_tool` to be async and use asyncio for concurrent execution. See advanced examples.

**Q: Can I persist conversations to a database?**
A: Yes, serialize `memory.get_history()` to JSON and store it. Restore by initializing memory with saved messages.

**Q: How do I handle rate limits?**
A: Add retry logic with exponential backoff in the agent loop or use tenacity library.

**Q: Can I stream responses to users?**
A: Yes, use OpenAI's streaming API in the agent loop. You'll need to modify the response handling.

## Next Steps

After understanding these primitives:

1. **Read the tests** (`tests/`) - See how components are used and tested
2. **Run the example** (`src/example.py`) - See all features in action
3. **Build your own agent** - Create custom tools and use cases
4. **Explore lessons** - See these primitives used in real applications
5. **Extend the code** - Add features like streaming, async, persistence

## Contributing

When modifying these primitives:

1. Keep code simple and educational
2. Add comprehensive docstrings
3. Write tests for new features
4. Update this README
5. Ensure logging covers new code paths

## Resources

- **OpenAI Function Calling**: https://platform.openai.com/docs/guides/function-calling
- **Anthropic: Building Effective Agents**: https://www.anthropic.com/engineering/building-effective-agents
- **Python Logging**: https://docs.python.org/3/howto/logging.html
- **pytest Documentation**: https://docs.pytest.org/

---

**Remember:** These primitives teach you how agents work at the fundamental level. Understanding this foundation allows you to build with any framework—or create your own.
