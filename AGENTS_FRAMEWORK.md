# Agents Framework

After completing this course, you'll have built a lightweight agent framework from scratch. This `agents.py` module contains the production-ready patterns you learned.

## What's Inside

### 🧠 ConversationMemory
Manage multi-turn conversations (from Lesson 02):

```python
from agents import ConversationMemory

memory = ConversationMemory(instructions="You are a helpful assistant")
memory.add_message("user", "Hello")
memory.add_message("assistant", "Hi! How can I help you?")

# Get history for API calls
history = memory.get_history()
```

### 🔧 Tool
Automatic schema generation from Python functions (from Lesson 05-06):

```python
from agents import Tool

def get_weather(city: str, units: str = "celsius") -> str:
    """Get current weather for a city

    Args:
        city: City name (e.g., 'Paris', 'London')
        units: Temperature units ('celsius' or 'fahrenheit')
    """
    return f"Weather in {city}: 22°{units[0].upper()}"

# Create tool - schema auto-generated!
tool = Tool(get_weather)
schema = tool.get_schema()
result = tool.execute(city="Paris")
```

### 🤖 Agent
Complete agent with tool calling and loop (from Lesson 08-09):

```python
from agents import Agent, Tool

# Create agent
agent = Agent(
    model="gpt-4o-mini",
    instructions="You are a helpful assistant.",
    max_iterations=10
)

# Register tools
agent.register_tool(Tool(get_weather))
agent.register_tool(Tool(calculate))

# Chat - agent automatically decides which tools to use
response = agent.chat("What's the weather in Paris and London?")
print(response)
```

## Quick Start

```python
from agents import Agent, Tool

# Define your tools
def search_web(query: str) -> dict:
    """Search the web for information

    Args:
        query: Search query string
    """
    # Your search implementation here
    return {"results": f"Results for: {query}"}

def get_stock_price(symbol: str) -> dict:
    """Get current stock price

    Args:
        symbol: Stock symbol (e.g., 'AAPL', 'GOOGL')
    """
    # Your stock price implementation here
    return {"symbol": symbol, "price": 150.23}

# Create agent and register tools
agent = Agent(instructions="You are a helpful financial assistant")
agent.register_tool(Tool(search_web))
agent.register_tool(Tool(get_stock_price))

# Use it
response = agent.chat("What's the current price of Apple stock?")
print(response)
```

## How It Works

### The Agent Loop

When you call `agent.chat(message)`, here's what happens:

1. **Add message** to conversation history
2. **Call LLM** with tools available
3. **If LLM calls tools**:
   - Execute all tool calls
   - Add results to history
   - Go back to step 2
4. **If LLM returns text**:
   - Return that as final answer

This loop continues until the agent has a complete answer or reaches `max_iterations`.

### Tool Schema Generation

The `Tool` class automatically generates OpenAI schemas from your Python functions by:

1. Reading function signature for parameters
2. Extracting type hints (str, int, float, etc.)
3. Parsing docstring for descriptions
4. Building JSON schema automatically

No manual schema writing required!

## Customization

### Custom Tool Names

```python
tool = Tool(
    function=my_function,
    name="custom_name",
    description="Custom description"
)
```

### Agent Configuration

```python
agent = Agent(
    model="gpt-4",                    # Use GPT-4
    instructions="...",               # System prompt
    max_iterations=20,                # Allow more iterations
    api_key="your-key"               # Or use environment variable
)
```

### Conversation Management

```python
# Reset conversation
agent.reset()

# Access history
print(agent.conversation_history)
```

## Design Philosophy

This framework follows the patterns you learned in the course:

1. **Simple > Complex**: Clean APIs, minimal abstractions
2. **Transparent > Magic**: You understand how every piece works
3. **Production-Ready**: Error handling, type safety, proper patterns
4. **Extensible**: Easy to customize and build on

## Comparison to LangChain/LlamaIndex

| Feature | This Framework | LangChain |
|---------|---------------|-----------|
| **Lines of code** | ~500 | ~100,000+ |
| **Learning curve** | You built it! | Steep |
| **Debugging** | Easy | Complex |
| **Dependencies** | openai only | Many |
| **Customization** | Full control | Limited |
| **Understanding** | Complete | Black box |

After completing this course, you can:
- Use this framework for simple projects
- Understand LangChain/LlamaIndex internals
- Build your own custom framework
- Choose the right tool for each job

## When to Use This

✅ **Use this framework when:**
- Building simple to medium complexity agents
- You need full control and understanding
- You want minimal dependencies
- You're learning or teaching AI agents

❌ **Consider LangChain/LlamaIndex when:**
- You need pre-built integrations (100+ tools)
- Building complex multi-agent systems
- You want rapid prototyping
- Team already uses these frameworks

## Examples

See the course lessons for complete examples:
- **Lesson 08**: Agent loop fundamentals
- **Lesson 09**: Reusable agent class
- **Lesson 11**: FAQ agent with RAG
- **Lesson 12**: Research assistant

## Testing

```python
# Run the framework tests
uv run python agents.py
```

You should see the agent successfully:
1. Call weather tool
2. Call calculator tool
3. Handle multi-tool queries

## Next Steps

Now that you have this framework:

1. **Build something**: Create your own agent with custom tools
2. **Extend it**: Add features you need (streaming, retries, etc.)
3. **Share it**: Help others learn by showing your work
4. **Contribute**: Improve this framework and share back

Remember: You built this by completing the course. You understand every line. That's powerful.

## License

MIT - Use freely in your projects, commercial or personal.
