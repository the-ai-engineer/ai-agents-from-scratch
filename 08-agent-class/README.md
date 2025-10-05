# Lesson 08: Building a Production-Ready Agent Class

## What You'll Learn

In this lesson, you'll transform the raw agent loop pattern into a clean, reusable class architecture that's ready for production use.

Building one-off agents teaches the pattern. Building multiple agents reveals when to abstract. This lesson shows you how to create an Agent class that encapsulates complexity, provides a simple public interface, and scales to multiple agent implementations.

By the end of this lesson, you'll master agent class design principles, build flexible tool registration systems, implement internal conversation management, design clean public interfaces, and add proper error handling at the class level.

This is how you go from prototype to production. Every serious agent system uses this architecture.

## The Problem

You've built the agent loop. It works. Now you need to build three more agents: a research assistant, a code reviewer, and a customer service bot.

Do you copy-paste the agent loop code three times? No. That's unmaintainable.

Do you abstract immediately after building one agent? Also no. That's premature abstraction. You don't know what varies yet.

The right approach: build 2-3 agents using the raw pattern, notice what repeats, then abstract the common parts into a reusable class.

## When to Abstract: The Rule of Three

**Bad:** Abstract after building one agent. You're guessing at the interface.

**Good:** Abstract after building 2-3 agents. You understand what varies and what stays constant.

What stays constant across agents:
- The core agent loop logic
- Tool execution machinery
- Conversation history management
- Error handling patterns
- API interaction code

What varies between agents:
- Available tools
- System prompts
- Model selection
- Iteration limits
- Custom behavior

Your Agent class should make the constants invisible and the variables explicit.

## API Note: Tool Calling with Chat Completions

**Important:** This lesson uses the Chat Completions API (`client.chat.completions.create()`) because tool calling requires the `tools` parameter and message-based conversation format. The Responses API does not yet support tool calling in the same way.

When OpenAI adds tool calling support to the Responses API, the Agent class can be updated to use the simpler `client.responses.create()` interface. For now, agents with tool calling must use Chat Completions.

## How an Agent Class Works

The core insight: users shouldn't see the loop. They should see a simple interface:

```python
# Simple public interface
agent = Agent()
agent.register_tool("get_weather", get_weather_func, schema, "Fetches weather")
response = agent.chat("What's the weather in Tokyo?")
```

Behind the scenes, the class handles:
- Running the agent loop
- Executing tool calls
- Managing conversation history
- Preventing infinite loops
- Error handling

Complexity hidden. Interface clean.

## Code Example: Basic Agent Class

Here's a production-ready implementation:

```python
from openai import OpenAI
from pydantic import BaseModel
from typing import Callable, Any
import json


class Agent:
    """
    Autonomous agent with tool-calling capabilities.

    This class encapsulates the agent loop pattern, making it easy to create
    agents with different tools and behaviors.
    """

    def __init__(
        self,
        model: str = "gpt-4o-mini",
        system_prompt: str | None = None,
        max_iterations: int = 5,
        temperature: float = 0
    ):
        """
        Initialize the agent.

        Args:
            model: OpenAI model to use
            system_prompt: Optional system message to set agent behavior
            max_iterations: Maximum agent loop iterations
            temperature: LLM temperature (0 = deterministic)
        """
        self.client = OpenAI()
        self.model = model
        self.max_iterations = max_iterations
        self.temperature = temperature

        # Tool registry: maps tool names to functions
        self.tools: dict[str, Callable] = {}

        # Tool schemas for OpenAI API
        self.tool_schemas: list[dict] = []

        # Conversation history
        self.conversation_history: list[dict] = []

        # Set system prompt if provided
        if system_prompt:
            self.conversation_history.append({
                "role": "system",
                "content": system_prompt
            })

    def register_tool(
        self,
        name: str,
        func: Callable,
        args_schema: type[BaseModel],
        description: str
    ) -> None:
        """
        Register a tool that the agent can call.

        Args:
            name: Tool name (must match function name)
            func: The actual Python function to execute
            args_schema: Pydantic model defining the function's parameters
            description: Human-readable description of what the tool does

        Example:
            class GetWeatherArgs(BaseModel):
                location: str

            agent.register_tool(
                name="get_weather",
                func=get_weather_function,
                args_schema=GetWeatherArgs,
                description="Get current weather for a location"
            )
        """
        # Store function in registry
        self.tools[name] = func

        # Convert Pydantic schema to OpenAI format
        self.tool_schemas.append({
            "type": "function",
            "function": {
                "name": name,
                "description": description,
                "parameters": args_schema.model_json_schema()
            }
        })

    def chat(self, message: str, verbose: bool = False) -> str:
        """
        Send a message to the agent and get a response.

        This method handles the entire agent loop internally.

        Args:
            message: User's message or question
            verbose: If True, print detailed execution logs

        Returns:
            Agent's final response
        """
        # Add user message to history
        self.conversation_history.append({
            "role": "user",
            "content": message
        })

        # Run agent loop
        for iteration in range(self.max_iterations):
            if verbose:
                print(f"\n{'='*60}")
                print(f"Iteration {iteration + 1}/{self.max_iterations}")
                print('='*60)

            # Call LLM with conversation history + available tools
            # Note: Using chat.completions for tool calling support
            response = self.client.chat.completions.create(
                model=self.model,
                messages=self.conversation_history,
                tools=self.tool_schemas if self.tool_schemas else None,
                temperature=self.temperature
            )

            message = response.choices[0].message

            # Add assistant's response to history
            self.conversation_history.append(message)

            # Check for tool calls
            if not message.tool_calls:
                # No tools = agent has final answer
                if verbose:
                    print("\nAgent finished. Returning final answer.")
                return message.content or "No response generated"

            # Execute tool calls
            if verbose:
                print(f"\nAgent calling {len(message.tool_calls)} tool(s):")

            for tool_call in message.tool_calls:
                tool_name = tool_call.function.name
                tool_args = json.loads(tool_call.function.arguments)

                if verbose:
                    print(f"  - {tool_name}({tool_args})")

                # Execute the tool
                try:
                    result = self._execute_tool(tool_name, tool_args)
                    if verbose:
                        print(f"    Result: {result}")
                except Exception as e:
                    result = f"Error executing {tool_name}: {str(e)}"
                    if verbose:
                        print(f"    Error: {result}")

                # Add tool result to history
                self.conversation_history.append({
                    "role": "tool",
                    "tool_call_id": tool_call.id,
                    "content": str(result)
                })

        # Max iterations reached
        return f"Agent reached maximum iterations ({self.max_iterations}) without completing."

    def _execute_tool(self, name: str, args: dict) -> Any:
        """
        Internal method to execute a registered tool.

        Args:
            name: Tool name
            args: Tool arguments as a dictionary

        Returns:
            Tool execution result

        Raises:
            ValueError: If tool not registered
        """
        if name not in self.tools:
            raise ValueError(f"Tool '{name}' not registered")

        func = self.tools[name]
        return func(**args)

    def reset_conversation(self) -> None:
        """Clear conversation history (keeps system prompt if set)."""
        system_messages = [
            msg for msg in self.conversation_history
            if msg["role"] == "system"
        ]
        self.conversation_history = system_messages

    def get_conversation_history(self) -> list[dict]:
        """Get the full conversation history."""
        return self.conversation_history.copy()
```

## Using the Agent Class

Here's how simple it becomes to create agents:

```python
from pydantic import BaseModel

# Define tool schemas
class GetWeatherArgs(BaseModel):
    location: str

class GetTimeArgs(BaseModel):
    timezone: str

# Define tool functions
def get_weather(location: str) -> str:
    # Real implementation would call a weather API
    return f"Weather in {location}: Sunny, 72°F"

def get_current_time(timezone: str) -> str:
    # Real implementation would get actual time
    return f"Current time in {timezone}: 2:30 PM"

# Create agent
agent = Agent(
    model="gpt-4o-mini",
    system_prompt="You are a helpful assistant with access to weather and time information.",
    max_iterations=5,
    temperature=0
)

# Register tools
agent.register_tool(
    name="get_weather",
    func=get_weather,
    args_schema=GetWeatherArgs,
    description="Get current weather for a location"
)

agent.register_tool(
    name="get_current_time",
    func=get_current_time,
    args_schema=GetTimeArgs,
    description="Get current time for a timezone"
)

# Use agent
response = agent.chat("What's the weather in Paris and what time is it there?")
print(response)
```

That's it. Clean, simple, reusable.

## Design Principles

### 1. Simple Public Interface

Users see three methods:
- `register_tool()`: Add capabilities
- `chat()`: Get answers
- `reset_conversation()`: Start fresh

Everything else is internal (prefixed with `_`).

### 2. Flexible Configuration

All behavior is configurable at initialization:
- Model selection
- System prompt
- Max iterations
- Temperature

Different agents have different needs. Make it easy to customize.

### 3. Internal Complexity

The agent loop, tool execution, and history management are hidden. Users don't need to understand the loop to use the agent.

### 4. Tool Registry Pattern

Tools are registered dynamically. This allows:
- Different agents with different tool sets
- Adding tools at runtime
- Easy testing with mock tools

### 5. Conversation Management

History is maintained internally. Users can:
- Get history for debugging (`get_conversation_history()`)
- Reset when needed (`reset_conversation()`)
- But don't have to manage it manually

## Running the Example

This lesson includes a complete agent implementation:

```bash
cd 08-agent-class
uv run example.py
```

The example shows:
- Creating multiple agents with different tools
- Tool registration
- Multi-turn conversations
- Resetting conversation state

Try creating your own agent with custom tools.

## Key Takeaways

1. **Don't abstract too early.** Build 2-3 agents first, then extract the pattern. Otherwise you're guessing at the interface.

2. **Simple public interface.** Users shouldn't see the loop. `agent.chat(message)` is all they need.

3. **Hide complexity.** Tool execution, loop logic, and history management are internal details.

4. **Flexible tool registration.** Make it easy to add capabilities without changing the agent code.

5. **Maintain conversation history internally.** Users shouldn't have to think about message management.

## Common Pitfalls

1. **Premature abstraction**: Building an Agent class after one agent. You don't know what varies yet. Build 2-3 first.

2. **Exposing internals**: Making users call `_execute_tool()` or manage `conversation_history` directly. Keep the interface minimal.

3. **Hardcoding tools**: Putting tool definitions inside the Agent class. Use registration instead for flexibility.

4. **No error handling**: Tools can fail. Catch exceptions in `_execute_tool()` and return error messages instead of crashing.

5. **Not validating tool schemas**: Register tools with invalid schemas and get cryptic API errors. Validate early.

6. **Forgetting to reset history**: Long conversations accumulate tokens. Provide `reset_conversation()` for multi-session use.

## Real-World Impact

The Agent class architecture is how production systems are built. It provides:

**Reusability**: Write the agent loop once, use it for all agents. Companies report 70% reduction in code duplication.

**Testability**: Mock tools for testing. Test agent behavior without real API calls or external services.

**Maintainability**: Fix bugs in the agent loop once, all agents benefit. Update tool execution strategy once, everywhere improves.

**Developer experience**: Engineers can create new agents in minutes, not hours. Just register tools and go.

**Flexibility**: Same agent class powers research bots, customer service, code review, data analysis, and more. One architecture, many applications.

Teams using this pattern ship agents 5x faster than those rebuilding the loop every time.

## Assignment

Build three different agents using your Agent class:

1. **Weather Agent**: Tools for weather and time
2. **Math Agent**: Tools for calculations (add, multiply, divide)
3. **Research Agent**: Tools for web search and summarization

Each agent should:
- Use the same Agent class
- Register different tools
- Have a custom system prompt
- Handle multi-turn conversations

Notice how easy it is to create new agents once the abstraction is right. That's the power of good design.

## Next Steps

Now that you have a production-ready agent architecture, move to [Lesson 09 - Memory](../09-memory) to learn how to manage conversation history, handle token limits, and persist state for long-running agents.

## Resources

- [Object-Oriented Design Principles](https://en.wikipedia.org/wiki/SOLID)
- [The Rule of Three (Software Development)](https://en.wikipedia.org/wiki/Rule_of_three_(computer_programming))
- [Python Class Design](https://realpython.com/python-classes/)
- [LangChain Agent Implementation](https://github.com/langchain-ai/langchain/tree/master/libs/langchain/langchain/agents) - See how a production framework structures agents
