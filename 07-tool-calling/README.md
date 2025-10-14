# Tool Calling: From Basics to Production

## What You'll Learn

In this lesson, you'll learn how to give AI the ability to take actions in the real world through tool calling—the technique that transforms LLMs from pure text generators into systems that can query APIs, search databases, perform calculations, and interact with external services.

You'll start with the fundamentals (manual tool definitions) and progress to production patterns using the `@tool` decorator for automatic schema generation, type safety, and clean architecture.

By the end of this lesson, you'll understand:
- How tool calling bridges the gap between LLM intelligence and real-world actions
- The request-execute-respond loop that powers AI agents
- The `@tool` decorator pattern used in production systems
- Automatic schema generation from function signatures
- Tool registry patterns for managing multiple tools

## The Problem

LLMs are trained on static data. They can't check the current weather, look up stock prices, search your database, or send emails. They only know what was in their training data.

When a user asks "What's the weather in Paris?", the LLM can only say "I don't have access to real-time weather data." It knows weather exists. It understands the question. But it can't actually get the answer.

Additionally, manual tool definitions are tedious and error-prone. Every tool requires writing JSON Schema by hand, manually parsing arguments, and hoping the LLM sends valid data. No type safety. No validation. No IDE autocomplete.

## The Solution: Tool Calling with @tool Decorator

Tool calling bridges the gap between LLM intelligence and real-world actions. You define functions the AI can call—like `get_weather(city)` or `search_database(query)`—and the LLM decides when and how to use them.

The modern approach uses the `@tool` decorator for automatic schema generation:

```python
@tool
def get_weather(city: str) -> str:
    """Get the current weather for a given city.

    Args:
        city: City name, e.g., 'Paris', 'London', 'Tokyo'
    """
    return f"Weather in {city}"

# Schema generated automatically from function signature and docstring!
tools = [get_weather.tool.to_openai_format()]
```

The schema generates automatically from your function signature and docstring. Type hints define parameter types. Required vs optional parameters are detected automatically.

```mermaid
sequenceDiagram
    participant User as 👤 User
    participant Code as 💻 Your Code
    participant LLM as 🧠 LLM
    participant Tool as 🔧 Tool Function

    User->>Code: "What's the weather in Paris?"
    Code->>LLM: Query + Tool Definitions
    LLM->>Code: Call get_weather(city="Paris")
    Note over Code: Parse & validate arguments
    Code->>Tool: Execute get_weather("Paris")
    Tool->>Code: "Sunny, 22°C"
    Code->>LLM: Tool result
    LLM->>Code: "In Paris, it's sunny and 22°C"
    Code->>User: Final answer

    style User fill:#e8f5e9
    style Code fill:#e1f5ff
    style LLM fill:#fff9c4
    style Tool fill:#c8e6c9
```

Here's the pattern:

1. **User asks a question**: "What's the weather in Paris?"
2. **LLM decides to use a tool**: It recognizes it needs the `get_weather` function
3. **LLM generates tool arguments**: `{"city": "Paris"}`
4. **You execute the tool**: Call your actual `get_weather("Paris")` function
5. **LLM uses the result**: Takes the weather data and formulates a natural answer

The LLM orchestrates. You execute. The user gets a complete answer with real-time data.

## How It Works

### The @tool Decorator Pattern

The `@tool` decorator uses Python introspection to automatically generate OpenAI tool schemas from your function signatures:

```python
import inspect
from typing import Callable, Any, get_type_hints
from pydantic import BaseModel, create_model

def tool(func: Callable) -> Callable:
    """Decorator to mark a function as a tool."""
    func.tool = Tool.from_function(func)
    return func

class Tool(BaseModel):
    """Tool metadata for function calling."""
    name: str
    description: str
    parameters: dict[str, Any]

    @classmethod
    def from_function(cls, func: Callable) -> "Tool":
        """Create a Tool from a function using introspection."""
        return cls(
            name=func.__name__,
            description=inspect.getdoc(func) or "",
            parameters=create_json_schema(func),
        )

    def to_openai_format(self) -> dict:
        """Convert to OpenAI Responses API format."""
        return {
            "type": "function",
            "function": {
                "name": self.name,
                "description": self.description,
                "parameters": self.parameters,
            }
        }
```

This pattern:
- Uses `inspect.getdoc()` to extract docstrings as descriptions
- Uses `get_type_hints()` to determine parameter types
- Uses `inspect.signature()` to detect required vs optional parameters
- Uses Pydantic's `create_model()` to generate JSON schemas automatically

### Step-by-Step: Tool Calling Flow

#### Step 1: Define Your Tools

Use the `@tool` decorator on any function you want the LLM to call:

```python
@tool
def get_weather(city: str) -> str:
    """Get the current weather for a given city.

    Args:
        city: City name, e.g., 'Paris', 'London', 'Tokyo'
    """
    weather_db = {
        "paris": "Sunny, 22°C",
        "london": "Cloudy, 15°C",
        "tokyo": "Rainy, 18°C",
    }
    return weather_db.get(city.lower(), f"Weather data not available for {city}")
```

The docstring matters. The LLM uses it to decide when to call this tool. Include an Args section describing each parameter.

#### Step 2: Make the API Call with Tools

Pass your tools to the Responses API:

```python
from openai import OpenAI

client = OpenAI()

tools = [get_weather.tool.to_openai_format()]

response = client.responses.create(
    model="gpt-4o-mini",
    input=[{"role": "user", "content": "What's the weather in Paris?"}],
    tools=tools
)
```

The LLM sees your tools and decides whether to call one.

#### Step 3: Check if a Tool Was Called

The response.output might contain text or function calls:

```python
for item in response.output:
    if item.type == "function_call":
        print(f"Tool: {item.name}")
        print(f"Arguments: {item.arguments}")
    elif item.type == "message":
        print(f"Text response: {item.content}")
```

If the LLM called a tool, it returns the function name and JSON arguments. You don't execute anything yet—this is just the request.

#### Step 4: Execute the Tool

Parse the arguments and call your actual function:

```python
import json

args = json.loads(item.arguments)
result = get_weather(**args)

print(f"Result: {result}")  # "Sunny, 22°C"
```

This is where you perform the real work—API calls, database queries, calculations, whatever the tool does.

#### Step 5: Return the Result to the LLM

Send the tool result back so the LLM can formulate a final answer:

```python
input_with_result = [
    {"role": "user", "content": "What's the weather in Paris?"},
    {"type": "function_call_output", "call_id": item.call_id, "output": result}
]

final_response = client.responses.create(
    model="gpt-4o-mini",
    input=input_with_result,
    tools=tools
)

print(final_response.output_text)
# "The current weather in Paris is sunny with a temperature of 22°C."
```

The LLM takes your raw data and turns it into a natural language response.

## Code Examples

This lesson includes two example files:

### basic.py - Learning the Fundamentals

Three examples showing progression from manual schemas to the @tool decorator:

1. **Manual Tool Definition**: Traditional approach with hand-written JSON Schema
2. **Single Tool with @tool**: Automatic schema generation from function signature
3. **Multiple Tools with Registry**: Clean pattern for managing multiple tools

```bash
cd 07-tool-calling
uv run basic.py
```

### advanced.py - Production Patterns

Three examples demonstrating production-ready patterns:

1. **Basic @tool Decorator**: Automatic schema generation with validation
2. **Multiple Tools with Optional Parameters**: Handling default values and tool registry
3. **Type Validation and Error Handling**: Leveraging type hints for validation

```bash
cd 07-tool-calling
uv run advanced.py
```

## Key Implementation Patterns

### Tool Registry Pattern

Manage multiple tools cleanly using a dictionary:

```python
@tool
def get_weather(city: str) -> str:
    """Get current weather for a city."""
    return f"Weather in {city}"

@tool
def calculate(expression: str) -> str:
    """Evaluate a mathematical expression."""
    try:
        result = eval(expression, {"__builtins__": {}}, {})
        return str(result)
    except Exception as e:
        return f"Error: {str(e)}"

# Create registry
tool_registry = {
    "get_weather": get_weather,
    "calculate": calculate,
}

# Generate schemas automatically
tools = [func.tool.to_openai_format() for func in tool_registry.values()]

# Execute from registry
for output in response.output:
    if output.type == "function_call":
        args = json.loads(output.arguments)
        result = tool_registry[output.name](**args)
```

This pattern:
- Keeps tool definitions clean and organized
- Makes adding new tools trivial
- Simplifies execution with dictionary lookup
- Scales to dozens of tools without messy if/elif chains

### Optional Parameters

Handle optional parameters using default values:

```python
@tool
def send_email(to: str, subject: str, body: str = "No body provided") -> str:
    """Send an email to a recipient.

    Args:
        to: Recipient email address
        subject: Email subject line
        body: Email body content (optional)
    """
    return f"Email sent to {to} with subject '{subject}'"
```

The schema automatically marks `body` as optional and includes the default value.

### Error Handling

Always return errors as strings so the LLM can explain them:

```python
@tool
def calculate(expression: str) -> str:
    """Evaluate a mathematical expression."""
    try:
        result = eval(expression, {"__builtins__": {}}, {})
        return str(result)
    except Exception as e:
        return f"Error: {str(e)}"
```

When the tool fails, the LLM receives the error and explains it naturally to the user.

## Key Takeaways

1. **Use @tool decorator for cleaner code**: Eliminates manual JSON schemas. Function signature + docstring = automatic tool schema.

2. **Clear tool descriptions are critical**: The LLM uses descriptions to decide when to call tools. Write detailed docstrings with Args sections.

3. **Type hints are essential**: The decorator uses type hints to generate correct parameter types. Use `str`, `int`, `bool`, etc.

4. **Tool registry pattern scales**: Start with a dictionary mapping names to functions. It works for 3 tools or 30 tools.

5. **Always return tool results**: The tool call is just a request. You must execute it and return the result for the LLM to answer.

6. **Return errors gracefully**: When tools fail, return error messages as strings. The LLM can explain problems naturally.

7. **Optional parameters work automatically**: Parameters with defaults become optional in the schema automatically.

8. **Start with 2-3 tools**: Don't overwhelm the LLM with 20 tools at once. Add more as needed.

## Common Pitfalls

1. **Missing type hints**: The decorator needs type hints to generate schemas. `def foo(x)` won't work. Use `def foo(x: str)`.

2. **Poor docstrings**: "Get weather" doesn't tell the LLM enough. Write detailed descriptions with Args sections.

3. **Not using the tool registry pattern**: Managing multiple tools without a registry leads to messy if/elif chains.

4. **Not returning tool results**: Executing the tool but forgetting to send the result back to the LLM. The conversation hangs.

5. **Forgetting to call .tool.to_openai_format()**: The decorated function has a `.tool` attribute. You must call `.to_openai_format()` to get the schema.

6. **Too many tools at once**: Starting with 10+ tools confuses the LLM. It struggles to choose. Start small.

7. **Assuming tools are called**: Always check if `output.type == "function_call"`. Sometimes the LLM answers directly.

## When to Use Tool Calling

| Use Case | Use Tool Calling? |
|----------|------------------|
| Get current data (weather, stocks, prices) | Yes |
| Search databases or APIs | Yes |
| Perform calculations or conversions | Yes |
| Send emails, create tickets, update records | Yes |
| Extract structured data from text | No (use structured output) |
| Answer from existing knowledge | No (use prompting or RAG) |
| Creative writing or explanations | No (use plain text) |

Use tool calling when the LLM needs to interact with external systems or perform actions. Use structured output for data extraction. Use plain prompting when the LLM has enough knowledge.

## Real-World Impact

Tool calling is how AI moves from answering questions to taking actions. Here's what it enables in production:

**Customer Support Automation**: Look up order status, check account details, create support tickets, send confirmation emails—all automatically.

**Data Retrieval Systems**: Search internal databases, query APIs, pull real-time information, aggregate data from multiple sources.

**Calculation and Analysis**: Perform complex math, convert units, analyze data, generate reports with real numbers.

**Workflow Automation**: Create calendar events, send notifications, update CRM records, trigger webhooks—AI that actually does things.

**Research Assistants**: Search the web, query knowledge bases, fetch documentation, compile information from multiple sources.

Companies use tool calling to automate thousands of repetitive tasks, provide 24/7 data access, and build AI systems that take real-world actions reliably.

## Assignment

Build a multi-tool assistant with automatic schema generation:

1. **Weather tool**: Get current weather for any city
2. **Calculator tool**: Perform mathematical calculations
3. **Email validator tool**: Check if an email address is valid format
4. **Time tool** (bonus): Get current time in different timezones with optional format parameter

Requirements:
- Use the `@tool` decorator for all tools
- Implement the tool registry pattern
- Include at least one tool with optional parameters
- Write comprehensive docstrings with Args sections
- Test with 5+ questions that exercise different tools
- Include error handling that returns clear messages

Test cases should include:
- A question that needs each individual tool
- A question that needs no tools (tests LLM judgment)
- A question that causes an error (tests error handling)

## Next Steps

You've mastered tool calling from fundamentals to production patterns. Now it's time to learn how to orchestrate multiple LLM calls in sophisticated workflows.

Move to **08-workflow-patterns** to learn five fundamental patterns for building complex AI systems: chaining, routing, parallelization, orchestrator-workers, and evaluator-optimizer.

## Resources

- [OpenAI Function Calling Guide](https://platform.openai.com/docs/guides/function-calling) - Official documentation
- [Python Type Hints](https://docs.python.org/3/library/typing.html) - Master typing for better schemas
- [Pydantic Documentation](https://docs.pydantic.dev) - Understanding create_model and schemas
- [JSON Schema Documentation](https://json-schema.org/) - Schema specification details
