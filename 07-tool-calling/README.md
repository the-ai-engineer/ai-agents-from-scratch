# Tool Calling: From Basics to Production

## What You'll Learn

In this lesson, you'll learn how to give AI the ability to take actions in the real world through tool calling—the technique that transforms LLMs from pure text generators into systems that can query APIs, search databases, perform calculations, and interact with external services.

You'll start with manual JSON schemas to understand how tool calling works under the hood, then learn how a simple dataclass can reduce boilerplate and make your code cleaner.

By the end of this lesson, you'll understand:
- How tool calling bridges the gap between LLM intelligence and real-world actions
- The request-execute-respond loop that powers AI agents
- Manual tool schema definition (understanding the format)
- Using a dataclass to automate schema generation
- Tool registry patterns for managing multiple tools

## The Problem

LLMs are trained on static data. They can't check the current weather, look up stock prices, search your database, or send emails. They only know what was in their training data.

When a user asks "What's the weather in Paris?", the LLM can only say "I don't have access to real-time weather data." It knows weather exists. It understands the question. But it can't actually get the answer.

Additionally, writing tool definitions manually is tedious. Every tool requires writing JSON Schema by hand, which is error-prone and verbose.

## The Solution: Tool Calling

Tool calling bridges the gap between LLM intelligence and real-world actions. You define functions the AI can call—like `get_weather(city)` or `search_database(query)`—and the LLM decides when and how to use them.

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

### Step-by-Step: Tool Calling Flow

#### Step 1: Define Your Tool Schema

First, you need to tell the LLM about your tool in a format it understands (JSON Schema):

```python
tools = [{
    "type": "function",
    "name": "get_weather",
    "description": "Get the current weather for a given city",
    "parameters": {
        "type": "object",
        "properties": {
            "city": {
                "type": "string",
                "description": "The city name, e.g., Paris, London"
            }
        },
        "required": ["city"]
    }
}]
```

This JSON tells the LLM:
- There's a function called `get_weather`
- It takes one parameter: `city` (a string)
- The `city` parameter is required
- Use this to get weather for a city

#### Step 2: Make the API Call with Tools

Pass your tools to the Responses API:

```python
from openai import OpenAI

client = OpenAI()

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
```

If the LLM called a tool, it returns the function name and JSON arguments. You don't execute anything yet—this is just the request.

#### Step 4: Execute the Tool

Parse the arguments and call your actual function:

```python
import json

def get_weather(city: str) -> str:
    """Get the current weather for a given city."""
    weather_data = {
        "paris": "Sunny, 22°C",
        "london": "Cloudy, 15°C",
    }
    return weather_data.get(city.lower(), f"Weather data not available for {city}")

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

## Reducing Boilerplate with a Dataclass

Writing JSON schemas by hand is tedious. A simple dataclass can automate schema generation from function signatures:

```python
import inspect
from dataclasses import dataclass

@dataclass
class Tool:
    """Simple dataclass to convert functions to OpenAI tool schemas."""
    name: str
    description: str
    parameters: dict

    def to_dict(self) -> dict:
        """Convert to OpenAI Responses API format."""
        return {
            "type": "function",
            "name": self.name,
            "description": self.description,
            "parameters": self.parameters,
        }

    @classmethod
    def from_function(cls, func) -> "Tool":
        """Create a Tool from a Python function."""
        sig = inspect.signature(func)

        properties = {}
        required = []

        for name, param in sig.parameters.items():
            properties[name] = {"type": "string"}
            if param.default == inspect.Parameter.empty:
                required.append(name)

        return cls(
            name=func.__name__,
            description=inspect.getdoc(func) or "",
            parameters={
                "type": "object",
                "properties": properties,
                "required": required,
            },
        )
```

Now you can generate schemas automatically:

```python
def list_files(directory: str = ".") -> str:
    """List all files in a directory.

    Args:
        directory: Path to directory (default: current directory)
    """
    import os
    files = os.listdir(directory)
    return f"Files in '{directory}': {', '.join(files)}"

# Generate schema automatically
list_files_tool = Tool.from_function(list_files)
tools = [list_files_tool.to_dict()]
```

The dataclass:
- Extracts the function name automatically
- Uses the docstring as the description
- Detects required vs optional parameters
- Generates the JSON schema for you

## Code Examples

This lesson includes one comprehensive example file with four progressive examples:

### examples.py - Complete Tool Calling Guide

1. **Manual JSON Schema**: Write the schema by hand to understand the format (weather tool)
2. **Dataclass Helper**: Use a simple dataclass to automate schema generation (list files)
3. **Multiple Tools with Registry**: Clean pattern for managing multiple tools (find files, get file size)
4. **Error Handling**: Return errors as strings for graceful failure (read file with error handling)

```bash
uv run python 07-tool-calling/examples.py
```

Each example builds on the previous one, showing you the progression from manual schemas to production-ready patterns.

## Key Implementation Patterns

### Tool Registry Pattern

Manage multiple tools cleanly using a dictionary:

```python
def list_files(directory: str = ".") -> str:
    """List all files in a directory."""
    import os
    files = os.listdir(directory)
    return f"Files: {', '.join(files)}"

def find_files(pattern: str, directory: str = ".") -> str:
    """Find files matching a pattern in a directory."""
    import os
    import fnmatch
    matches = [f for f in os.listdir(directory) if fnmatch.fnmatch(f, pattern)]
    return f"Found {len(matches)} file(s): {', '.join(matches)}"

# Create registry
tool_registry = {
    "list_files": list_files,
    "find_files": find_files,
}

# Generate schemas automatically
tools = [Tool.from_function(func).to_dict() for func in tool_registry.values()]

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
def send_email(to: str, subject: str, body: str = "No body provided") -> str:
    """Send an email to a recipient.

    Args:
        to: Recipient email address
        subject: Email subject line
        body: Email body content (optional)
    """
    return f"Email sent to {to} with subject '{subject}'"
```

The dataclass automatically marks `body` as optional when it sees a default value.

### Error Handling

Always return errors as strings so the LLM can explain them:

```python
def read_file(filename: str) -> str:
    """Read the contents of a text file."""
    try:
        with open(filename, 'r') as f:
            return f"File content: {f.read()}"
    except FileNotFoundError:
        return f"Error: File '{filename}' not found"
    except PermissionError:
        return f"Error: Permission denied reading '{filename}'"
    except Exception as e:
        return f"Error: {str(e)}"
```

When the tool fails, the LLM receives the error and explains it naturally to the user.

## Key Takeaways

1. **Understand the manual format first**: Writing JSON schemas by hand teaches you exactly what the LLM needs.

2. **Dataclass helpers reduce boilerplate**: Once you understand the format, automate it with a simple dataclass.

3. **Clear tool descriptions are critical**: The LLM uses descriptions to decide when to call tools. Write detailed docstrings.

4. **Tool registry pattern scales**: Start with a dictionary mapping names to functions. It works for 3 tools or 30 tools.

5. **Always return tool results**: The tool call is just a request. You must execute it and return the result for the LLM to answer.

6. **Return errors gracefully**: When tools fail, return error messages as strings. The LLM can explain problems naturally.

7. **Optional parameters work automatically**: Parameters with defaults become optional in the schema.

8. **Start with 2-3 tools**: Don't overwhelm the LLM with 20 tools at once. Add more as needed.

## Common Pitfalls

1. **Poor docstrings**: "Get weather" doesn't tell the LLM enough. Write detailed descriptions.

2. **Not using the tool registry pattern**: Managing multiple tools without a registry leads to messy if/elif chains.

3. **Not returning tool results**: Executing the tool but forgetting to send the result back to the LLM. The conversation hangs.

4. **Too many tools at once**: Starting with 10+ tools confuses the LLM. It struggles to choose. Start small.

5. **Assuming tools are called**: Always check if `output.type == "function_call"`. Sometimes the LLM answers directly.

6. **Forgetting the call_id**: Function call outputs must include the `call_id` from the original function call.

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

Build a multi-tool file system assistant:

1. **List files tool**: List all files in a directory
2. **Read file tool**: Read the contents of a text file
3. **File info tool**: Get file size and modification time
4. **Search files tool** (bonus): Search for text within files with optional case-sensitive parameter

Requirements:
- Start with manual JSON schemas for at least one tool
- Use the dataclass helper for the others
- Implement the tool registry pattern
- Include at least one tool with optional parameters
- Write comprehensive docstrings
- Test with 5+ questions that exercise different tools
- Include error handling that returns clear messages

Test cases should include:
- "What files are in the current directory?" (list files)
- "Read the README.md file" (read file)
- "What is the size of examples.py?" (file info)
- "What is Python?" (needs no tools - tests LLM judgment)
- "Read the file xyz123.txt" (causes FileNotFoundError - tests error handling)

## Next Steps

You've mastered tool calling from fundamentals to production patterns. Now it's time to learn how to orchestrate multiple LLM calls in sophisticated workflows.

Move to **08-workflow-patterns** to learn five fundamental patterns for building complex AI systems: chaining, routing, parallelization, orchestrator-workers, and evaluator-optimizer.

## Resources

- [OpenAI Function Calling Guide](https://platform.openai.com/docs/guides/function-calling) - Official documentation
- [Python Inspect Module](https://docs.python.org/3/library/inspect.html) - For function introspection
- [Dataclasses](https://docs.python.org/3/library/dataclasses.html) - Python dataclass documentation
- [JSON Schema Documentation](https://json-schema.org/) - Schema specification details
