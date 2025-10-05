# Give Your AI Superpowers: Tool Calling Fundamentals

## What You'll Learn

In this lesson, you'll learn how to give AI the ability to take actions in the real world through tool calling—the technique that transforms LLMs from pure text generators into systems that can query APIs, search databases, perform calculations, and interact with external services.

This is where AI moves from answering questions to actually doing things. You'll master the fundamentals of defining tools, letting the LLM decide when to use them, and building the request-execute-respond loop that powers AI agents.

By the end of this lesson, you'll understand the pattern that makes AI useful for automation, data retrieval, and real-world tasks.

## The Problem

LLMs are trained on static data. They can't check the current weather, look up stock prices, search your database, or send emails. They only know what was in their training data.

When a user asks "What's the weather in Paris?", the LLM can only say "I don't have access to real-time weather data." It knows weather exists. It understands the question. But it can't actually get the answer.

This limitation makes LLMs useless for most real-world tasks. You need AI that can interact with the world, not just talk about it.

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
    Note over Code: Parse arguments
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

Tool calling follows a simple five-step pattern that you'll use in every implementation.

### Step 1: Define Your Tools

Tools are defined using JSON Schema. You specify the function name, description, and parameters:

```python
tools = [{
    "type": "function",
    "function": {
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
    }
}]
```

The description matters. The LLM uses it to decide when to call this tool. "Get weather" is vague. "Get the current weather for a given city" is clear.

### Step 2: Make the API Call with Tools

Pass your tools array to the API:

```python
from openai import OpenAI

client = OpenAI()

response = client.chat.completions.create(
    model="gpt-4o-mini",
    messages=[{"role": "user", "content": "What's the weather in Paris?"}],
    tools=tools
)
```

The LLM sees your tools and decides whether to call one.

### Step 3: Check if a Tool Was Called

The response might contain text or a tool call request:

```python
message = response.choices[0].message

if message.tool_calls:
    tool_call = message.tool_calls[0]
    print(f"Tool: {tool_call.function.name}")
    print(f"Arguments: {tool_call.function.arguments}")
else:
    print(f"Text response: {message.content}")
```

If the LLM called a tool, it returns the function name and JSON arguments. You don't execute anything yet—this is just the request.

### Step 4: Execute the Tool

Parse the arguments and call your actual function:

```python
import json

args = json.loads(tool_call.function.arguments)
result = get_weather(**args)  # Your actual implementation

print(f"Result: {result}")  # "Sunny, 22°C"
```

This is where you perform the real work—API calls, database queries, calculations, whatever the tool does.

### Step 5: Return the Result to the LLM

Send the tool result back so the LLM can formulate a final answer:

```python
messages = [
    {"role": "user", "content": "What's the weather in Paris?"},
    message,  # The assistant's tool call
    {
        "role": "tool",
        "tool_call_id": tool_call.id,
        "content": result
    }
]

final_response = client.chat.completions.create(
    model="gpt-4o-mini",
    messages=messages,
    tools=tools
)

print(final_response.choices[0].message.content)
# "The current weather in Paris is sunny with a temperature of 22°C."
```

The LLM takes your raw data and turns it into a natural language response.

## Code Examples

### Example 1: Basic Weather Tool

Complete working example of a single tool:

```python
import os
import json
from openai import OpenAI

client = OpenAI()

def get_weather(city: str) -> str:
    """Your actual weather implementation"""
    # In reality, you'd call a weather API
    weather_data = {
        "paris": "Sunny, 22°C",
        "london": "Cloudy, 15°C",
        "tokyo": "Rainy, 18°C"
    }
    return weather_data.get(city.lower(), f"Weather data not available for {city}")

# Define the tool
tools = [{
    "type": "function",
    "function": {
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
    }
}]

# User question
user_message = "What's the weather like in Paris?"

# Step 1: Call API with tools
response = client.chat.completions.create(
    model="gpt-4o-mini",
    messages=[{"role": "user", "content": user_message}],
    tools=tools
)

message = response.choices[0].message

# Step 2: Check if tool was called
if message.tool_calls:
    tool_call = message.tool_calls[0]

    # Step 3: Execute the tool
    args = json.loads(tool_call.function.arguments)
    result = get_weather(**args)

    # Step 4: Return result to LLM
    messages = [
        {"role": "user", "content": user_message},
        message,
        {
            "role": "tool",
            "tool_call_id": tool_call.id,
            "content": result
        }
    ]

    # Step 5: Get final response
    final_response = client.chat.completions.create(
        model="gpt-4o-mini",
        messages=messages,
        tools=tools
    )

    print(final_response.choices[0].message.content)
```

### Example 2: Multiple Tools - LLM Chooses

Give the LLM multiple tools and let it decide which one to use:

```python
def calculate(expression: str) -> str:
    """Safely evaluate a math expression"""
    try:
        result = eval(expression, {"__builtins__": {}}, {})
        return str(result)
    except Exception as e:
        return f"Error: {str(e)}"

def get_current_time(timezone: str = "UTC") -> str:
    """Get current time"""
    from datetime import datetime
    return datetime.now().strftime("%Y-%m-%d %H:%M:%S") + f" ({timezone})"

tools = [
    {
        "type": "function",
        "function": {
            "name": "get_weather",
            "description": "Get the current weather for a given city",
            "parameters": {
                "type": "object",
                "properties": {
                    "city": {"type": "string", "description": "City name"}
                },
                "required": ["city"]
            }
        }
    },
    {
        "type": "function",
        "function": {
            "name": "calculate",
            "description": "Evaluate a mathematical expression",
            "parameters": {
                "type": "object",
                "properties": {
                    "expression": {
                        "type": "string",
                        "description": "Math expression like '2 + 2' or '10 * 5'"
                    }
                },
                "required": ["expression"]
            }
        }
    },
    {
        "type": "function",
        "function": {
            "name": "get_current_time",
            "description": "Get the current time in a timezone",
            "parameters": {
                "type": "object",
                "properties": {
                    "timezone": {
                        "type": "string",
                        "description": "Timezone like UTC, EST, PST",
                        "default": "UTC"
                    }
                }
            }
        }
    }
]

# Test different questions
questions = [
    "What's the weather in London?",  # Uses get_weather
    "What is 15 * 24?",               # Uses calculate
    "What time is it?",               # Uses get_current_time
]

for question in questions:
    response = client.chat.completions.create(
        model="gpt-4o-mini",
        messages=[{"role": "user", "content": question}],
        tools=tools
    )

    message = response.choices[0].message

    if message.tool_calls:
        tool_call = message.tool_calls[0]
        function_name = tool_call.function.name
        args = json.loads(tool_call.function.arguments)

        # Execute the appropriate tool
        if function_name == "get_weather":
            result = get_weather(**args)
        elif function_name == "calculate":
            result = calculate(**args)
        elif function_name == "get_current_time":
            result = get_current_time(**args)

        print(f"Question: {question}")
        print(f"Tool used: {function_name}")
        print(f"Result: {result}\n")
```

The LLM automatically chooses the right tool based on the question. No explicit routing logic needed.

### Example 3: LLM Decides NOT to Call a Tool

The LLM only calls tools when necessary:

```python
tools = [{
    "type": "function",
    "function": {
        "name": "get_weather",
        "description": "Get the current weather for a city",
        "parameters": {
            "type": "object",
            "properties": {
                "city": {"type": "string"}
            },
            "required": ["city"]
        }
    }
}]

# Questions that don't need tools
questions = [
    "What is the capital of France?",
    "Explain what Python is",
    "Tell me a joke"
]

for question in questions:
    response = client.chat.completions.create(
        model="gpt-4o-mini",
        messages=[{"role": "user", "content": question}],
        tools=tools
    )

    message = response.choices[0].message

    if message.tool_calls:
        print(f"{question} -> Tool called (unexpected)")
    else:
        print(f"{question} -> No tool called (correct)")
```

The LLM understands when it has enough knowledge to answer directly.

### Example 4: Parallel Tool Calls

Call multiple tools at once for efficiency:

```python
# Question that needs multiple tool calls
question = "What's the weather in Paris, London, and Tokyo?"

response = client.chat.completions.create(
    model="gpt-4o-mini",
    messages=[{"role": "user", "content": question}],
    tools=tools,
    parallel_tool_calls=True  # Enable parallel execution
)

message = response.choices[0].message

if message.tool_calls:
    print(f"LLM called {len(message.tool_calls)} tools in parallel")

    # Execute all tool calls
    tool_results = []
    for tool_call in message.tool_calls:
        args = json.loads(tool_call.function.arguments)
        result = get_weather(**args)

        tool_results.append({
            "role": "tool",
            "tool_call_id": tool_call.id,
            "content": result
        })

    # Return all results
    messages = [
        {"role": "user", "content": question},
        message,
        *tool_results
    ]

    final_response = client.chat.completions.create(
        model="gpt-4o-mini",
        messages=messages,
        tools=tools
    )

    print(final_response.choices[0].message.content)
```

The LLM recognizes it needs to call the same tool three times and does so in parallel.

### Example 5: Error Handling

Tools can fail. Return errors to the LLM gracefully:

```python
question = "What is 10 divided by 0?"

response = client.chat.completions.create(
    model="gpt-4o-mini",
    messages=[{"role": "user", "content": question}],
    tools=[calculator_tool]
)

message = response.choices[0].message

if message.tool_calls:
    tool_call = message.tool_calls[0]
    args = json.loads(tool_call.function.arguments)

    # This will error (division by zero)
    result = calculate(**args)  # Returns "Error: division by zero"

    # Return error to LLM - it handles it gracefully
    messages = [
        {"role": "user", "content": question},
        message,
        {"role": "tool", "tool_call_id": tool_call.id, "content": result}
    ]

    final_response = client.chat.completions.create(
        model="gpt-4o-mini",
        messages=messages,
        tools=[calculator_tool]
    )

    print(final_response.choices[0].message.content)
    # "Division by zero is undefined in mathematics..."
```

The LLM receives the error message and explains the problem to the user naturally.

## Running the Example

The `example.py` file demonstrates all five tool calling patterns:

```bash
cd 04-tool-calling-basics
uv run example.py
```

You'll see examples of:
1. Basic tool calling (weather lookup)
2. Multiple tools (LLM chooses the right one)
3. No tool needed (LLM answers directly)
4. Parallel tool calls (multiple cities at once)
5. Error handling (division by zero)

## Key Takeaways

1. **Clear tool descriptions are critical**: The LLM uses descriptions to decide when to call tools. "Get weather" vs "Get current weather for a given city" makes a huge difference.

2. **Always return tool results**: The tool call is just a request. You must execute it and return the result for the LLM to answer.

3. **Use the 'required' field**: Tell the LLM which parameters are mandatory so it doesn't skip them.

4. **Descriptive tool names**: Use `get_weather` not `weather`. Use `search_database` not `search`. Clarity matters.

5. **Handle missing tool calls**: Not every request needs a tool. Check if `message.tool_calls` exists before executing.

6. **Return errors to the LLM**: When tools fail, send the error message back. The LLM can explain problems gracefully.

7. **Start with 2-3 tools**: Don't overwhelm the LLM with 20 tools at once. Add more as needed.

## Common Pitfalls

1. **Unclear tool descriptions**: "Get weather" doesn't tell the LLM enough. "Get current weather conditions for a given city using real-time data" is better.

2. **Not returning tool results**: Executing the tool but forgetting to send the result back to the LLM. The conversation hangs.

3. **Forgetting the 'required' field**: Without it, the LLM might skip required parameters and your tool call fails.

4. **Too many tools**: Starting with 10+ tools confuses the LLM. It struggles to choose. Start small.

5. **Not handling errors**: Tools fail. APIs go down. Databases timeout. Always return error messages to the LLM.

6. **Assuming tools are called**: Always check if `message.tool_calls` exists. Sometimes the LLM answers directly.

7. **Weak parameter descriptions**: "city" vs "city name in English, e.g., Paris, London, Tokyo". Be specific.

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

Build a multi-tool assistant that can help users with three different tasks:

1. **Weather tool**: Get current weather for any city
2. **Calculator tool**: Perform mathematical calculations
3. **Email validator tool**: Check if an email address is valid format

Create all three tools with proper descriptions and parameters. Test with at least 5 different questions that exercise each tool. Include:
- A question that needs each individual tool
- A question that needs no tools (tests the LLM's judgment)
- A question that causes an error (tests error handling)

Bonus: Add a fourth tool that combines results from multiple other tools (e.g., "If it's raining in London, send an email to...").

## Next Steps

You've mastered the fundamentals of tool calling. Now it's time to make your tools production-ready with validation, error handling, and clean abstractions.

Move to [05-tool-calling-pydantic](/Users/owainlewis/Projects/the-ai-engineer/ai-agents-from-scratch/05-tool-calling-pydantic) to learn how to use Pydantic for automatic schema generation, argument validation, and professional tool patterns.

## Resources

- [OpenAI Function Calling Guide](https://platform.openai.com/docs/guides/function-calling) - Official documentation for tool calling
- [JSON Schema Documentation](https://json-schema.org/) - Understanding tool parameter schemas
- [Function Calling Best Practices](https://platform.openai.com/docs/guides/function-calling/best-practices) - Tips from OpenAI
- [Tool Calling Examples](https://github.com/openai/openai-cookbook/blob/main/examples/How_to_call_functions_with_chat_models.ipynb) - OpenAI cookbook examples
