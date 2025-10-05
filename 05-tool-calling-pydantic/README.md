# Production-Ready Tool Calling with Pydantic

## What You'll Learn

In this lesson, you'll transform basic tool calling into production-grade systems using Pydantic for automatic validation, type safety, and clean architecture.

You've learned how to define tools manually with JSON Schema. Now you'll see how to eliminate boilerplate, catch errors before they happen, and build reusable tool abstractions that scale from 3 tools to 30.

This is how professional AI engineers build reliable tool systems that handle edge cases gracefully and maintain code quality as complexity grows.

## The Problem

Manual tool definitions are tedious and error-prone. Every tool requires writing JSON Schema by hand, manually parsing arguments, and hoping the LLM sends valid data.

Here's what you're dealing with without Pydantic:

```python
# Manual JSON schema (repetitive, error-prone)
tool = {
    "type": "function",
    "function": {
        "name": "get_weather",
        "description": "Get weather",
        "parameters": {
            "type": "object",
            "properties": {
                "city": {"type": "string", "description": "City name"}
            },
            "required": ["city"]
        }
    }
}

# Manual parsing (no validation)
args = json.loads(tool_call.function.arguments)
city = args.get("city")  # What if it's missing? What if it's the wrong type?
result = get_weather(city)
```

No type safety. No validation. No IDE autocomplete. If the LLM sends invalid data, your code crashes. If you forget to handle optional parameters, you get runtime errors. As you add more tools, maintenance becomes a nightmare.

## The Solution: Pydantic for Tools

Pydantic automates everything. You define argument schemas as Python classes, and Pydantic handles validation, type coercion, schema generation, and error messages automatically.

Here's the difference:

**Without Pydantic (manual):**
```python
# Write JSON schema by hand
args = json.loads(tool_call.function.arguments)
city = args.get("city")  # No validation, no types
result = get_weather(city)
```

**With Pydantic (automatic):**
```python
class GetWeatherArgs(BaseModel):
    city: str = Field(description="City name")

# Automatic validation and type safety
args = GetWeatherArgs.model_validate_json(tool_call.function.arguments)
result = get_weather(args.city)  # Type-safe, IDE autocomplete works
```

The schema generates automatically from your Pydantic model. Validation happens before execution. Type checking works in your IDE. Error messages tell you exactly what's wrong.

## How It Works

Pydantic integration follows three steps that replace all your manual work.

### Step 1: Define Argument Schemas with Pydantic

Instead of writing JSON Schema manually, create a Pydantic model:

```python
from pydantic import BaseModel, Field

class GetWeatherArgs(BaseModel):
    """Arguments for the get_weather tool"""
    city: str = Field(
        description="City name, e.g., 'Paris', 'London', 'Tokyo'"
    )
```

That's it. The schema generates automatically.

### Step 2: Generate OpenAI Tool Schema Automatically

Pydantic models convert to JSON Schema with one method:

```python
tool = {
    "type": "function",
    "function": {
        "name": "get_weather",
        "description": "Get current weather for a city",
        "parameters": GetWeatherArgs.model_json_schema()  # Automatic!
    }
}

# The schema is generated from your Pydantic model
print(tool["function"]["parameters"])
# {
#   "type": "object",
#   "properties": {
#     "city": {"type": "string", "description": "City name, e.g., 'Paris'..."}
#   },
#   "required": ["city"]
# }
```

No manual JSON Schema writing. The schema stays in sync with your Python types automatically.

### Step 3: Validate Arguments Before Execution

When the LLM calls your tool, validate the arguments instantly:

```python
from pydantic import ValidationError

if message.tool_calls:
    tool_call = message.tool_calls[0]

    try:
        # Validate arguments with Pydantic
        args = GetWeatherArgs.model_validate_json(tool_call.function.arguments)

        # Now args is typed and validated
        result = get_weather(args.city)

    except ValidationError as e:
        # Pydantic tells you exactly what's wrong
        result = f"Invalid arguments: {e}"
```

If the LLM sends a number instead of a string, Pydantic catches it. If required fields are missing, you get clear error messages. No crashes at runtime.

## Code Examples

### Example 1: Basic Pydantic Validation

Replace manual JSON Schema with automatic generation:

```python
from openai import OpenAI
from pydantic import BaseModel, Field
from pydantic import ValidationError
import json

client = OpenAI()

# Define argument schema with Pydantic
class GetWeatherArgs(BaseModel):
    city: str = Field(description="City name, e.g., 'Paris', 'London'")

# Your tool implementation
def get_weather(city: str) -> str:
    weather_db = {
        "paris": "Sunny, 22°C",
        "london": "Cloudy, 15°C",
        "tokyo": "Rainy, 18°C"
    }
    return weather_db.get(city.lower(), f"Weather data not available for {city}")

# Generate tool schema automatically
tool = {
    "type": "function",
    "function": {
        "name": "get_weather",
        "description": "Get current weather for a city",
        "parameters": GetWeatherArgs.model_json_schema()  # Automatic!
    }
}

# Make API call
response = client.chat.completions.create(
    model="gpt-4o-mini",
    messages=[{"role": "user", "content": "What's the weather in Paris?"}],
    tools=[tool]
)

message = response.choices[0].message

if message.tool_calls:
    tool_call = message.tool_calls[0]

    try:
        # Validate with Pydantic
        args = GetWeatherArgs.model_validate_json(tool_call.function.arguments)
        print(f"Validated: city={args.city}")

        # Execute with type-safe args
        result = get_weather(args.city)

    except ValidationError as e:
        result = f"Validation error: {e}"

    # Return result to LLM (same as before)
    messages = [
        {"role": "user", "content": "What's the weather in Paris?"},
        message,
        {"role": "tool", "tool_call_id": tool_call.id, "content": result}
    ]

    final_response = client.chat.completions.create(
        model="gpt-4o-mini",
        messages=messages,
        tools=[tool]
    )

    print(final_response.choices[0].message.content)
```

### Example 2: Reusable Tool Wrapper Class

Build a `Tool` class that handles schema generation and validation automatically:

```python
from pydantic import BaseModel, Field, ValidationError
from typing import Callable

class Tool(BaseModel):
    """Reusable tool wrapper with automatic schema generation"""
    name: str
    description: str
    args_schema: type[BaseModel]
    function: Callable

    class Config:
        arbitrary_types_allowed = True

    def to_openai_format(self) -> dict:
        """Convert to OpenAI tool format"""
        return {
            "type": "function",
            "function": {
                "name": self.name,
                "description": self.description,
                "parameters": self.args_schema.model_json_schema()
            }
        }

    def execute(self, arguments: str) -> str:
        """Execute tool with automatic validation"""
        try:
            # Validate arguments
            args = self.args_schema.model_validate_json(arguments)
            # Execute function with validated args
            return self.function(**args.model_dump())
        except ValidationError as e:
            return f"Validation error: {e}"
        except Exception as e:
            return f"Execution error: {e}"

# Define your argument schemas
class GetWeatherArgs(BaseModel):
    city: str = Field(description="City name")

class SendEmailArgs(BaseModel):
    to: str = Field(description="Recipient email address")
    subject: str = Field(description="Email subject line")
    body: str = Field(description="Email body content")

# Define your functions
def get_weather(city: str) -> str:
    return f"Weather in {city}: Sunny"

def send_email(to: str, subject: str, body: str) -> str:
    return f"Email sent to {to}"

# Create tools using the wrapper
weather_tool = Tool(
    name="get_weather",
    description="Get current weather for a city",
    args_schema=GetWeatherArgs,
    function=get_weather
)

email_tool = Tool(
    name="send_email",
    description="Send an email to a recipient",
    args_schema=SendEmailArgs,
    function=send_email
)

tools = [weather_tool, email_tool]
tools_openai = [t.to_openai_format() for t in tools]

# Use the tools
response = client.chat.completions.create(
    model="gpt-4o-mini",
    messages=[{"role": "user", "content": "What's the weather in London?"}],
    tools=tools_openai
)

message = response.choices[0].message

if message.tool_calls:
    for tool_call in message.tool_calls:
        # Find matching tool
        tool = next(t for t in tools if t.name == tool_call.function.name)

        # Execute with automatic validation
        result = tool.execute(tool_call.function.arguments)
        print(f"Tool: {tool.name}")
        print(f"Result: {result}")
```

Now adding new tools is trivial. Create the Pydantic model, write the function, wrap it with `Tool()`. Done.

### Example 3: Validation with Constraints

Pydantic validates data types and enforces constraints automatically:

```python
from pydantic import BaseModel, Field

class SearchDatabaseArgs(BaseModel):
    query: str = Field(description="Search query string")
    limit: int = Field(
        default=10,
        ge=1,        # Greater than or equal to 1
        le=100,      # Less than or equal to 100
        description="Maximum number of results (1-100)"
    )

def search_database(query: str, limit: int = 10) -> str:
    return f"Found {limit} results for '{query}'"

# Generate schema with constraints
tool = {
    "type": "function",
    "function": {
        "name": "search_database",
        "description": "Search the database",
        "parameters": SearchDatabaseArgs.model_json_schema()
    }
}

# The schema includes constraints automatically
print(tool["function"]["parameters"])
# {
#   "type": "object",
#   "properties": {
#     "query": {"type": "string", ...},
#     "limit": {"type": "integer", "minimum": 1, "maximum": 100, "default": 10}
#   },
#   "required": ["query"]
# }

# Validation enforces constraints
try:
    args = SearchDatabaseArgs.model_validate_json('{"query": "python", "limit": 5}')
    print(f"Valid: limit={args.limit}")  # 5 is between 1-100

    args = SearchDatabaseArgs.model_validate_json('{"query": "python", "limit": 200}')
except ValidationError as e:
    print("Validation failed: limit must be <= 100")
```

Pydantic enforces constraints before your function ever executes. No manual checking needed.

### Example 4: Error Handling - Validation vs Runtime Errors

Distinguish between validation errors (bad input) and runtime errors (execution failure):

```python
def get_weather(city: str) -> str:
    weather_db = {"paris": "Sunny, 22°C"}
    result = weather_db.get(city.lower())
    if result is None:
        raise ValueError(f"No weather data for {city}")
    return result

if message.tool_calls:
    tool_call = message.tool_calls[0]

    try:
        # Validation error: caught before execution
        args = GetWeatherArgs.model_validate_json(tool_call.function.arguments)

        try:
            # Runtime error: caught during execution
            result = get_weather(args.city)
        except ValueError as e:
            result = f"Tool error: {str(e)}"

    except ValidationError as e:
        result = f"Invalid arguments: {str(e)}"

    # Return error to LLM - it handles gracefully
    messages.append({
        "role": "tool",
        "tool_call_id": tool_call.id,
        "content": result
    })
```

Validation errors mean the LLM sent bad data. Runtime errors mean your tool encountered a problem. Both get returned to the LLM for graceful handling.

### Example 5: Optional Parameters with Defaults

Handle optional parameters cleanly with Pydantic defaults:

```python
from pydantic import BaseModel, Field
from typing import Literal

class GetWeatherAdvancedArgs(BaseModel):
    city: str = Field(description="City name")
    units: Literal["celsius", "fahrenheit"] = Field(
        default="celsius",
        description="Temperature units"
    )
    include_forecast: bool = Field(
        default=False,
        description="Include 5-day forecast"
    )

def get_weather_advanced(
    city: str,
    units: str = "celsius",
    include_forecast: bool = False
) -> str:
    temp = "22°C" if units == "celsius" else "72°F"
    forecast = " | Forecast: Sunny" if include_forecast else ""
    return f"{city}: {temp}{forecast}"

# Test with varying parameters
questions = [
    "What's the weather in Paris?",              # Uses defaults
    "What's the weather in London in fahrenheit?", # Overrides units
    "Weather in Tokyo with forecast",            # Overrides forecast
]

for question in questions:
    response = client.chat.completions.create(
        model="gpt-4o-mini",
        messages=[{"role": "user", "content": question}],
        tools=[tool]
    )

    if response.choices[0].message.tool_calls:
        tool_call = response.choices[0].message.tool_calls[0]
        args = GetWeatherAdvancedArgs.model_validate_json(
            tool_call.function.arguments
        )

        print(f"Question: {question}")
        print(f"Args: city={args.city}, units={args.units}, forecast={args.include_forecast}")
        print(f"Result: {get_weather_advanced(**args.model_dump())}\n")
```

Optional parameters work automatically. The LLM can omit them, and Pydantic fills in defaults.

## Running the Example

The `example.py` file demonstrates five production patterns:

```bash
cd 05-tool-calling-pydantic
uv run example.py
```

You'll see examples of:
1. Basic Pydantic validation
2. Reusable Tool wrapper class
3. Validation with constraints (min/max values)
4. Error handling (validation vs runtime errors)
5. Optional parameters with defaults

## Key Takeaways

1. **Pydantic provides free validation**: Use it. Type checking, constraint validation, and error messages come automatically.

2. **Field() descriptions help the LLM**: Better descriptions lead to better tool usage. "city name in English, e.g., Paris, London" is clearer than "city".

3. **Catch ValidationError separately**: Handle it differently from runtime errors. Validation errors mean bad input. Runtime errors mean execution failure.

4. **Don't over-abstract early**: The Tool wrapper class is useful at 3+ tools. For 1-2 tools, manual definitions are fine.

5. **Return errors to the LLM**: Both validation errors and runtime errors should go back to the LLM. It can explain problems to users naturally.

6. **Use Literal types for enums**: Force the LLM to choose from specific options like `Literal["celsius", "fahrenheit"]`.

7. **Schema generation is automatic**: Your Pydantic model is the source of truth. The OpenAI schema stays in sync automatically.

## Common Pitfalls

1. **Not catching ValidationError**: Let it crash instead of handling gracefully. Always catch and return to the LLM.

2. **Over-engineering too early**: Creating a Tool abstraction class for your first 1-2 tools. Start simple. Abstract when complexity justifies it.

3. **Weak Field() descriptions**: Using `city: str` instead of `city: str = Field(description="City name in English")`. The LLM needs context.

4. **Not testing validation**: Write tests with invalid arguments to ensure validation works as expected.

5. **Swallowing errors**: Catching exceptions but not logging or returning them. Always return error messages to the LLM.

6. **Forgetting arbitrary_types_allowed**: When storing Callables in Pydantic models, you need `class Config: arbitrary_types_allowed = True`.

7. **Not using constraints**: Pydantic supports `ge`, `le`, `min_length`, `max_length`, regex patterns, and more. Use them to validate input.

## When to Abstract with Tool Wrapper Class

| Scenario | Use Tool Wrapper? |
|----------|------------------|
| 1-2 simple tools | No - Keep it simple |
| 3-5 tools with similar patterns | Maybe - Consider abstracting |
| 6+ tools or complex validation | Yes - Abstract for maintainability |
| Reusing tools across projects | Yes - Build reusable patterns |
| Prototyping and experimenting | No - Stay flexible |

Abstract when complexity justifies it. Don't create frameworks for simple problems.

## Real-World Impact

Production AI systems use Pydantic for tools because it eliminates entire classes of bugs:

**Type Safety**: Catch argument type mismatches at validation time instead of crashing at runtime. Your IDE warns you before you even run the code.

**Automatic Documentation**: The Pydantic model is your documentation. Field descriptions become OpenAI schema descriptions. Comments aren't needed.

**Maintainability**: Change a field type in your Pydantic model and the schema updates automatically. No manual JSON Schema editing.

**Error Clarity**: ValidationError messages tell you exactly what's wrong: "field 'limit' must be <= 100". No cryptic errors.

**Testing**: Write unit tests for your Pydantic schemas. Ensure validation catches edge cases before production.

**Scaling**: Start with 3 tools. Add 10 more. The pattern stays consistent. No architectural rewrites needed.

Companies building production AI systems use Pydantic because reliability matters more than clever code. When your system processes thousands of tool calls daily, validation prevents expensive failures.

## Assignment

Build a production-ready multi-tool system using Pydantic:

Create three tools with complete validation:

1. **User lookup tool**: Search for users by email or ID with validated input
   - Email must be valid format
   - ID must be positive integer
   - At least one parameter required

2. **Calculate discount tool**: Compute discounts with constraints
   - Original price must be positive
   - Discount percent between 0-100
   - Optional tax rate (0-50%)

3. **Schedule meeting tool**: Create calendar events
   - Title required (1-100 characters)
   - Date in YYYY-MM-DD format
   - Duration in minutes (15-480)
   - Attendees list (1-20 people)

Implement the Tool wrapper class pattern. Test with valid inputs, invalid inputs, and edge cases. Ensure validation errors return clear messages to the LLM.

Bonus: Add unit tests for your Pydantic schemas that verify constraints work correctly.

## Next Steps

You've mastered production tool calling with validation and clean architecture. Now it's time to build systems that chain multiple tools together and reason across multiple steps.

Move to [06-agent-loop](/Users/owainlewis/Projects/the-ai-engineer/ai-agents-from-scratch/06-agent-loop) to learn how to build agent loops that use tools iteratively, handle multi-step reasoning, and solve complex problems autonomously.

## Resources

- [Pydantic Documentation](https://docs.pydantic.dev) - Complete guide to Pydantic models and validation
- [Pydantic Field Constraints](https://docs.pydantic.dev/latest/usage/fields/) - All available validation constraints
- [OpenAI Function Calling](https://platform.openai.com/docs/guides/function-calling) - Official tool calling documentation
- [Python Type Hints](https://docs.python.org/3/library/typing.html) - Master typing for better schemas
- [Pydantic JSON Schema](https://docs.pydantic.dev/latest/usage/json_schema/) - Understanding schema generation
