"""
Lesson 04: Tool Calling - Advanced Examples

Learn how to build production-ready tools with validation and error handling
using the @tool decorator pattern.
"""

import os
import json
import inspect
from typing import Callable, Any, get_type_hints
from openai import OpenAI
from pydantic import BaseModel, Field, ValidationError, create_model
from dotenv import load_dotenv

load_dotenv()
client = OpenAI(api_key=os.getenv("OPENAI_API_KEY"))


# ============================================================================
# @tool Decorator - Automatic Schema Generation
# ============================================================================

def extract_function_params(func: Callable) -> tuple[dict, list[str]]:
    """Extract parameter fields and required list from a function."""
    sig = inspect.signature(func)
    type_hints = get_type_hints(func)

    fields = {}
    required = []

    for param_name, param in sig.parameters.items():
        if param_name == "self":
            continue

        param_type = type_hints.get(param_name, str)

        if param.default == inspect.Parameter.empty:
            default = ...
            required.append(param_name)
        else:
            default = None
            param_type = param_type | None

        fields[param_name] = (param_type, default)

    return fields, required


def create_json_schema(func: Callable) -> dict[str, Any]:
    """Create a JSON schema from a function signature."""
    fields, required = extract_function_params(func)

    args_schema = create_model(f"{func.__name__}_args", **fields)
    schema = args_schema.model_json_schema()
    schema["required"] = required
    schema["additionalProperties"] = False

    return schema


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


def tool(func: Callable) -> Callable:
    """Decorator to mark a function as a tool with automatic schema generation."""
    func.tool = Tool.from_function(func)
    return func


def basic_tool_decorator():
    """Example 1: Using @tool decorator for automatic schema generation"""
    print("=" * 70)
    print("EXAMPLE 1: Basic @tool Decorator")
    print("=" * 70)

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

    # Schema generated automatically from function signature and docstring!
    tools = [get_weather.tool.to_openai_format()]

    print("Auto-generated schema:")
    print(json.dumps(tools[0]["function"]["parameters"], indent=2))

    question = "What's the weather in Paris?"
    print(f"\nUser: {question}")

    response = client.responses.create(
        model="gpt-4o-mini",
        input=[{"role": "user", "content": question}],
        tools=tools
    )

    # Check for tool calls in output
    for output in response.output:
        if output.type == "function_call":
            print(f"Tool called: {output.name}")

            # Execute the tool
            args = json.loads(output.arguments)
            result = get_weather(**args)
            print(f"Result: {result}")

            # Return to LLM
            input_with_result = [
                {"role": "user", "content": question},
                {"type": "function_call_output", "call_id": output.call_id, "output": result}
            ]

            final_response = client.responses.create(
                model="gpt-4o-mini",
                input=input_with_result,
                tools=tools
            )

            print(f"Assistant: {final_response.output_text}")
            break


def multiple_tools_with_defaults():
    """Example 2: Multiple tools with optional parameters"""
    print("\n" + "=" * 70)
    print("EXAMPLE 2: Multiple Tools with Optional Parameters")
    print("=" * 70)

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

    @tool
    def send_email(to: str, subject: str, body: str = "No body provided") -> str:
        """Send an email to a recipient.

        Args:
            to: Recipient email address
            subject: Email subject line
            body: Email body content (optional)
        """
        return f"Email sent to {to} with subject '{subject}' and body: {body}"

    # Create tool registry
    tool_registry = {
        "get_weather": get_weather,
        "send_email": send_email,
    }

    # Generate schemas automatically
    tools = [func.tool.to_openai_format() for func in tool_registry.values()]

    question = "What's the weather in London? If it's cloudy, send an email to john@example.com with subject 'Weather Alert'."
    print(f"User: {question}")

    response = client.responses.create(
        model="gpt-4o-mini",
        input=[{"role": "user", "content": question}],
        tools=tools
    )

    # Check for tool calls in output
    tool_call_count = sum(1 for output in response.output if output.type == "function_call")

    if tool_call_count > 0:
        print(f"\n{tool_call_count} tool(s) called")

        input_messages = [{"role": "user", "content": question}]

        for output in response.output:
            if output.type == "function_call":
                print(f"\nExecuting: {output.name}")

                # Execute from registry
                args = json.loads(output.arguments)
                result = tool_registry[output.name](**args)
                print(f"Result: {result}")

                input_messages.append({
                    "type": "function_call_output",
                    "call_id": output.call_id,
                    "output": result
                })

        final_response = client.responses.create(
            model="gpt-4o-mini",
            input=input_messages,
            tools=tools
        )

        print(f"\nAssistant: {final_response.output_text}")


def tool_with_validation():
    """Example 3: Tool with type validation and error handling"""
    print("\n" + "=" * 70)
    print("EXAMPLE 3: Tool with Type Validation")
    print("=" * 70)

    @tool
    def calculate(expression: str) -> str:
        """Evaluate a mathematical expression.

        Args:
            expression: A mathematical expression like "2 + 2" or "15 * 24"
        """
        try:
            result = eval(expression, {"__builtins__": {}}, {})
            return str(result)
        except Exception as e:
            return f"Error: {str(e)}"

    @tool
    def search_database(query: str, limit: int = 10) -> str:
        """Search the database for matching records.

        Args:
            query: Search query string
            limit: Maximum number of results to return (default: 10)
        """
        return f"Found {limit} results for query: '{query}'"

    # Create tool registry
    tool_registry = {
        "calculate": calculate,
        "search_database": search_database,
    }

    # Generate schemas
    tools = [func.tool.to_openai_format() for func in tool_registry.values()]

    questions = [
        "What is 25 * 48?",
        "Search the database for 'python tutorials' and limit to 5 results",
    ]

    for question in questions:
        print(f"\nUser: {question}")

        response = client.responses.create(
            model="gpt-4o-mini",
            input=[{"role": "user", "content": question}],
            tools=tools
        )

        for output in response.output:
            if output.type == "function_call":
                print(f"Tool called: {output.name}")

                # Execute from registry
                args = json.loads(output.arguments)
                result = tool_registry[output.name](**args)
                print(f"Result: {result}")

                input_with_result = [
                    {"role": "user", "content": question},
                    {"type": "function_call_output", "call_id": output.call_id, "output": result}
                ]

                final_response = client.responses.create(
                    model="gpt-4o-mini",
                    input=input_with_result,
                    tools=tools
                )

                print(f"Assistant: {final_response.output_text}")
                break


if __name__ == "__main__":
    basic_tool_decorator()
    multiple_tools_with_defaults()
    tool_with_validation()
