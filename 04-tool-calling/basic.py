"""
Lesson 04: Tool Calling - Basic Examples

Learn how to give AI the ability to call functions and take actions.
"""

import os
import json
import inspect
from typing import Callable, Any, get_type_hints
from openai import OpenAI
from dotenv import load_dotenv
from pydantic import BaseModel, create_model

load_dotenv()
client = OpenAI(api_key=os.getenv("OPENAI_API_KEY"))


# ============================================================================
# @tool Decorator - Clean Pattern for Tool Definition
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
    """Decorator to mark a function as a tool."""
    func.tool = Tool.from_function(func)
    return func


# ============================================================================
# Tool Functions
# ============================================================================

def get_weather(city: str) -> str:
    """Get the current weather for a given city.

    Args:
        city: The city name, e.g., Paris, London, Tokyo
    """
    weather_data = {
        "paris": "Sunny, 22°C",
        "london": "Cloudy, 15°C",
        "tokyo": "Rainy, 18°C",
        "new york": "Partly cloudy, 20°C"
    }
    return weather_data.get(city.lower(), f"Weather data not available for {city}")


def calculate(expression: str) -> str:
    """Evaluate a mathematical expression.

    Args:
        expression: A mathematical expression like "2 + 2" or "15 * 24"
    """
    try:
        # Use eval with restricted builtins for simple math
        # In production, use a proper math parser like py-expression-eval
        result = eval(expression, {"__builtins__": {}}, {})
        return str(result)
    except Exception as e:
        return f"Error: {str(e)}"


def basic_tool_call_manual():
    """Example 1: Manual tool definition (traditional approach)"""
    print("=" * 70)
    print("EXAMPLE 1: Manual Tool Definition")
    print("=" * 70)

    # Manual JSON schema definition
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

    user_message = "What's the weather like in Paris?"
    print(f"User: {user_message}")

    # Step 1: LLM decides to call tool
    response = client.responses.create(
        model="gpt-4o-mini",
        input=[{"role": "user", "content": user_message}],
        tools=tools
    )

    # Check for tool calls in output
    for output in response.output:
        if output.type == "function_call":
            print(f"Tool called: {output.name}")

            # Step 2: Execute the tool
            args = json.loads(output.arguments)
            result = get_weather(**args)
            print(f"Result: {result}")

            # Step 3: Return result to LLM for final response
            input_with_result = [
                {"role": "user", "content": user_message},
                {"type": "function_call_output", "call_id": output.call_id, "output": result}
            ]

            final_response = client.responses.create(
                model="gpt-4o-mini",
                input=input_with_result,
                tools=tools
            )

            print(f"Assistant: {final_response.output_text}")
            break


def basic_tool_call_decorator():
    """Example 2: Using @tool decorator (improved approach)"""
    print("\n" + "=" * 70)
    print("EXAMPLE 2: Using @tool Decorator (Improved)")
    print("=" * 70)

    # Decorate the function
    @tool
    def get_weather_decorated(city: str) -> str:
        """Get the current weather for a given city.

        Args:
            city: The city name, e.g., Paris, London, Tokyo
        """
        weather_data = {
            "paris": "Sunny, 22°C",
            "london": "Cloudy, 15°C",
            "tokyo": "Rainy, 18°C",
        }
        return weather_data.get(city.lower(), f"Weather data not available for {city}")

    # Schema generated automatically!
    tools = [get_weather_decorated.tool.to_openai_format()]

    print("Auto-generated schema:")
    print(json.dumps(tools[0]["function"]["parameters"], indent=2))

    user_message = "What's the weather like in Tokyo?"
    print(f"\nUser: {user_message}")

    response = client.responses.create(
        model="gpt-4o-mini",
        input=[{"role": "user", "content": user_message}],
        tools=tools
    )

    for output in response.output:
        if output.type == "function_call":
            print(f"Tool called: {output.name}")

            # Execute the tool
            args = json.loads(output.arguments)
            result = get_weather_decorated(**args)
            print(f"Result: {result}")

            # Return result to LLM
            input_with_result = [
                {"role": "user", "content": user_message},
                {"type": "function_call_output", "call_id": output.call_id, "output": result}
            ]

            final_response = client.responses.create(
                model="gpt-4o-mini",
                input=input_with_result,
                tools=tools
            )

            print(f"Assistant: {final_response.output_text}")
            break


def multiple_tools_decorator():
    """Example 3: Multiple tools with @tool decorator"""
    print("\n" + "=" * 70)
    print("EXAMPLE 3: Multiple Tools with @tool Decorator")
    print("=" * 70)

    # Decorate both functions
    @tool
    def get_weather_multi(city: str) -> str:
        """Get the current weather for a given city.

        Args:
            city: The city name, e.g., Paris, London, Tokyo
        """
        weather_data = {
            "paris": "Sunny, 22°C",
            "london": "Cloudy, 15°C",
            "tokyo": "Rainy, 18°C",
        }
        return weather_data.get(city.lower(), f"Weather data not available for {city}")

    @tool
    def calculate_multi(expression: str) -> str:
        """Evaluate a mathematical expression.

        Args:
            expression: A mathematical expression like "2 + 2" or "15 * 24"
        """
        try:
            result = eval(expression, {"__builtins__": {}}, {})
            return str(result)
        except Exception as e:
            return f"Error: {str(e)}"

    # Create a registry of tools
    tool_registry = {
        "get_weather_multi": get_weather_multi,
        "calculate_multi": calculate_multi,
    }

    # Generate schemas automatically
    tools = [func.tool.to_openai_format() for func in tool_registry.values()]

    questions = [
        "What's the weather in London?",
        "What is 15 * 24?",
    ]

    for question in questions:
        print(f"\nUser: {question}")

        response = client.responses.create(
            model="gpt-4o-mini",
            input=[{"role": "user", "content": question}],
            tools=tools
        )

        # Check for tool calls in output
        for output in response.output:
            if output.type == "function_call":
                function_name = output.name
                args = json.loads(output.arguments)

                print(f"Tool called: {function_name}")

                # Execute from registry
                result = tool_registry[function_name](**args)
                print(f"Result: {result}")

                # Get final response
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
    basic_tool_call_manual()
    basic_tool_call_decorator()
    multiple_tools_decorator()
