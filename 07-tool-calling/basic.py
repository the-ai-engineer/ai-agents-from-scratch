"""
Lesson 04: Tool Calling - Basic Examples

Learn how to give AI the ability to call functions and take actions.
This lesson shows three approaches:
1. Manual tool definition (understand the raw format)
2. @tool decorator with single tool
3. @tool decorator with multiple tools

Run from project root:
    uv run python 07-tool-calling/basic.py
"""

import os
import json
import sys
from pathlib import Path
from openai import OpenAI
from dotenv import load_dotenv

# Add project root to path for imports
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

from src.tool import tool

load_dotenv()
client = OpenAI(api_key=os.getenv("OPENAI_API_KEY"))


##=================================================##
## Example 1: Manual tool definition
##=================================================##

def get_weather(city: str) -> str:
    """Get the current weather for a given city."""
    weather_data = {
        "paris": "Sunny, 22°C",
        "london": "Cloudy, 15°C",
        "tokyo": "Rainy, 18°C",
    }
    return weather_data.get(city.lower(), f"Weather data not available for {city}")


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

# Step 1: LLM decides to call tool
response = client.responses.create(
    model="gpt-4o-mini",
    input=[{"role": "user", "content": user_message}],
    tools=tools
)

# Step 2: Execute the tool
for output in response.output:
    if output.type == "function_call":
        args = json.loads(output.arguments)
        result = get_weather(**args)

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

        # final_response.output_text


##=================================================##
## Example 2: Using @tool decorator (automatic schema)
##=================================================##

@tool
def get_weather_auto(city: str) -> str:
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


# Schema generated automatically from function signature and docstring
tools = [get_weather_auto.tool.to_dict()]

user_message = "What's the weather like in Tokyo?"

response = client.responses.create(
    model="gpt-4o-mini",
    input=[{"role": "user", "content": user_message}],
    tools=tools
)

for output in response.output:
    if output.type == "function_call":
        args = json.loads(output.arguments)
        result = get_weather_auto(**args)

        input_with_result = [
            {"role": "user", "content": user_message},
            {"type": "function_call_output", "call_id": output.call_id, "output": result}
        ]

        final_response = client.responses.create(
            model="gpt-4o-mini",
            input=input_with_result,
            tools=tools
        )

        # final_response.output_text


##=================================================##
## Example 3: Multiple tools with registry pattern
##=================================================##

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


# Create a registry of tools
tool_registry = {
    "get_weather_multi": get_weather_multi,
    "calculate": calculate,
}

# Generate schemas automatically
tools = [func.tool.to_dict() for func in tool_registry.values()]

user_message = "What's the weather in London and what is 15 * 24?"

response = client.responses.create(
    model="gpt-4o-mini",
    input=[{"role": "user", "content": user_message}],
    tools=tools
)

# Execute all requested tool calls
for output in response.output:
    if output.type == "function_call":
        args = json.loads(output.arguments)
        result = tool_registry[output.name](**args)

        input_with_result = [
            {"role": "user", "content": user_message},
            {"type": "function_call_output", "call_id": output.call_id, "output": result}
        ]

        final_response = client.responses.create(
            model="gpt-4o-mini",
            input=input_with_result,
            tools=tools
        )

        # final_response.output_text
