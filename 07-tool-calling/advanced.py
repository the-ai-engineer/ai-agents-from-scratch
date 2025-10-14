"""
Lesson 04: Tool Calling - Advanced Examples

Production-ready tool calling patterns with error handling and multiple tools.

Run from project root:
    uv run python 07-tool-calling/advanced.py
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
## Example 1: Multiple tools with optional parameters
##=================================================##

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
tools = [func.tool.to_dict() for func in tool_registry.values()]

question = "What's the weather in London? If it's cloudy, send an email to john@example.com with subject 'Weather Alert'."

response = client.responses.create(
    model="gpt-4o-mini",
    input=[{"role": "user", "content": question}],
    tools=tools
)

# Execute all requested tool calls
input_messages = [{"role": "user", "content": question}]

for output in response.output:
    if output.type == "function_call":
        args = json.loads(output.arguments)
        result = tool_registry[output.name](**args)

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

# final_response.output_text


##=================================================##
## Example 2: Error handling in tools
##=================================================##

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
tools = [func.tool.to_dict() for func in tool_registry.values()]

question = "What is 25 * 48?"

response = client.responses.create(
    model="gpt-4o-mini",
    input=[{"role": "user", "content": question}],
    tools=tools
)

for output in response.output:
    if output.type == "function_call":
        args = json.loads(output.arguments)
        result = tool_registry[output.name](**args)

        input_with_result = [
            {"role": "user", "content": question},
            {"type": "function_call_output", "call_id": output.call_id, "output": result}
        ]

        final_response = client.responses.create(
            model="gpt-4o-mini",
            input=input_with_result,
            tools=tools
        )

        # final_response.output_text
