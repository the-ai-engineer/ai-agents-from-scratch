"""
Lesson 07: The Agent Loop

Learn how agents can reason through multiple steps and chain tools together.
"""

import os
import json
import sys
from pathlib import Path
from openai import OpenAI
from dotenv import load_dotenv

# Add parent directory to path to import tool_helpers
sys.path.insert(0, str(Path(__file__).parent.parent))
from tool_helpers import Tool, safe_calculate

load_dotenv()
client = OpenAI(api_key=os.getenv("OPENAI_API_KEY"))


def get_weather(city: str) -> str:
    """Get current weather for a city

    Args:
        city: City name (e.g., 'Paris', 'London', 'Tokyo')
    """
    weather_db = {
        "paris": "Sunny, 22°C",
        "london": "Cloudy, 15°C",
        "tokyo": "Rainy, 18°C",
    }
    return weather_db.get(city.lower(), f"Weather data not available for {city}")


def calculate(expression: str) -> str:
    """Safely evaluate a mathematical expression

    Args:
        expression: Math expression to evaluate (e.g., '2+2', '157.09*493.89')
    """
    try:
        result = safe_calculate(expression)
        return str(result)
    except ValueError as e:
        return f"Error: {str(e)}"


# Create tools using the simplified Tool class
# Schemas are automatically generated from function signatures and docstrings!
weather_tool = Tool(get_weather)
calculator_tool = Tool(calculate)

# Convert to OpenAI format
TOOLS = [
    weather_tool.get_schema(),
    calculator_tool.get_schema()
]

# Create a tool registry for easy lookup
TOOL_REGISTRY = {
    "get_weather": weather_tool,
    "calculate": calculator_tool
}


def execute_tool(tool_name: str, arguments: dict) -> str:
    """Execute a tool by name"""
    tool = TOOL_REGISTRY.get(tool_name)
    if tool:
        result = tool.execute(**arguments)
        # Tool.execute() returns a dict, extract the result
        return result.get("result", str(result))
    else:
        return f"Unknown tool: {tool_name}"


def agent_loop(user_message: str, max_iterations: int = 5) -> str:
    """
    The core agent loop pattern.
    Keeps calling the LLM and executing tools until we get a final answer.

    Note: Uses Chat Completions API because Responses API doesn't support tool calling yet.
    """
    messages = [{"role": "user", "content": user_message}]

    for iteration in range(max_iterations):
        response = client.chat.completions.create(
            model="gpt-4o-mini",
            messages=messages,
            tools=TOOLS
        )

        message = response.choices[0].message

        # If no tool calls, we have a final answer
        if not message.tool_calls:
            return message.content or "No response generated"

        # Execute all tool calls
        print(f"Iteration {iteration + 1}: {len(message.tool_calls)} tool(s) called")

        # Add assistant message with tool calls to history
        messages.append(message)

        for tool_call in message.tool_calls:
            tool_name = tool_call.function.name
            arguments = json.loads(tool_call.function.arguments)

            result = execute_tool(tool_name, arguments)
            print(f"  {tool_name}({arguments}) → {result}")

            messages.append({
                "role": "tool",
                "tool_call_id": tool_call.id,
                "content": result
            })

    return "Max iterations reached"


def multi_tool_example():
    """Example 1: Agent uses multiple tools"""
    question = "What's the weather in Paris and London? Which city is warmer?"

    print(f"User: {question}\n")
    answer = agent_loop(question)
    print(f"\nFinal answer: {answer}")


def tool_chaining_example():
    """Example 2: Agent chains tools (uses output of one as input to another)"""
    question = "Calculate 15 * 24, then multiply that result by 3"

    print(f"\n\nUser: {question}\n")
    answer = agent_loop(question)
    print(f"\nFinal answer: {answer}")


if __name__ == "__main__":
    multi_tool_example()
    tool_chaining_example()
