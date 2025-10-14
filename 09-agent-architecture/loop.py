"""
Lesson 06: Agent Architecture - Part 1: The Agent Loop

Learn how agents can reason through multiple steps and chain tools together.
This example shows the raw agent loop implementation before abstraction.
"""

import os
import json
import sys
from pathlib import Path
from openai import OpenAI
from dotenv import load_dotenv

# Add parent directory to path to import agents framework
sys.path.insert(0, str(Path(__file__).parent.parent))
from agents import tool, ConversationMemory

load_dotenv()
client = OpenAI(api_key=os.getenv("OPENAI_API_KEY"))


# Define tools using @tool decorator
@tool
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


@tool
def calculate(expression: str) -> str:
    """Evaluate a mathematical expression

    Args:
        expression: Math expression to evaluate (e.g., '2+2', '157.09*493.89')
    """
    try:
        # Note: eval() is used here for simplicity in this teaching example
        # Production code should use a proper math parser library
        result = eval(expression, {"__builtins__": {}}, {})
        return str(result)
    except Exception as e:
        return f"Error: {str(e)}"


# Create tool registry using @tool decorated functions
TOOLS = [get_weather, calculate]
TOOL_REGISTRY = {func.__name__: func for func in TOOLS}


def _execute_tool(function_call) -> str:
    """Execute a tool call and return the result as a string."""
    tool_name = function_call.name

    if tool_name not in TOOL_REGISTRY:
        return json.dumps({"error": f"Tool '{tool_name}' not found"})

    try:
        args = json.loads(function_call.arguments)
        result = TOOL_REGISTRY[tool_name](**args)
        return json.dumps({"result": result})
    except Exception as e:
        return json.dumps({"error": str(e)})


def _has_function_calls(output: list) -> bool:
    """Check if output contains any function calls."""
    return any(item.type == "function_call" for item in output)


def _extract_text_response(output: list) -> str:
    """Extract text content from output items."""
    for item in output:
        if item.type == "message":
            for content in item.content:
                if content.type == "output_text":
                    return content.text
    return "No response generated"


def agent_loop(user_message: str, max_iterations: int = 5) -> str:
    """
    The core agent loop pattern.
    Keeps calling the LLM and executing tools until we get a final answer.
    """
    # Use ConversationMemory to manage history
    memory = ConversationMemory()
    memory.add_user_message(user_message)

    # Get tool schemas in OpenAI format
    tool_schemas = [func.tool.to_openai_format() for func in TOOLS]

    for iteration in range(max_iterations):
        response = client.responses.create(
            model="gpt-4o-mini",
            input=memory.get_items(),
            tools=tool_schemas
        )

        # Check if LLM wants to use tools
        if not _has_function_calls(response.output):
            # No tools needed - we have our final answer!
            answer = _extract_text_response(response.output)
            memory.add_response_output(response.output)
            return answer

        # LLM wants to use tools - add output to memory first
        memory.add_response_output(response.output)

        # Count and execute all function calls
        tool_calls_count = sum(1 for item in response.output if item.type == "function_call")
        print(f"Iteration {iteration + 1}: {tool_calls_count} tool(s) called")

        for item in response.output:
            if item.type == "function_call":
                tool_name = item.name
                arguments = json.loads(item.arguments)

                result = _execute_tool(item)
                result_data = json.loads(result)
                print(f"  {tool_name}({arguments}) → {result_data.get('result', result_data.get('error'))}")

                # Add tool result to memory
                memory.add_function_output(item.call_id, result)

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
