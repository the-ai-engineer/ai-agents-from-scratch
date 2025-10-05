"""
Lesson 07: The Agent Loop

Learn how agents can reason through multiple steps and chain tools together.
"""

import os
import json
from openai import OpenAI
from pydantic import BaseModel, Field
from dotenv import load_dotenv

load_dotenv()
client = OpenAI(api_key=os.getenv("OPENAI_API_KEY"))


def get_weather(city: str) -> str:
    """Get weather for a city"""
    weather_db = {
        "paris": "Sunny, 22°C",
        "london": "Cloudy, 15°C",
        "tokyo": "Rainy, 18°C",
    }
    return weather_db.get(city.lower(), f"Weather data not available for {city}")


def calculate(expression: str) -> str:
    """Calculate a mathematical expression"""
    try:
        # Use eval with restricted builtins for simple math
        # In production, use a proper math parser like py-expression-eval
        result = eval(expression, {"__builtins__": {}}, {})
        return str(result)
    except Exception as e:
        return f"Error: {str(e)}"


class GetWeatherArgs(BaseModel):
    city: str = Field(description="City name")


class CalculateArgs(BaseModel):
    expression: str = Field(description="Math expression to evaluate")


TOOLS = [
    {
        "type": "function",
        "function": {
            "name": "get_weather",
            "description": "Get current weather for a city",
            "parameters": GetWeatherArgs.model_json_schema()
        }
    },
    {
        "type": "function",
        "function": {
            "name": "calculate",
            "description": "Evaluate a mathematical expression",
            "parameters": CalculateArgs.model_json_schema()
        }
    }
]


def execute_tool(tool_name: str, arguments: dict) -> str:
    """Execute a tool by name"""
    if tool_name == "get_weather":
        return get_weather(**arguments)
    elif tool_name == "calculate":
        return calculate(**arguments)
    else:
        return f"Unknown tool: {tool_name}"


def agent_loop(user_message: str, max_iterations: int = 5) -> str:
    """
    The core agent loop pattern.
    Keeps calling the LLM and executing tools until we get a final answer.
    """
    input_list = [{"role": "user", "content": user_message}]

    for iteration in range(max_iterations):
        response = client.responses.create(
            model="gpt-4o-mini",
            input=input_list,
            tools=TOOLS
        )

        # If no tool calls, we have a final answer
        if not response.tool_calls:
            return response.output_text

        # Execute all tool calls
        print(f"Iteration {iteration + 1}: {len(response.tool_calls)} tool(s) called")

        # Add assistant message with tool calls to history
        input_list.append({
            "role": "assistant",
            "content": "",
            "tool_calls": [
                {"id": tc.id, "type": "function",
                 "function": {"name": tc.function.name, "arguments": tc.function.arguments}}
                for tc in response.tool_calls
            ]
        })

        for tool_call in response.tool_calls:
            tool_name = tool_call.function.name
            arguments = json.loads(tool_call.function.arguments)

            result = execute_tool(tool_name, arguments)
            print(f"  {tool_name}({arguments}) → {result}")

            input_list.append({
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
