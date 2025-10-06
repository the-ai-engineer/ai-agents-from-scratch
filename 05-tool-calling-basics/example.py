"""
Lesson 04: Tool Calling Basics

Learn how to give AI the ability to call functions and take actions.
"""

import os
import json
from openai import OpenAI
from dotenv import load_dotenv

load_dotenv()
client = OpenAI(api_key=os.getenv("OPENAI_API_KEY"))


def get_weather(city: str) -> str:
    """Simulate getting weather for a city"""
    weather_data = {
        "paris": "Sunny, 22°C",
        "london": "Cloudy, 15°C",
        "tokyo": "Rainy, 18°C",
        "new york": "Partly cloudy, 20°C"
    }
    return weather_data.get(city.lower(), f"Weather data not available for {city}")


def calculate(expression: str) -> str:
    """Safely evaluate a mathematical expression"""
    try:
        # Use eval with restricted builtins for simple math
        # In production, use a proper math parser like py-expression-eval
        result = eval(expression, {"__builtins__": {}}, {})
        return str(result)
    except Exception as e:
        return f"Error: {str(e)}"


def basic_tool_call():
    """Example 1: Basic tool call

    Note: Uses Chat Completions API because tool calling requires it.
    """
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
    response = client.chat.completions.create(
        model="gpt-4o-mini",
        messages=[{"role": "user", "content": user_message}],
        tools=tools
    )

    message = response.choices[0].message

    if message.tool_calls:
        tool_call = message.tool_calls[0]
        print(f"Tool called: {tool_call.function.name}")

        # Step 2: Execute the tool
        args = json.loads(tool_call.function.arguments)
        result = get_weather(**args)
        print(f"Result: {result}")

        # Step 3: Return result to LLM for final response
        messages = [
            {"role": "user", "content": user_message},
            message,
            {"role": "tool", "tool_call_id": tool_call.id, "content": result}
        ]

        final_response = client.chat.completions.create(
            model="gpt-4o-mini",
            messages=messages,
            tools=tools
        )

        print(f"Assistant: {final_response.choices[0].message.content}")


def multiple_tools():
    """Example 2: LLM chooses between multiple tools

    Note: Uses Chat Completions API because tool calling requires it.
    """
    tools = [
        {
            "type": "function",
            "function": {
                "name": "get_weather",
                "description": "Get the current weather for a given city",
                "parameters": {
                    "type": "object",
                    "properties": {
                        "city": {"type": "string"}
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
                        "expression": {"type": "string"}
                    },
                    "required": ["expression"]
                }
            }
        }
    ]

    questions = [
        "What's the weather in London?",
        "What is 15 * 24?",
    ]

    for question in questions:
        print(f"\n\nUser: {question}")

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

            print(f"Tool called: {function_name}")

            # Execute the appropriate tool
            if function_name == "get_weather":
                result = get_weather(**args)
            elif function_name == "calculate":
                result = calculate(**args)

            print(f"Result: {result}")

            # Get final response
            messages_with_result = [
                {"role": "user", "content": question},
                message,
                {"role": "tool", "tool_call_id": tool_call.id, "content": result}
            ]

            final_response = client.chat.completions.create(
                model="gpt-4o-mini",
                messages=messages_with_result,
                tools=tools
            )

            print(f"Assistant: {final_response.choices[0].message.content}")


if __name__ == "__main__":
    basic_tool_call()
    multiple_tools()
