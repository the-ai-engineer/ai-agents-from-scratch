"""
Lesson 05: Production Tool Calling with Pydantic

Learn how to build production-ready tools with validation and error handling.
"""

import os
import json
from openai import OpenAI
from pydantic import BaseModel, Field, ValidationError
from typing import Callable
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


def send_email(to: str, subject: str, body: str) -> str:
    """Simulate sending an email"""
    return f"Email sent to {to} with subject '{subject}'"


class GetWeatherArgs(BaseModel):
    """Arguments for get_weather tool"""
    city: str = Field(description="City name, e.g., 'Paris', 'London', 'Tokyo'")


class SendEmailArgs(BaseModel):
    """Arguments for send_email tool"""
    to: str = Field(description="Recipient email address")
    subject: str = Field(description="Email subject line")
    body: str = Field(description="Email body content")


def basic_pydantic_validation():
    """Example 1: Pydantic auto-generates schemas and validates

    Note: Uses Chat Completions API because tool calling requires it.
    """
    tool = {
        "type": "function",
        "function": {
            "name": "get_weather",
            "description": "Get current weather for a city",
            "parameters": GetWeatherArgs.model_json_schema()
        }
    }

    print("Auto-generated schema:")
    print(json.dumps(tool["function"]["parameters"], indent=2))

    question = "What's the weather in Paris?"
    print(f"\nUser: {question}")

    response = client.chat.completions.create(
        model="gpt-4o-mini",
        messages=[{"role": "user", "content": question}],
        tools=[tool]
    )

    message = response.choices[0].message

    if message.tool_calls:
        tool_call = message.tool_calls[0]

        # Validate with Pydantic
        args = GetWeatherArgs.model_validate_json(tool_call.function.arguments)
        print(f"Validated args: city={args.city}")

        result = get_weather(args.city)
        print(f"Result: {result}")

        # Return to LLM
        messages_with_result = [
            {"role": "user", "content": question},
            message,
            {"role": "tool", "tool_call_id": tool_call.id, "content": result}
        ]

        final_response = client.chat.completions.create(
            model="gpt-4o-mini",
            messages=messages_with_result,
            tools=[tool]
        )

        print(f"Assistant: {final_response.choices[0].message.content}")


def tool_wrapper_class():
    """Example 2: Reusable Tool wrapper class"""
    class Tool(BaseModel):
        """Reusable tool wrapper"""
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
            """Execute tool with validation"""
            try:
                # Validate arguments with Pydantic
                args = self.args_schema.model_validate_json(arguments)
                return self.function(**args.model_dump())
            except ValidationError as e:
                return f"Validation error: {e}"
            except Exception as e:
                return f"Execution error: {e}"

    # Define tools
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

    question = "What's the weather in London? If it's cloudy, send an email to john@example.com telling him to bring a jacket."
    print(f"\nUser: {question}")

    response = client.chat.completions.create(
        model="gpt-4o-mini",
        messages=[{"role": "user", "content": question}],
        tools=tools_openai
    )

    message = response.choices[0].message

    if message.tool_calls:
        print(f"{len(message.tool_calls)} tool(s) called")

        messages = [
            {"role": "user", "content": question},
            message
        ]

        for tool_call in message.tool_calls:
            tool = next(t for t in tools if t.name == tool_call.function.name)

            print(f"Executing: {tool.name}")
            result = tool.execute(tool_call.function.arguments)
            print(f"Result: {result}")

            messages.append({
                "role": "tool",
                "tool_call_id": tool_call.id,
                "content": result
            })

        final_response = client.chat.completions.create(
            model="gpt-4o-mini",
            messages=messages,
            tools=tools_openai
        )

        print(f"Assistant: {final_response.choices[0].message.content}")


if __name__ == "__main__":
    basic_pydantic_validation()
    tool_wrapper_class()
