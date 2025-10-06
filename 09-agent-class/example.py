"""
Lesson 08: The Agent Class

Learn how to abstract the agent loop into a reusable, production-ready class.
"""

import os
import json
from typing import Callable
from openai import OpenAI
from pydantic import BaseModel, Field
from dotenv import load_dotenv

load_dotenv()


class Tool(BaseModel):
    """Tool definition wrapper for clean organization"""
    name: str
    description: str
    args_schema: type[BaseModel]
    implementation: Callable

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
        """Execute the tool with arguments"""
        try:
            args_dict = json.loads(arguments)
            result = self.implementation(**args_dict)
            return str(result)
        except Exception as e:
            return f"Error: {str(e)}"


class Agent:
    """Reusable agent class with tool calling support"""

    def __init__(self, model: str = "gpt-4o-mini", max_iterations: int = 5):
        self.client = OpenAI(api_key=os.getenv("OPENAI_API_KEY"))
        self.model = model
        self.max_iterations = max_iterations
        self.tools: dict[str, Tool] = {}
        self.conversation_history = []

    def register_tool(self, name: str, function: Callable, args_schema: type[BaseModel], description: str):
        """Register a tool with the agent"""
        tool = Tool(
            name=name,
            description=description,
            args_schema=args_schema,
            implementation=function
        )
        self.tools[tool.name] = tool

    @property
    def tool_schemas(self):
        """Get OpenAI-formatted tool schemas"""
        return [t.to_openai_format() for t in self.tools.values()] if self.tools else None

    def chat(self, message: str) -> str:
        """Send a message and get a response (handles agent loop internally)

        Note: Uses Chat Completions API because Responses API doesn't support tool calling yet.
        """
        self.conversation_history.append({"role": "user", "content": message})

        for iteration in range(self.max_iterations):
            response = self.client.chat.completions.create(
                model=self.model,
                messages=self.conversation_history,
                tools=self.tool_schemas
            )

            message_obj = response.choices[0].message

            # If no tool calls, return final answer
            if not message_obj.tool_calls:
                answer = message_obj.content or "No response generated"
                self.conversation_history.append({"role": "assistant", "content": answer})
                return answer

            # Add assistant message with tool calls
            self.conversation_history.append(message_obj)

            # Execute tools
            for tool_call in message_obj.tool_calls:
                tool_name = tool_call.function.name

                if tool_name in self.tools:
                    result = self.tools[tool_name].execute(tool_call.function.arguments)
                else:
                    result = f"Tool '{tool_name}' not found"

                self.conversation_history.append({
                    "role": "tool",
                    "tool_call_id": tool_call.id,
                    "content": result
                })

        return "Max iterations reached"


# Tool implementations
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
    expression: str = Field(description="Mathematical expression")


def basic_agent_example():
    """Example 1: Basic agent usage"""
    agent = Agent(model="gpt-4o-mini", max_iterations=5)

    # Register tools
    agent.register_tool("get_weather", get_weather, GetWeatherArgs, "Get current weather for a city")
    agent.register_tool("calculate", calculate, CalculateArgs, "Evaluate a mathematical expression")

    # Use the agent
    questions = [
        "What's the weather in Paris?",
        "What is 15 * 24?",
        "What's the weather in London and what is 100 + 50?"
    ]

    for question in questions:
        print(f"\nUser: {question}")
        answer = agent.chat(question)
        print(f"Assistant: {answer}")


def conversation_memory_example():
    """Example 2: Agent maintains conversation context"""
    agent = Agent()
    agent.register_tool("get_weather", get_weather, GetWeatherArgs, "Get weather for a city")

    print("\n\n=== Conversation with Memory ===\n")

    # Multi-turn conversation
    print("User: What's the weather in Paris?")
    answer1 = agent.chat("What's the weather in Paris?")
    print(f"Assistant: {answer1}")

    print("\nUser: What about London?")
    answer2 = agent.chat("What about London?")
    print(f"Assistant: {answer2}")

    print("\nUser: Which one is warmer?")
    answer3 = agent.chat("Which one is warmer?")
    print(f"Assistant: {answer3}")


if __name__ == "__main__":
    basic_agent_example()
    conversation_memory_example()
