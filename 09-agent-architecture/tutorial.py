"""
Lesson 09: Building an AI Agent from Scratch

Learn how to build an autonomous AI agent that uses tools in a loop.

Key concept: "Agents are models using tools in a loop"

Run from project root:
    uv run python 09-agent-architecture/tutorial.py

Or use in Jupyter/IPython for interactive exploration.
"""

import json
from openai import OpenAI
from dotenv import load_dotenv

load_dotenv()


##=================================================##
## Step 1: Define Tools
##=================================================##


def get_weather(city: str) -> str:
    """Get current weather for a city."""
    weather_db = {
        "paris": "Sunny, 22°C",
        "london": "Cloudy, 15°C",
        "tokyo": "Rainy, 18°C",
    }
    return weather_db.get(city.lower(), f"Weather data not available for {city}")


def get_time(city: str) -> str:
    """Get current time for a city."""
    time_db = {
        "paris": "14:30",
        "london": "13:30",
        "tokyo": "21:30",
    }
    return time_db.get(city.lower(), f"Time data not available for {city}")


# Tool schemas in OpenAI format
TOOL_SCHEMAS = [
    {
        "type": "function",
        "name": "get_weather",
        "description": "Get current weather for a city",
        "parameters": {
            "type": "object",
            "properties": {"city": {"type": "string", "description": "City name"}},
            "required": ["city"],
        },
    },
    {
        "type": "function",
        "name": "get_time",
        "description": "Get current time for a city",
        "parameters": {
            "type": "object",
            "properties": {"city": {"type": "string", "description": "City name"}},
            "required": ["city"],
        },
    },
]

# Tool registry
TOOLS = {
    "get_weather": get_weather,
    "get_time": get_time,
}


##=================================================##
## Step 2: Build the Agent Class
##=================================================##


class Agent:
    """
    An autonomous AI agent that uses tools in a loop.

    The agent loop:
    1. Call LLM with available tools
    2. If LLM calls tools, execute them and add results to history
    3. Repeat until LLM gives final answer
    """

    def __init__(self, model: str = "gpt-4o-mini"):
        self.client = OpenAI()
        self.model = model
        self.messages = []

    def run(self, user_message: str, max_iterations: int = 5) -> str:
        """
        Send a message and get a response.

        This is the agent loop - it keeps calling the LLM and executing
        tools until the LLM has a final answer.
        """
        # Add user message
        self.messages.append({"role": "user", "content": user_message})

        # The agent loop
        for _ in range(max_iterations):
            # Call LLM with tools
            response = self.client.responses.create(
                model=self.model, input=self.messages, tools=TOOL_SCHEMAS
            )

            # Process response
            has_tool_calls = False
            final_text = None

            for item in response.output:
                if item.type == "message":
                    # LLM generated text (final answer)
                    if item.content and len(item.content) > 0:
                        final_text = item.content[0].text
                    self.messages.append(
                        {"role": "assistant", "content": final_text or ""}
                    )

                elif item.type == "function_call":
                    # LLM wants to call a tool
                    has_tool_calls = True

                    # Add function call to history
                    self.messages.append(
                        {
                            "type": "function_call",
                            "call_id": item.call_id,
                            "name": item.name,
                            "arguments": item.arguments,
                        }
                    )

                    # Execute tool
                    result = self._execute_tool(item)

                    # Add result to history
                    self.messages.append(
                        {
                            "type": "function_call_output",
                            "call_id": item.call_id,
                            "output": result,
                        }
                    )

            # If no tool calls, we're done
            if not has_tool_calls and final_text:
                return final_text

        return "Max iterations reached"

    def _execute_tool(self, function_call) -> str:
        """Execute a tool call and return result as JSON."""
        tool_name = function_call.name

        print(f"Executing tool: {tool_name}")

        if tool_name not in TOOLS:
            return json.dumps({"error": f"Tool '{tool_name}' not found"})

        try:
            args = json.loads(function_call.arguments)
            result = TOOLS[tool_name](**args)
            return json.dumps({"result": result})
        except Exception as e:
            return json.dumps({"error": f"{type(e).__name__}: {str(e)}"})

    def reset(self):
        """Clear conversation history."""
        self.messages = []


##=================================================##
## Example 1: Single Tool
##=================================================##

agent = Agent()
agent.run("What's the weather in Paris?")
# "The weather in Paris is sunny with a temperature of 22°C."


##=================================================##
## Example 2: Multiple Tools
##=================================================##

agent = Agent()
agent.run("What's the weather in Tokyo and what time is it there?")
# "The weather in Tokyo is rainy at 18°C, and the current time is 21:30."


##=================================================##
## Example 3: Conversation Memory
##=================================================##

agent = Agent()

agent.run("What's the weather in Paris?")
# "The weather in Paris is sunny with a temperature of 22°C."

agent.run("What about London?")
# "The weather in London is cloudy with a temperature of 15°C."

agent.run("Which city is warmer?")
# "Paris is warmer with a temperature of 22°C compared to London's 15°C."


##=================================================##
## Example 4: Multi-Step Reasoning
##=================================================##

agent = Agent()
agent.run("Check the weather in Paris, London, and Tokyo. Which is warmest?")
# The agent will call get_weather 3 times, then answer: "Paris is the warmest at 22°C"
