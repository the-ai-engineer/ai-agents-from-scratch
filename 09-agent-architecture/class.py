"""
Lesson 06: Agent Architecture - The Agent Class

Learn how to abstract the agent loop into a reusable, production-ready class.

This lesson teaches how to BUILD the Agent class from scratch using:
- ConversationMemory for managing conversation history
- @tool decorator for automatic schema generation
- Helper methods for cleaner code organization

Run from project root:
    uv run python 09-agent-architecture/class.py
"""

import json
import inspect
from typing import Callable, Optional, Any, get_type_hints
from openai import OpenAI
from pydantic import BaseModel, create_model
from dotenv import load_dotenv

load_dotenv()


##=================================================##
## ConversationMemory - Multi-Turn Dialogue Management
##=================================================##

class ConversationMemory:
    """
    Manages conversation history for multi-turn conversations.

    This is the foundation of agent memory - it keeps track of all messages
    including user inputs, assistant responses, and tool call results.
    """

    def __init__(self, system_prompt: str = None):
        self.items = []
        if system_prompt:
            self.add_system_message(system_prompt)

    def add_system_message(self, content: str):
        """Add a system message (instructions for the LLM)."""
        self.items.append({"role": "system", "content": content})

    def add_user_message(self, content: str):
        """Add a user message to the conversation."""
        self.items.append({"role": "user", "content": content})

    def add_response_output(self, output: list):
        """Add the entire output array from a response."""
        self.items.extend(output)

    def add_function_output(self, call_id: str, result: str):
        """Add a function call result to the conversation."""
        self.items.append(
            {"type": "function_call_output", "call_id": call_id, "output": result}
        )

    def get_items(self) -> list[dict]:
        """Get the full conversation history as items."""
        return self.items

    def clear(self):
        """Clear all items except system messages."""
        self.items = [item for item in self.items if item.get("role") == "system"]

    def __len__(self) -> int:
        return len(self.items)


##=================================================##
## Tool - Automatic Schema Generation
##=================================================##

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
    """
    Decorator to mark a function as a tool.

    Example:
        @tool
        def get_weather(city: str) -> str:
            '''Get weather for a city'''
            return f"Weather in {city}: 22°C"
    """
    func.tool = Tool.from_function(func)
    return func


##=================================================##
## Agent - Reusable Agent with Tool Calling
##=================================================##

class Agent:
    """
    An AI agent that can use tools to accomplish tasks.

    The agent loop:
    1. Send message to LLM with available tools
    2. If LLM wants to use a tool, execute it
    3. Send tool results back to LLM
    4. Repeat until LLM gives final answer (or max iterations)
    """

    def __init__(
        self,
        model: str = "gpt-4o-mini",
        max_iterations: int = 5,
        system_prompt: Optional[str] = None,
        tools: Optional[list[Callable]] = None,
    ):
        """
        Initialize the agent.

        Args:
            model: Which OpenAI model to use
            max_iterations: Maximum number of tool-calling loops
            system_prompt: System instructions for the agent
            tools: List of functions decorated with @tool
        """
        self.client = OpenAI()
        self.model = model
        self.max_iterations = max_iterations
        self.memory = ConversationMemory(system_prompt=system_prompt)
        self.tools: dict[str, Callable] = {}

        if tools:
            self.register_tools(*tools)

    def register_tool(self, func: Callable) -> None:
        """Register a single tool function."""
        if not hasattr(func, "tool"):
            raise ValueError(f"Function {func.__name__} must be decorated with @tool")
        self.tools[func.__name__] = func

    def register_tools(self, *funcs: Callable) -> None:
        """Register multiple tools at once."""
        for func in funcs:
            self.register_tool(func)

    @property
    def tool_schemas(self) -> Optional[list[dict]]:
        """Get OpenAI-formatted tool schemas."""
        if not self.tools:
            return None
        return [func.tool.to_openai_format() for func in self.tools.values()]

    def _execute_tool(self, function_call) -> str:
        """Execute a tool call and return the result as a string."""
        tool_name = function_call.name

        if tool_name not in self.tools:
            return json.dumps({"error": f"Tool '{tool_name}' not found"})

        try:
            args = json.loads(function_call.arguments)
            result = self.tools[tool_name](**args)
            return json.dumps({"result": result})
        except Exception as e:
            return json.dumps({"error": str(e)})

    def _has_function_calls(self, output: list) -> bool:
        """Check if output contains any function calls."""
        return any(item.type == "function_call" for item in output)

    def _extract_text_response(self, output: list) -> str:
        """Extract text content from output items."""
        for item in output:
            if item.type == "message":
                for content in item.content:
                    if content.type == "output_text":
                        return content.text
        return "No response generated"

    def chat(self, message: str) -> str:
        """
        Send a message and get a response.

        This is the main agent loop that handles:
        - Sending messages to the LLM
        - Detecting when the LLM wants to use tools
        - Executing tools and sending results back
        - Returning the final answer
        """
        self.memory.add_user_message(message)

        for iteration in range(self.max_iterations):
            try:
                response = self.client.responses.create(
                    model=self.model,
                    input=self.memory.get_items(),
                    tools=self.tool_schemas,
                )
            except Exception as e:
                return f"Error calling API: {str(e)}"

            # Check if LLM wants to use tools
            if not self._has_function_calls(response.output):
                # No tools needed - we have our final answer
                answer = self._extract_text_response(response.output)
                self.memory.add_response_output(response.output)
                return answer

            # LLM wants to use tools - add output to memory first
            self.memory.add_response_output(response.output)

            # Execute all function calls
            for item in response.output:
                if item.type == "function_call":
                    result = self._execute_tool(item)
                    self.memory.add_function_output(item.call_id, result)

        return "Max iterations reached without final answer"

    def reset(self) -> None:
        """Clear conversation history."""
        self.memory.clear()


##=================================================##
## Example Tools
##=================================================##

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
        expression: Math expression to evaluate (e.g., '2+2', '15*24')
    """
    try:
        result = eval(expression, {"__builtins__": {}}, {})
        return str(result)
    except Exception as e:
        return f"Error: {str(e)}"


##=================================================##
## Example 1: Basic agent usage with tools
##=================================================##

# Create agent with tools passed in constructor
agent = Agent(
    model="gpt-4o-mini",
    max_iterations=5,
    system_prompt="You are a helpful assistant with access to tools.",
    tools=[get_weather, calculate]
)

# Use the agent
agent.chat("What's the weather in Paris?")
# agent.chat("What is 15 * 24?")
# agent.chat("What's the weather in London and what is 100 + 50?")


##=================================================##
## Example 2: Agent maintains conversation context
##=================================================##

# Create agent and register tools
agent = Agent(
    system_prompt="You are a helpful weather assistant.",
    tools=[get_weather]
)

# Multi-turn conversation
agent.chat("What's the weather in Paris?")
agent.chat("What about London?")
agent.chat("Which one is warmer?")

# len(agent.memory)  # Shows memory contains multiple items
