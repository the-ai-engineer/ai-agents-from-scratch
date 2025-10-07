"""
Minimal AI Agent Framework - Using OpenAI Responses API
A step-by-step teaching framework for building AI agents.

Step 1: Basic API calls
Step 2: Structured output
Step 3: Memory (The foundation of agents)
Step 4: Tool calling
Step 5: Agent with tool-calling loop
"""

import os
import json
import inspect
from typing import Callable, Optional, Any, get_type_hints
from openai import OpenAI
from pydantic import BaseModel, create_model


# ============================================================================
# STEP 1: Basic API Calling
# ============================================================================


def simple_chat(
    message: str, model: str = "gpt-4o-mini", temperature: float = 1.0
) -> str:
    """
    The simplest way to call an LLM.
    Send a message, get a response back.

    Args:
        message: The user's message
        model: Which model to use
        temperature: Controls randomness (0.0 = deterministic, 2.0 = very random)
    """
    client = OpenAI(api_key=os.getenv("OPENAI_API_KEY"))

    response = client.responses.create(
        model=model, input=message, temperature=temperature
    )

    return response.output_text


# ============================================================================
# STEP 2: Structured Output
# ============================================================================


def structured_chat(
    message: str, response_format: type[BaseModel], model: str = "gpt-4o-mini"
) -> BaseModel:
    """
    Get structured output from an LLM using Pydantic models.
    Instead of free text, get a validated Pydantic model.

    The .parse() method automatically handles:
    - Converting your Pydantic model to a JSON schema
    - Instructing the LLM to follow that schema
    - Parsing and validating the response

    Args:
        message: The user's message
        response_format: A Pydantic model class defining the expected structure
        model: Which model to use

    Returns:
        An instance of the Pydantic model with validated data
    """
    client = OpenAI(api_key=os.getenv("OPENAI_API_KEY"))

    response = client.responses.parse(
        model=model,
        input=message,
        text_format=response_format,
    )

    return response.output_parsed


# ============================================================================
# STEP 3: Tool Calling
# ============================================================================


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
    """Decorator to mark a function as a tool."""
    func.tool = Tool.from_function(func)
    return func


# ============================================================================
# STEP 4: Memory Management (The Foundation of Agents)
# ============================================================================

# Memory is what transforms a single LLM call into an agent.
# Without memory, each interaction is isolated.
# With memory, the LLM can:
#   - Remember previous exchanges
#   - Maintain context across multiple turns
#   - Build on previous tool calls and results
#
# This is the key insight: Agents = LLMs + Tools + Memory


class ConversationMemory:
    """Manages conversation history for multi-turn conversations."""

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


# ============================================================================
# STEP 5: The Agent (Putting It All Together)
# ============================================================================


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
        self.client = OpenAI(api_key=os.getenv("OPENAI_API_KEY"))
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
                # No tools needed - we have our final answer!
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


# ============================================================================
# EXAMPLES
# ============================================================================

if __name__ == "__main__":
    print("=" * 70)
    print("STEP 1: Simple Chat")
    print("=" * 70)
    response = simple_chat("What is 2+2?", temperature=0.7)
    print(f"Response: {response}\n")

    print("=" * 70)
    print("STEP 2: Structured Output")
    print("=" * 70)

    class MathSolution(BaseModel):
        steps: list[str]
        answer: int

    solution = structured_chat(
        "Solve 15 * 24 step by step", response_format=MathSolution
    )
    print(f"Steps: {solution.steps}")
    print(f"Answer: {solution.answer}\n")

    print("=" * 70)
    print("STEP 3: Memory - Multi-Turn Conversations")
    print("=" * 70)
    print("Without memory, each LLM call is isolated.")
    print("With memory, we can have conversations!\n")

    client = OpenAI(api_key=os.getenv("OPENAI_API_KEY"))
    memory = ConversationMemory(system_prompt="You are a helpful math tutor.")

    # Turn 1
    memory.add_user_message("My favorite number is 7")
    response = client.responses.create(model="gpt-4o-mini", input=memory.get_items())
    memory.add_response_output(response.output)
    print("User: My favorite number is 7")
    print(f"Assistant: {response.output_text}\n")

    # Turn 2 - The LLM remembers!
    memory.add_user_message("What's my favorite number times 3?")
    response = client.responses.create(model="gpt-4o-mini", input=memory.get_items())
    memory.add_response_output(response.output)
    print("User: What's my favorite number times 3?")
    print(f"Assistant: {response.output_text}")
    print("\n✓ The LLM remembered your favorite number across turns!")
    print("✓ This is the foundation of the 'Augmented LLM' pattern\n")

    print("=" * 70)
    print("STEP 4: Tool Calling with Agent")
    print("=" * 70)

    @tool
    def get_weather(location: str, units: str = "celsius") -> str:
        """Get the current weather in a given location.

        Args:
            location: The city and state, e.g. San Francisco, CA
            units: Temperature units (celsius or fahrenheit)
        """
        return f"The weather in {location} is 72°{units[0].upper()} and sunny"

    @tool
    def calculate(expression: str) -> str:
        """Evaluate a mathematical expression.

        Args:
            expression: A mathematical expression like "2 + 2"
        """
        try:
            return str(eval(expression))
        except Exception as e:
            return f"Error: {str(e)}"

    agent = Agent(
        system_prompt="You are a helpful assistant with access to tools.",
        tools=[get_weather, calculate],
    )

    print("\nQuery 1: What's the weather in Paris?")
    response = agent.chat("What's the weather in Paris?")
    print(f"Response: {response}")

    print("\nQuery 2: What's 15 * 24?")
    response = agent.chat("What's 15 * 24?")
    print(f"Response: {response}")

    print("\nQuery 3: Multi-step reasoning")
    response = agent.chat(
        "If it's 72°F in Paris, what is that in Celsius? Use the calculator."
    )
    print(f"Response: {response}")
