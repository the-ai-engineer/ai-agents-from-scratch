"""
Tool module for creating and managing agent tools.

This module provides a simple abstraction for converting Python functions
into OpenAI-compatible tool schemas. It's designed to make tool creation
as simple as writing a regular function with a docstring.

Usage:
    # Method 1: Direct instantiation
    tool_obj = Tool.from_function(my_function)

    # Method 2: Decorator (convenience)
    @tool
    def my_function(param: str) -> str:
        '''Function description.'''
        return "result"
"""

from dataclasses import dataclass
from typing import Callable
import inspect


@dataclass
class Tool:
    """
    Represents an agent tool with name, description, and parameter schema.

    Tools are functions that agents can call to interact with external systems,
    retrieve information, or perform computations.
    """

    name: str
    description: str
    parameters: dict

    def to_openai_format(self) -> dict:
        """Convert tool to OpenAI function calling format."""
        return {
            "type": "function",
            "name": self.name,
            "description": self.description,
            "parameters": self.parameters,
        }

    @classmethod
    def from_function(cls, func: Callable) -> "Tool":
        """
        Create a Tool from a Python function.

        Automatically extracts the function name, docstring, and parameters
        to create a tool schema. This makes tool creation as simple as
        writing a function with a good docstring.

        Args:
            func: A Python function to convert to a tool

        Returns:
            Tool instance representing the function

        Example:
            def get_weather(city: str, units: str = "celsius") -> str:
                '''Get weather for a city in specified units.'''
                return f"Weather in {city}: 72°F"

            tool = Tool.from_function(get_weather)

        Note:
            Currently all parameters are treated as strings. For production use,
            you may want to infer types from annotations.
        """
        sig = inspect.signature(func)

        properties = {}
        required = []

        for name, param in sig.parameters.items():
            properties[name] = {"type": "string"}
            if param.default == inspect.Parameter.empty:
                required.append(name)

        return cls(
            name=func.__name__,
            description=inspect.getdoc(func) or "",
            parameters={
                "type": "object",
                "properties": properties,
                "required": required,
            },
        )

    def __repr__(self) -> str:
        """String representation of the tool."""
        param_names = list(self.parameters.get("properties", {}).keys())
        return f"Tool(name={self.name}, params={param_names})"


def tool(func: Callable) -> Callable:
    """
    Decorator to mark a function as a tool.

    This decorator automatically generates a Tool instance from the function
    and attaches it as the 'tool' attribute. This allows for clean syntax:

    Example:
        @tool
        def get_weather(city: str) -> str:
            '''Get weather for a city.'''
            return f"Weather in {city}"

        # Access the tool schema
        schema = get_weather.tool.to_openai_format()

        # Call the function normally
        result = get_weather("Paris")

    Args:
        func: The function to decorate

    Returns:
        The original function with a 'tool' attribute attached
    """
    func.tool = Tool.from_function(func)
    return func
