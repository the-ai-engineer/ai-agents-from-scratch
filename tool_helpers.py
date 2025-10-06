"""
Helper utilities for building AI agent tools with automatic schema generation.

This module provides:
1. Safe calculator without eval()
2. Decorator to auto-generate OpenAI tool schemas from Python functions
"""

import operator
import re
from typing import get_type_hints, Callable, Any
from pydantic import BaseModel, Field, create_model
from inspect import signature, getdoc


def safe_calculate(expression: str) -> float:
    """
    Safely evaluate mathematical expressions without using eval().

    Supports: +, -, *, /, **, (), decimals
    Does NOT support: variables, functions, imports, or arbitrary code

    Args:
        expression: Mathematical expression like "2+2" or "15.5 * 3.2"

    Returns:
        Result of the calculation

    Raises:
        ValueError: If expression is invalid or contains unsupported operations

    Examples:
        >>> safe_calculate("2 + 2")
        4.0
        >>> safe_calculate("157.09 * 493.89")
        77585.1801
    """
    # Remove whitespace
    expression = expression.replace(" ", "")

    # Validate that expression only contains allowed characters
    if not re.match(r'^[\d+\-*/().]+$', expression):
        raise ValueError("Expression contains invalid characters. Only digits and +, -, *, /, **, () are allowed.")

    # Define safe operators
    ops = {
        '+': operator.add,
        '-': operator.sub,
        '*': operator.mul,
        '/': operator.truediv,
        '**': operator.pow,
    }

    def parse_number(s: str) -> float:
        """Parse a number from string"""
        try:
            return float(s)
        except ValueError:
            raise ValueError(f"Invalid number: {s}")

    def find_matching_paren(expr: str, start: int) -> int:
        """Find the matching closing parenthesis"""
        count = 1
        for i in range(start + 1, len(expr)):
            if expr[i] == '(':
                count += 1
            elif expr[i] == ')':
                count -= 1
                if count == 0:
                    return i
        raise ValueError("Unmatched parentheses")

    def evaluate(expr: str) -> float:
        """Recursively evaluate expression"""
        # Handle parentheses first
        while '(' in expr:
            start = expr.rfind('(')
            end = find_matching_paren(expr, start)
            inner = expr[start+1:end]
            result = evaluate(inner)
            expr = expr[:start] + str(result) + expr[end+1:]

        # Handle exponentiation (right-to-left)
        if '**' in expr:
            parts = expr.split('**')
            result = parse_number(parts[-1])
            for part in reversed(parts[:-1]):
                result = ops['**'](parse_number(part), result)
            return result

        # Handle multiplication and division (left-to-right)
        for op in ['*', '/']:
            if op in expr:
                parts = re.split(r'([*/])', expr)
                result = parse_number(parts[0])
                for i in range(1, len(parts), 2):
                    operation = parts[i]
                    operand = parse_number(parts[i+1])
                    result = ops[operation](result, operand)
                return result

        # Handle addition and subtraction (left-to-right)
        # Need to be careful with negative numbers
        parts = re.split(r'([+-])', expr)
        if parts[0] == '':
            # Leading +/- sign
            parts = parts[1:]
            result = parse_number(parts[1]) if parts[0] == '+' else -parse_number(parts[1])
            parts = parts[2:]
        else:
            result = parse_number(parts[0])
            parts = parts[1:]

        for i in range(0, len(parts), 2):
            if i >= len(parts):
                break
            operation = parts[i]
            operand = parse_number(parts[i+1])
            result = ops[operation](result, operand)

        return result

    try:
        return evaluate(expression)
    except ZeroDivisionError:
        raise ValueError("Division by zero")
    except Exception as e:
        raise ValueError(f"Invalid expression: {str(e)}")


def tool_schema(func: Callable) -> dict:
    """
    Automatically generate OpenAI tool schema from a Python function.

    The function must have:
    - Type hints for all parameters
    - A docstring describing what it does
    - (Optional) Field descriptions in the docstring using "Args:" format

    Args:
        func: Python function to convert to tool schema

    Returns:
        OpenAI-compatible tool schema dictionary

    Example:
        >>> def get_weather(city: str, units: str = "celsius") -> str:
        ...     '''Get current weather for a city
        ...
        ...     Args:
        ...         city: City name, e.g., 'Paris' or 'London'
        ...         units: Temperature units ('celsius' or 'fahrenheit')
        ...     '''
        ...     pass
        >>> schema = tool_schema(get_weather)
        >>> print(schema['name'])
        'get_weather'
    """
    # Get function metadata
    sig = signature(func)
    type_hints = get_type_hints(func)
    doc = getdoc(func) or ""

    # Parse docstring for description and parameter descriptions
    lines = doc.split('\n')
    description = lines[0] if lines else func.__name__

    # Extract parameter descriptions from docstring
    param_descriptions = {}
    in_args_section = False
    for line in lines:
        line = line.strip()
        if line.startswith('Args:'):
            in_args_section = True
            continue
        if in_args_section:
            if line.startswith('Returns:') or line.startswith('Raises:') or line.startswith('Example:'):
                break
            if ':' in line:
                param_name = line.split(':')[0].strip()
                param_desc = ':'.join(line.split(':')[1:]).strip()
                param_descriptions[param_name] = param_desc

    # Build parameters schema
    properties = {}
    required = []

    for param_name, param in sig.parameters.items():
        if param_name == 'self':
            continue

        param_type = type_hints.get(param_name, str)

        # Map Python types to JSON Schema types
        type_mapping = {
            str: "string",
            int: "integer",
            float: "number",
            bool: "boolean",
            list: "array",
            dict: "object",
        }

        json_type = type_mapping.get(param_type, "string")

        properties[param_name] = {
            "type": json_type,
            "description": param_descriptions.get(param_name, f"The {param_name} parameter")
        }

        # Check if parameter is required (no default value)
        if param.default == param.empty:
            required.append(param_name)

    return {
        "type": "function",
        "function": {
            "name": func.__name__,
            "description": description,
            "parameters": {
                "type": "object",
                "properties": properties,
                "required": required
            }
        }
    }


class BaseTool:
    """
    Base class for agent tools with automatic schema generation.

    Usage:
        class CalculatorTool(BaseTool):
            name = "calculator"  # Optional: override default name

            def execute(self, expression: str) -> dict:
                '''Performs mathematical calculations

                Args:
                    expression: Math expression to evaluate (e.g., '2+2', '15*7.5')
                '''
                result = safe_calculate(expression)
                return {"result": result}

        tool = CalculatorTool()
        schema = tool.get_schema()
        result = tool.execute("2 + 2")
    """

    name: str = None  # Override in subclass to set custom name

    def get_schema(self) -> dict:
        """Generate OpenAI tool schema from the execute method"""
        schema = tool_schema(self.execute)

        # Use custom name if provided, otherwise derive from class name
        if self.name:
            schema['function']['name'] = self.name
        else:
            # Convert CalculatorTool -> calculator
            class_name = self.__class__.__name__
            if class_name.endswith('Tool'):
                class_name = class_name[:-4]
            schema['function']['name'] = class_name.lower()

        return schema

    def execute(self, **kwargs) -> dict:
        """Override this method in subclasses"""
        raise NotImplementedError("Subclasses must implement execute()")


# Example tools using the new pattern

class CalculatorTool(BaseTool):
    """A tool for performing safe mathematical calculations"""

    def execute(self, expression: str) -> dict:
        """Performs mathematical calculations

        Args:
            expression: Mathematical expression to evaluate (e.g., '2+2', '157.09*493.89')
        """
        try:
            result = safe_calculate(expression)
            return {"result": result}
        except ValueError as e:
            return {"error": str(e)}


class WeatherTool(BaseTool):
    """A tool for getting weather information"""

    def execute(self, city: str, units: str = "celsius") -> dict:
        """Get current weather for a city

        Args:
            city: City name, e.g., 'Paris', 'London', 'Tokyo'
            units: Temperature units ('celsius' or 'fahrenheit')
        """
        # Mock data
        weather_db = {
            "paris": {"celsius": "Sunny, 22°C", "fahrenheit": "Sunny, 72°F"},
            "london": {"celsius": "Cloudy, 15°C", "fahrenheit": "Cloudy, 59°F"},
            "tokyo": {"celsius": "Rainy, 18°C", "fahrenheit": "Rainy, 64°F"},
        }

        city_lower = city.lower()
        if city_lower in weather_db:
            return {"weather": weather_db[city_lower].get(units, weather_db[city_lower]["celsius"])}
        else:
            return {"error": f"Weather data not available for {city}"}


class Tool(BaseTool):
    """
    Universal tool wrapper that automatically generates schemas from any Python function.

    Just pass your function and everything else is auto-detected from:
    - Type hints (for parameter types)
    - Docstring (for descriptions)
    - Function name (for tool name)

    Args:
        function: The Python function to wrap
        name: Optional override for tool name (default: function.__name__)
        description: Optional override for tool description (default: from docstring)

    Example - Simplest form (everything auto-detected):
        >>> def get_weather(city: str, units: str = "celsius") -> str:
        ...     '''Get current weather for a city
        ...
        ...     Args:
        ...         city: City name (e.g., 'Paris', 'London')
        ...         units: Temperature units ('celsius' or 'fahrenheit')
        ...     '''
        ...     return f"Weather in {city}: 22°{units[0].upper()}"

        >>> tool = Tool(get_weather)
        >>> schema = tool.get_schema()
        >>> result = tool.execute(city="Paris")

    Example - With optional overrides:
        >>> tool = Tool(
        ...     function=get_weather,
        ...     name="weather",  # override default name
        ...     description="Get real-time weather data"  # override docstring
        ... )
    """

    def __init__(
        self,
        function: Callable,
        name: str = None,
        description: str = None
    ):
        self.function = function
        self._name_override = name
        self._description_override = description

    def get_schema(self) -> dict:
        """Generate OpenAI tool schema from the wrapped function"""
        schema = tool_schema(self.function)

        # Apply overrides if provided
        if self._name_override:
            schema['function']['name'] = self._name_override

        if self._description_override:
            schema['function']['description'] = self._description_override

        return schema

    def execute(self, **kwargs) -> dict:
        """Execute the wrapped function"""
        result = self.function(**kwargs)

        # Ensure result is always a dict
        if isinstance(result, dict):
            return result
        return {"result": result}


# Simple function decorator approach
def as_tool(func: Callable) -> Tool:
    """
    Decorator to convert a function into a tool.

    Args:
        func: Function to convert to a tool

    Returns:
        Tool instance

    Example:
        >>> @as_tool
        ... def search_web(query: str) -> dict:
        ...     '''Search the web for information
        ...
        ...     Args:
        ...         query: Search query string
        ...     '''
        ...     return {"results": f"Results for: {query}"}

        >>> schema = search_web.get_schema()
        >>> result = search_web.execute(query="Python tutorials")
    """
    return Tool(func)


if __name__ == "__main__":
    # Test safe calculator
    print("="*60)
    print("Testing safe_calculate:")
    print("="*60)
    print(f"2 + 2 = {safe_calculate('2 + 2')}")
    print(f"157.09 * 493.89 = {safe_calculate('157.09 * 493.89')}")
    print(f"(10 + 5) * 2 = {safe_calculate('(10 + 5) * 2')}")
    print()

    # Test the new simplified Tool class
    print("="*60)
    print("Testing Tool class (simplest form):")
    print("="*60)

    def get_weather(city: str, units: str = "celsius") -> str:
        """Get current weather for a city

        Args:
            city: City name (e.g., 'Paris', 'London')
            units: Temperature units ('celsius' or 'fahrenheit')
        """
        weather_db = {
            "paris": {"celsius": "22°C", "fahrenheit": "72°F"},
            "london": {"celsius": "15°C", "fahrenheit": "59°F"},
        }
        temp = weather_db.get(city.lower(), {}).get(units, "Unknown")
        return f"Weather in {city}: {temp}"

    # Simplest usage - just pass the function!
    weather_tool = Tool(get_weather)

    print("Auto-generated schema:")
    schema = weather_tool.get_schema()
    print(f"  Name: {schema['function']['name']}")
    print(f"  Description: {schema['function']['description']}")
    print(f"  Parameters: {list(schema['function']['parameters']['properties'].keys())}")
    print(f"  Required: {schema['function']['parameters']['required']}")
    print()

    print("Executing tool:")
    result = weather_tool.execute(city="Paris")
    print(f"  Result: {result}")
    print()

    # Test with overrides
    print("="*60)
    print("Testing Tool with overrides:")
    print("="*60)
    weather_tool_custom = Tool(
        function=get_weather,
        name="weather",
        description="Get real-time weather data"
    )
    schema = weather_tool_custom.get_schema()
    print(f"  Name (overridden): {schema['function']['name']}")
    print(f"  Description (overridden): {schema['function']['description']}")
    print()

    # Test @as_tool decorator
    print("="*60)
    print("Testing @as_tool decorator:")
    print("="*60)

    @as_tool
    def search_web(query: str, max_results: int = 10) -> dict:
        """Search the web for information

        Args:
            query: Search query string
            max_results: Maximum number of results to return
        """
        return {"results": f"Found results for: {query} (max: {max_results})"}

    schema = search_web.get_schema()
    print(f"  Name: {schema['function']['name']}")
    print(f"  Description: {schema['function']['description']}")
    result = search_web.execute(query="Python tutorials", max_results=5)
    print(f"  Result: {result}")
    print()

    # Test legacy BaseTool approach still works
    print("="*60)
    print("Testing legacy BaseTool (for custom tools):")
    print("="*60)
    calc_tool = CalculatorTool()
    schema = calc_tool.get_schema()
    print(f"  Name: {schema['function']['name']}")
    result = calc_tool.execute("2 + 2")
    print(f"  Result: {result}")
