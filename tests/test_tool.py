"""
Simplified tests for the Tool class.

These tests focus on the core behaviors students need to understand:
1. How to create tools from functions
2. How required/optional parameters work
3. How tools convert to OpenAI format
"""

from src.tool import Tool


def test_tool_creation_basics():
    """Test creating tools from functions with various parameter types."""

    # Simple function with one required parameter
    def get_weather(city: str) -> str:
        """Get the current weather for a city."""
        return f"Weather in {city}: Sunny"

    # Function with mix of required and optional parameters
    def calculate(operation: str, a: str, b: str = "0") -> str:
        """Perform a mathematical operation."""
        return "result"

    # Create tools
    weather_tool = Tool.from_function(get_weather)
    calc_tool = Tool.from_function(calculate)

    # Verify basic properties
    assert weather_tool.name == "get_weather"
    assert weather_tool.description == "Get the current weather for a city."
    assert "city" in weather_tool.parameters["required"]

    # Verify required vs optional params
    assert "operation" in calc_tool.parameters["required"]
    assert "a" in calc_tool.parameters["required"]
    assert "b" not in calc_tool.parameters["required"]  # Has default
    assert "b" in calc_tool.parameters["properties"]  # Still in schema


def test_tool_to_openai_format():
    """Test that tools convert to correct OpenAI API format."""

    def search(query: str) -> str:
        """Search for information."""
        return "results"

    tool = Tool.from_function(search)
    schema = tool.to_dict()

    # Verify OpenAI structure
    assert schema["type"] == "function"
    assert schema["function"]["name"] == "search"
    assert schema["function"]["description"] == "Search for information."
    assert schema["function"]["parameters"]["type"] == "object"
    assert "query" in schema["function"]["parameters"]["properties"]
    assert "query" in schema["function"]["parameters"]["required"]


def test_tool_edge_cases():
    """Test tools handle edge cases gracefully."""

    # Function with no parameters
    def get_time():
        """Get current time."""
        return "12:00"

    # Function without docstring
    def no_docs(x: str):
        return x

    time_tool = Tool.from_function(get_time)
    assert time_tool.parameters["required"] == []
    assert time_tool.parameters["properties"] == {}

    no_doc_tool = Tool.from_function(no_docs)
    assert no_doc_tool.description == ""
