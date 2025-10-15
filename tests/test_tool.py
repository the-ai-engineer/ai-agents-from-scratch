"""
Tests for the Tool class.

These tests focus on core behaviors:
1. Creating tools from functions
2. Required/optional parameters
3. Converting to OpenAI format
4. The @tool decorator
"""

from src.tool import Tool, tool


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

    tool_obj = Tool.from_function(search)
    schema = tool_obj.to_openai_format()

    # Verify OpenAI Responses API format (flat structure)
    assert schema["type"] == "function"
    assert schema["name"] == "search"
    assert schema["description"] == "Search for information."
    assert schema["parameters"]["type"] == "object"
    assert "query" in schema["parameters"]["properties"]
    assert "query" in schema["parameters"]["required"]


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


def test_tool_decorator():
    """Test the @tool decorator works correctly."""

    @tool
    def get_weather(city: str) -> str:
        """Get weather for a city."""
        return f"Weather in {city}"

    # Function still callable
    result = get_weather("Paris")
    assert "Paris" in result

    # Tool attribute exists
    assert hasattr(get_weather, 'tool')
    assert isinstance(get_weather.tool, Tool)

    # Tool has correct properties
    assert get_weather.tool.name == "get_weather"
    assert get_weather.tool.description == "Get weather for a city."

    # Can convert to OpenAI format
    schema = get_weather.tool.to_openai_format()
    assert schema["type"] == "function"
    assert schema["name"] == "get_weather"


def test_tool_repr():
    """Test tool string representation."""

    def search(query: str, limit: str = "10") -> str:
        """Search."""
        return "results"

    tool_obj = Tool.from_function(search)
    repr_str = repr(tool_obj)

    assert "Tool" in repr_str
    assert "search" in repr_str
    assert "query" in repr_str
    assert "limit" in repr_str


def test_multiple_parameters():
    """Test tool with multiple parameters."""

    def send_email(to: str, subject: str, body: str = "") -> str:
        """Send an email."""
        return "sent"

    tool_obj = Tool.from_function(send_email)

    # Required params
    assert "to" in tool_obj.parameters["required"]
    assert "subject" in tool_obj.parameters["required"]

    # Optional params
    assert "body" not in tool_obj.parameters["required"]

    # All params in properties
    props = tool_obj.parameters["properties"]
    assert "to" in props
    assert "subject" in props
    assert "body" in props

    # All currently treated as strings
    assert props["to"]["type"] == "string"
    assert props["subject"]["type"] == "string"
    assert props["body"]["type"] == "string"
