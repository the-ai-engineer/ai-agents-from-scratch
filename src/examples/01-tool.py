"""
Example 01: Creating Tools from Functions

This example demonstrates how to convert Python functions into
OpenAI-compatible tool schemas using the Tool class.

Run from project root:
    uv run python src/examples/01-tool.py
"""

import sys
from pathlib import Path

# Add project root to path so we can import from src
project_root = Path(__file__).parent.parent.parent
sys.path.insert(0, str(project_root))

from src.tool import Tool


def get_weather(city: str) -> str:
    """Get the current weather for a city."""
    # In a real application, this would call a weather API
    weather_data = {
        "Tokyo": "Sunny, 72F",
        "London": "Rainy, 55F",
        "New York": "Cloudy, 68F",
        "Paris": "Sunny, 70F",
    }
    return weather_data.get(city, f"Weather data not available for {city}")


weather_tool = Tool.from_function(get_weather)
print(weather_tool.to_dict())
