"""
Example demonstrating the core agent and tool primitives.

This example shows how to:
1. Create tools from Python functions
2. Build an agent with tool-calling capabilities
3. Enable logging to see what's happening under the hood
4. Have a conversation with an agent

Run this example from project root:
    uv run python src/examples/example.py
"""

import logging
import os
import sys
from pathlib import Path
from dotenv import load_dotenv

# Add project root to path so we can import from src
project_root = Path(__file__).parent.parent.parent
sys.path.insert(0, str(project_root))

from src import Agent, Tool

# Load environment variables
load_dotenv()

# Configure logging to see what's happening
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    datefmt='%H:%M:%S'
)

# To see DEBUG logs (more detailed), uncomment this:
# logging.getLogger('src').setLevel(logging.DEBUG)


def main():
    """Run the example demonstrating agent capabilities."""

    print("=" * 70)
    print("AI Agent Example - Core Primitives Demo")
    print("=" * 70)
    print()

    # ========================================================================
    # Example 1: Creating Tools
    # ========================================================================
    print("📦 Example 1: Creating Tools from Python Functions")
    print("-" * 70)

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

    def calculate(operation: str, a: str, b: str) -> str:
        """Perform a mathematical operation on two numbers."""
        x, y = float(a), float(b)
        if operation == "add":
            return str(x + y)
        elif operation == "subtract":
            return str(x - y)
        elif operation == "multiply":
            return str(x * y)
        elif operation == "divide":
            return str(x / y) if y != 0 else "Error: Division by zero"
        else:
            return f"Unknown operation: {operation}"

    # Convert functions to tools
    weather_tool = Tool.from_function(get_weather)
    calc_tool = Tool.from_function(calculate)

    print(f"Created tool: {weather_tool}")
    print(f"Created tool: {calc_tool}")
    print()

    # ========================================================================
    # Example 2: Simple Agent with No Tools
    # ========================================================================
    print("🤖 Example 2: Simple Agent (No Tools)")
    print("-" * 70)

    simple_agent = Agent(
        system_prompt="You are a helpful assistant. Be concise."
    )

    response = simple_agent.chat("What is the capital of France?")
    print(f"User: What is the capital of France?")
    print(f"Agent: {response}")
    print()

    # ========================================================================
    # Example 3: Agent with Weather Tool
    # ========================================================================
    print("🌤️  Example 3: Agent with Weather Tool")
    print("-" * 70)

    weather_agent = Agent(
        system_prompt="You are a helpful weather assistant.",
        max_iterations=5
    )
    weather_agent.register_tool(get_weather)

    response = weather_agent.chat("What's the weather in Tokyo?")
    print(f"User: What's the weather in Tokyo?")
    print(f"Agent: {response}")
    print()

    # ========================================================================
    # Example 4: Agent with Multiple Tools
    # ========================================================================
    print("🔧 Example 4: Agent with Multiple Tools")
    print("-" * 70)

    multi_tool_agent = Agent(
        system_prompt="You are a helpful assistant with access to weather and calculator tools.",
        max_iterations=5
    )
    multi_tool_agent.register_tools(get_weather, calculate)

    # Test 1: Calculator
    response = multi_tool_agent.chat("What is 156 multiplied by 23?")
    print(f"User: What is 156 multiplied by 23?")
    print(f"Agent: {response}")
    print()

    # Test 2: Weather
    response = multi_tool_agent.chat("How's the weather in London?")
    print(f"User: How's the weather in London?")
    print(f"Agent: {response}")
    print()

    # ========================================================================
    # Example 5: Multi-Step Reasoning
    # ========================================================================
    print("🧠 Example 5: Multi-Step Reasoning")
    print("-" * 70)

    reasoning_agent = Agent(
        system_prompt="You are a helpful assistant. Use tools when needed.",
        max_iterations=5
    )
    reasoning_agent.register_tools(get_weather, calculate)

    response = reasoning_agent.chat(
        "What's the weather in Tokyo and Paris? "
        "Then tell me which city is warmer."
    )
    print(f"User: What's the weather in Tokyo and Paris? Then tell me which city is warmer.")
    print(f"Agent: {response}")
    print()

    # ========================================================================
    # Example 6: Conversation Memory
    # ========================================================================
    print("💭 Example 6: Conversation Memory")
    print("-" * 70)

    memory_agent = Agent(
        system_prompt="You are a helpful assistant. Remember the conversation context."
    )

    response1 = memory_agent.chat("My favorite city is Paris.")
    print(f"User: My favorite city is Paris.")
    print(f"Agent: {response1}")
    print()

    response2 = memory_agent.chat("What's my favorite city?")
    print(f"User: What's my favorite city?")
    print(f"Agent: {response2}")
    print()

    # Reset and try again
    memory_agent.reset()
    response3 = memory_agent.chat("What's my favorite city?")
    print(f"[After reset]")
    print(f"User: What's my favorite city?")
    print(f"Agent: {response3}")
    print()

    # ========================================================================
    # Summary
    # ========================================================================
    print("=" * 70)
    print("✨ Summary")
    print("=" * 70)
    print()
    print("You've learned:")
    print("  ✓ How to create tools from Python functions")
    print("  ✓ How to build agents with tool-calling capabilities")
    print("  ✓ How agents use tools in a loop to solve problems")
    print("  ✓ How conversation memory works")
    print("  ✓ How multi-step reasoning works")
    print()
    print("Next steps:")
    print("  • Check the tests/ folder to see how components are tested")
    print("  • Read the source code in src/ to understand internals")
    print("  • Build your own agent with custom tools!")
    print()


if __name__ == "__main__":
    # Check if API key is set
    if not os.getenv("OPENAI_API_KEY"):
        print("Error: OPENAI_API_KEY not found in environment")
        print("Create a .env file with: OPENAI_API_KEY=your-key-here")
        exit(1)

    main()
