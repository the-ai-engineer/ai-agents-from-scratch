"""
Integration tests using real OpenAI API calls.

These tests verify the agent works end-to-end with the real API.
They are skipped unless OPENAI_API_KEY is set and pytest runs with -m integration.

Run with:
    pytest -m integration              # Run only integration tests
    pytest tests/                      # Run all tests (unit + integration)
    pytest -m "not integration"        # Skip integration tests (default for CI)

Cost: ~$0.01-0.02 per full test run with gpt-4o-mini
"""

import pytest
import os
from dotenv import load_dotenv
from src.agent_sync import AgentSync
from src.agent import Agent
from pydantic import BaseModel

# Load environment variables from .env file
load_dotenv()


# Mark all tests in this file as integration tests
pytestmark = pytest.mark.integration


# Skip all tests if no API key is set
skip_if_no_key = pytest.mark.skipif(
    not os.getenv("OPENAI_API_KEY"),
    reason="OPENAI_API_KEY not set. Set it to run integration tests."
)


@skip_if_no_key
class TestRealAPIBasics:
    """Test basic agent functionality with real API."""

    def test_simple_chat_sync(self):
        """Test agent can chat without tools."""
        agent = AgentSync(system_prompt="Be very concise. Answer in one word when possible.")
        response = agent.run("What is 2+2? Answer with just the number.")

        # Loose assertions - we can't predict exact LLM output
        assert len(response) > 0
        assert "4" in response

    def test_conversation_memory(self):
        """Test agent remembers conversation history."""
        agent = AgentSync(system_prompt="Be concise.")

        # First message
        response1 = agent.run("My favorite color is blue.")
        assert len(response1) > 0

        # Second message - should remember
        response2 = agent.run("What's my favorite color?")
        assert "blue" in response2.lower()

    def test_system_prompt_followed(self):
        """Test agent follows system prompt."""
        agent = AgentSync(system_prompt="You are a pirate. Always say 'arr' in your response.")
        response = agent.run("Hello!")

        assert "arr" in response.lower()

    def test_reset_clears_memory(self):
        """Test reset clears conversation history."""
        agent = AgentSync(system_prompt="Be concise.")

        agent.run("My name is Alice.")
        agent.reset()
        response = agent.run("What's my name?")

        # After reset, agent shouldn't know the name
        assert "alice" not in response.lower() or "don't know" in response.lower()


@skip_if_no_key
class TestRealAPIToolCalling:
    """Test tool calling with real API."""

    def test_single_tool_call(self):
        """Test agent can call a tool and use the result."""
        def get_temperature() -> str:
            """Get the current temperature."""
            return "72 degrees Fahrenheit"

        agent = AgentSync(system_prompt="Be concise.")
        agent.add_tool(get_temperature)

        response = agent.run("What's the temperature?")

        assert "72" in response

    def test_multiple_tool_calls(self):
        """Test agent can call multiple tools in sequence."""
        def get_weather() -> str:
            """Get the current weather."""
            return "sunny"

        def get_temperature() -> str:
            """Get the current temperature."""
            return "72 degrees"

        agent = AgentSync(system_prompt="Be concise.")
        agent.add_tools(get_weather, get_temperature)

        response = agent.run("What's the weather and temperature?")

        assert "sunny" in response.lower()
        assert "72" in response

    def test_tool_with_parameters(self):
        """Test agent can call tool with parameters."""
        def calculate(operation: str, a: str, b: str) -> str:
            """Perform a mathematical operation.

            Args:
                operation: The operation (add, subtract, multiply, divide)
                a: First number
                b: Second number
            """
            a_num, b_num = float(a), float(b)

            if operation == "add":
                return str(a_num + b_num)
            elif operation == "multiply":
                return str(a_num * b_num)
            else:
                return "Unknown operation"

        agent = AgentSync(system_prompt="Be concise. When you get a result, just state it.")
        agent.add_tool(calculate)

        response = agent.run("What is 5 multiplied by 7?")

        assert "35" in response


@skip_if_no_key
class TestRealAPIStructuredOutput:
    """Test structured output with real API."""

    def test_structured_output_basic(self):
        """Test agent can return structured Pydantic model."""
        class Person(BaseModel):
            name: str
            age: int
            city: str

        agent = AgentSync()
        result = agent.run(
            "Extract person info: John is 30 years old and lives in NYC",
            response_format=Person
        )

        assert isinstance(result, Person)
        assert result.name.lower() == "john"
        assert result.age == 30
        assert "nyc" in result.city.lower() or "new york" in result.city.lower()

    def test_structured_output_list(self):
        """Test agent can return complex nested structure."""
        class Task(BaseModel):
            title: str
            priority: str

        class TaskList(BaseModel):
            tasks: list[Task]

        agent = AgentSync()
        result = agent.run(
            "Extract tasks: 1. Fix bug (high priority) 2. Write docs (low priority)",
            response_format=TaskList
        )

        assert isinstance(result, TaskList)
        assert len(result.tasks) == 2
        assert result.tasks[0].priority == "high"
        assert result.tasks[1].priority == "low"


@skip_if_no_key
class TestRealAPIAsync:
    """Test async agent with real API."""

    @pytest.mark.asyncio
    async def test_async_agent_basic(self):
        """Test async agent works with real API."""
        agent = Agent(system_prompt="Be very concise.")
        response = await agent.run("Say hello in one word.")

        assert len(response) > 0
        assert isinstance(response, str)

    @pytest.mark.asyncio
    async def test_async_agent_with_tools(self):
        """Test async agent can call tools."""
        def get_time() -> str:
            """Get current time."""
            return "3:00 PM"

        agent = Agent()
        agent.add_tool(get_time)

        response = await agent.run("What time is it?")

        assert "3:00" in response or "3" in response


@skip_if_no_key
class TestRealAPIEdgeCases:
    """Test edge cases with real API."""

    def test_empty_tool_result(self):
        """Test agent handles tools that return empty strings."""
        def empty_tool() -> str:
            """Get nothing."""
            return ""

        agent = AgentSync()
        agent.add_tool(empty_tool)

        response = agent.run("Call the empty tool and tell me what happened.")

        # Agent should handle empty result gracefully
        assert len(response) > 0

    def test_long_tool_result(self):
        """Test agent handles tools with long outputs."""
        def long_tool() -> str:
            """Get a lot of text."""
            return "data " * 200

        agent = AgentSync()
        agent.add_tool(long_tool)

        response = agent.run("Call the long tool.")

        # Agent should process and respond
        assert len(response) > 0
