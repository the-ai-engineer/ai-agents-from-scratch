"""
Unit tests for Agent and AgentSync classes.

These tests focus on internal logic without mocking the OpenAI API:
1. Initialization and configuration
2. Tool registration and management
3. Internal helper methods (_call_tool, error handling)
4. State management (reset, messages)
5. Async tool detection and parallel execution
"""

import pytest
import asyncio
from unittest.mock import Mock
from src.agent import Agent
from src.agent_sync import AgentSync


class TestAgentSyncBasics:
    """Test synchronous agent initialization and configuration."""

    def test_agent_initialization(self):
        """Test agent can be created with various configurations."""
        # Default agent
        agent = AgentSync()
        assert agent.model == "gpt-4o-mini"
        assert len(agent.messages) == 0
        assert len(agent.tools) == 0

        # Configured agent
        agent = AgentSync(system_prompt="You are a bot.", model="gpt-4")
        assert agent.model == "gpt-4"
        assert len(agent.messages) == 1
        assert agent.messages[0]["role"] == "system"
        assert agent.messages[0]["content"] == "You are a bot."

    def test_tool_registration_chainable(self):
        """Test tool registration returns self for chaining."""
        def weather(city: str) -> str:
            """Get weather."""
            return f"Weather in {city}"

        agent = AgentSync()
        result = agent.add_tool(weather)

        assert result is agent  # Chainable API
        assert "weather" in agent.tools

    def test_multiple_tool_registration(self):
        """Test registering multiple tools at once."""
        def weather(city: str) -> str:
            """Get weather."""
            return "sunny"

        def calculate(x: str) -> str:
            """Calculate."""
            return "42"

        agent = AgentSync()
        agent.add_tools(weather, calculate)

        assert len(agent.tools) == 2
        assert "weather" in agent.tools
        assert "calculate" in agent.tools
        # Verify schema is stored
        assert "schema" in agent.tools["weather"]
        assert "func" in agent.tools["weather"]

    def test_reset_clears_conversation_keeps_system_prompt(self):
        """Test reset clears conversation but preserves system prompt."""
        agent = AgentSync(system_prompt="You are helpful.")
        agent.messages.append({"role": "user", "content": "Hello"})
        agent.messages.append({"role": "assistant", "content": "Hi!"})

        agent.reset()

        # Only system prompt remains
        assert len(agent.messages) == 1
        assert agent.messages[0]["role"] == "system"
        assert agent.messages[0]["content"] == "You are helpful."

    def test_reset_keeps_tools(self):
        """Test reset doesn't affect registered tools."""
        def search(q: str) -> str:
            """Search."""
            return "results"

        agent = AgentSync()
        agent.add_tool(search)
        agent.messages.append({"role": "user", "content": "Hello"})

        agent.reset()

        assert len(agent.tools) == 1
        assert "search" in agent.tools


class TestAgentAsyncBasics:
    """Test async agent initialization."""

    def test_agent_initialization(self):
        """Test async agent initialization."""
        agent = Agent()
        assert agent.model == "gpt-4o-mini"
        assert len(agent.messages) == 0

        agent = Agent(system_prompt="You are a bot.")
        assert len(agent.messages) == 1
        assert agent.messages[0]["content"] == "You are a bot."

    def test_async_tool_detection(self):
        """Test agent correctly identifies async tools."""
        def sync_tool(x: str) -> str:
            """Sync tool."""
            return "sync"

        async def async_tool(x: str) -> str:
            """Async tool."""
            return "async"

        agent = Agent()
        agent.add_tool(sync_tool)
        agent.add_tool(async_tool)

        assert agent.tools["sync_tool"]["is_async"] is False
        assert agent.tools["async_tool"]["is_async"] is True


class TestToolExecution:
    """Test internal tool execution logic without mocking OpenAI."""

    def test_successful_tool_execution(self):
        """Test _call_tool executes function and returns JSON."""
        def get_weather(city: str) -> str:
            """Get weather."""
            return f"Sunny in {city}"

        agent = AgentSync()
        agent.add_tool(get_weather)

        # Create mock tool call (Responses API format - flat structure)
        mock_call = Mock()
        mock_call.call_id = "call_123"
        mock_call.name = "get_weather"  # Direct attribute, not nested
        mock_call.arguments = '{"city": "Paris"}'

        result = agent._call_tool(mock_call)

        # Should return JSON with result
        assert "Paris" in result
        assert "result" in result
        assert "Sunny" in result

    def test_tool_not_found_error(self):
        """Test calling non-existent tool returns error."""
        agent = AgentSync()

        mock_call = Mock()
        mock_call.name = "nonexistent"
        mock_call.arguments = '{}'

        result = agent._call_tool(mock_call)

        assert "error" in result.lower()
        assert "not found" in result.lower()

    def test_malformed_json_arguments_error(self):
        """Test malformed JSON in arguments is handled."""
        def weather(city: str) -> str:
            """Get weather."""
            return "Sunny"

        agent = AgentSync()
        agent.add_tool(weather)

        mock_call = Mock()
        mock_call.name = "weather"
        mock_call.arguments = 'not valid json'

        result = agent._call_tool(mock_call)

        assert "error" in result.lower()

    def test_tool_raises_exception(self):
        """Test tool that raises exception returns error."""
        def broken_tool(x: str) -> str:
            """Broken tool."""
            raise ValueError("Something went wrong")

        agent = AgentSync()
        agent.add_tool(broken_tool)

        mock_call = Mock()
        mock_call.name = "broken_tool"
        mock_call.arguments = '{"x": "test"}'

        result = agent._call_tool(mock_call)

        assert "error" in result.lower()
        assert "ValueError" in result

    @pytest.mark.asyncio
    async def test_async_tool_execution(self):
        """Test async tools execute properly."""
        async def async_weather(city: str) -> str:
            """Get weather async."""
            await asyncio.sleep(0.001)  # Simulate async work
            return f"Sunny in {city}"

        agent = Agent()
        agent.add_tool(async_weather)

        mock_call = Mock()
        mock_call.call_id = "call_123"
        mock_call.name = "async_weather"
        mock_call.arguments = '{"city": "Tokyo"}'

        result = await agent._call_tool(mock_call)

        assert "Tokyo" in result
        assert "result" in result


class TestParallelExecution:
    """Test async agent's parallel tool execution."""

    @pytest.mark.asyncio
    async def test_async_tools_run_in_parallel(self):
        """Test multiple async tools execute concurrently."""
        call_order = []

        async def tool1(x: str) -> str:
            """Tool 1."""
            call_order.append("tool1_start")
            await asyncio.sleep(0.02)
            call_order.append("tool1_end")
            return "result1"

        async def tool2(x: str) -> str:
            """Tool 2."""
            call_order.append("tool2_start")
            await asyncio.sleep(0.02)
            call_order.append("tool2_end")
            return "result2"

        agent = Agent()
        agent.add_tools(tool1, tool2)

        # Mock tool calls (Responses API format)
        mock_call1 = Mock()
        mock_call1.call_id = "call_1"
        mock_call1.name = "tool1"
        mock_call1.arguments = '{"x": "a"}'

        mock_call2 = Mock()
        mock_call2.call_id = "call_2"
        mock_call2.name = "tool2"
        mock_call2.arguments = '{"x": "b"}'

        await agent._execute_tools([mock_call1, mock_call2])

        # Both tools should start before either ends (parallel execution)
        # If sequential, order would be: tool1_start, tool1_end, tool2_start, tool2_end
        # If parallel, tool2_start happens before tool1_end
        assert call_order.index("tool1_start") < call_order.index("tool1_end")
        assert call_order.index("tool2_start") < call_order.index("tool2_end")
        assert call_order.index("tool2_start") < call_order.index("tool1_end")

    @pytest.mark.asyncio
    async def test_tool_errors_dont_break_parallel_execution(self):
        """Test one tool failing doesn't stop others."""
        async def good_tool(x: str) -> str:
            """Good tool."""
            await asyncio.sleep(0.01)
            return "success"

        async def bad_tool(x: str) -> str:
            """Bad tool."""
            raise ValueError("I fail")

        agent = Agent()
        agent.add_tools(good_tool, bad_tool)

        mock_call1 = Mock()
        mock_call1.call_id = "call_1"
        mock_call1.name = "good_tool"
        mock_call1.arguments = '{"x": "a"}'

        mock_call2 = Mock()
        mock_call2.call_id = "call_2"
        mock_call2.name = "bad_tool"
        mock_call2.arguments = '{"x": "b"}'

        # Should not raise, errors are captured
        await agent._execute_tools([mock_call1, mock_call2])

        # Both results should be added to messages
        assert len(agent.messages) == 2

        # Good tool result should be present
        assert any("success" in msg.get("content", "") for msg in agent.messages)

        # Bad tool error should be captured
        assert any("error" in msg.get("content", "").lower() for msg in agent.messages)


class TestStateManagement:
    """Test message and tool state management."""

    def test_tool_schemas_generated_correctly(self):
        """Test tool schemas are in correct OpenAI Responses API format."""
        def search(query: str, limit: str = "10") -> str:
            """Search for information with optional limit."""
            return "results"

        agent = AgentSync()
        agent.add_tool(search)

        schema = agent.tools["search"]["schema"]

        # Responses API uses flat format (no nested "function" key)
        assert schema["type"] == "function"
        assert schema["name"] == "search"
        assert schema["description"] == "Search for information with optional limit."
        assert "query" in schema["parameters"]["properties"]
        assert "limit" in schema["parameters"]["properties"]
        assert "query" in schema["parameters"]["required"]
        assert "limit" not in schema["parameters"]["required"]  # Has default

    def test_messages_list_accessible(self):
        """Test messages list can be inspected."""
        agent = AgentSync(system_prompt="You are helpful.")
        agent.messages.append({"role": "user", "content": "Hello"})

        assert len(agent.messages) == 2
        assert agent.messages[0]["role"] == "system"
        assert agent.messages[1]["role"] == "user"
        assert agent.messages[1]["content"] == "Hello"
