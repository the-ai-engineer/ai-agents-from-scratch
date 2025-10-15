"""
Tests for Agent and AgentSync classes.

These tests focus on core behaviors:
1. Tool registration
2. Agent loop execution
3. Structured output
4. Error handling
"""

import pytest
import asyncio
from unittest.mock import Mock, patch, AsyncMock
from src.agent import Agent
from src.agent_sync import AgentSync
from pydantic import BaseModel


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

    def test_tool_registration(self):
        """Test registering tools with an agent."""
        def weather(city: str) -> str:
            """Get weather."""
            return f"Weather in {city}"

        def calculate(x: str) -> str:
            """Calculate."""
            return "result"

        agent = AgentSync()

        # Register single tool (chainable)
        result = agent.add_tool(weather)
        assert result is agent  # Chainable
        assert "weather" in agent.tools
        assert len(agent.tools) == 1

        # Register multiple tools
        agent.add_tools(calculate)
        assert len(agent.tools) == 2

    def test_reset(self):
        """Test reset clears conversation but keeps tools and system prompt."""
        def search(q: str) -> str:
            """Search."""
            return "results"

        agent = AgentSync(system_prompt="You are helpful.")
        agent.add_tool(search)
        agent.messages.append({"role": "user", "content": "Hello"})

        agent.reset()

        # Conversation cleared but system prompt remains
        assert len(agent.messages) == 1
        assert agent.messages[0]["role"] == "system"

        # Tools still registered
        assert len(agent.tools) == 1


class TestAgentAsyncBasics:
    """Test async agent initialization."""

    def test_agent_initialization(self):
        """Test async agent can be created with configurations."""
        agent = Agent()
        assert agent.model == "gpt-4o-mini"
        assert len(agent.messages) == 0

        agent = Agent(system_prompt="You are a bot.")
        assert len(agent.messages) == 1
        assert agent.messages[0]["role"] == "system"

    def test_tool_registration(self):
        """Test registering tools with async agent."""
        def weather(city: str) -> str:
            """Get weather."""
            return f"Weather in {city}"

        agent = Agent()
        result = agent.add_tool(weather)
        assert result is agent  # Chainable
        assert "weather" in agent.tools
        assert agent.tools["weather"]["is_async"] == False

    def test_async_tool_detection(self):
        """Test agent detects async tools."""
        async def async_search(q: str) -> str:
            """Async search."""
            await asyncio.sleep(0)
            return "results"

        agent = Agent()
        agent.add_tool(async_search)
        assert agent.tools["async_search"]["is_async"] == True


class TestToolExecution:
    """Test tool execution and error handling."""

    def test_successful_tool_execution_sync(self):
        """Test tools execute successfully in sync agent."""
        def get_weather(city: str) -> str:
            """Get weather."""
            return f"Sunny in {city}"

        agent = AgentSync()
        agent.add_tool(get_weather)

        # Mock a tool call
        mock_call = Mock()
        mock_call.id = "call_123"
        mock_call.function.name = "get_weather"
        mock_call.function.arguments = '{"city": "Paris"}'

        result = agent._call_tool(mock_call)

        assert "Paris" in result
        assert "result" in result  # JSON format

    def test_tool_error_handling_sync(self):
        """Test that tool errors are handled gracefully."""
        agent = AgentSync()

        # Tool not found
        mock_call = Mock()
        mock_call.function.name = "nonexistent"
        mock_call.function.arguments = '{}'
        result = agent._call_tool(mock_call)
        assert "error" in result.lower()

        # Malformed JSON
        def weather(city: str) -> str:
            """Get weather."""
            return "Sunny"

        agent.add_tool(weather)
        mock_call.function.name = "weather"
        mock_call.function.arguments = 'not json'
        result = agent._call_tool(mock_call)
        assert "error" in result.lower()

    @pytest.mark.asyncio
    async def test_async_tool_execution(self):
        """Test async tools execute properly."""
        async def async_weather(city: str) -> str:
            """Get weather async."""
            await asyncio.sleep(0.01)
            return f"Sunny in {city}"

        agent = Agent()
        agent.add_tool(async_weather)

        mock_call = Mock()
        mock_call.id = "call_123"
        mock_call.function.name = "async_weather"
        mock_call.function.arguments = '{"city": "Tokyo"}'

        result = await agent._call_tool(mock_call)

        assert "Tokyo" in result
        assert "result" in result


class TestAgentLoopSync:
    """Test the synchronous agent loop with mocked API calls."""

    @patch('src.agent_sync.OpenAI')
    def test_simple_run_no_tools(self, mock_openai):
        """Test a simple run where LLM returns immediate answer."""
        mock_client = Mock()
        mock_openai.return_value = mock_client

        mock_message = Mock()
        mock_message.content = "The capital of France is Paris."
        mock_message.tool_calls = None
        mock_message.model_dump.return_value = {
            "role": "assistant",
            "content": "The capital of France is Paris."
        }

        mock_client.chat.completions.create.return_value = Mock(
            choices=[Mock(message=mock_message)]
        )

        agent = AgentSync()
        response = agent.run("What's the capital of France?")

        assert "Paris" in response
        assert mock_client.chat.completions.create.call_count == 1

    @patch('src.agent_sync.OpenAI')
    def test_run_with_tool_call(self, mock_openai):
        """Test agent calls tool then provides final answer."""
        def get_weather(city: str) -> str:
            """Get weather."""
            return f"Sunny in {city}"

        mock_client = Mock()
        mock_openai.return_value = mock_client

        # First call: LLM requests tool
        mock_tool_call = Mock()
        mock_tool_call.id = "call_123"
        mock_tool_call.function.name = "get_weather"
        mock_tool_call.function.arguments = '{"city": "Tokyo"}'

        first_msg = Mock()
        first_msg.content = None
        first_msg.tool_calls = [mock_tool_call]
        first_msg.model_dump.return_value = {
            "role": "assistant",
            "tool_calls": [{"id": "call_123"}]
        }

        # Second call: LLM provides final answer
        second_msg = Mock()
        second_msg.content = "It's sunny in Tokyo!"
        second_msg.tool_calls = None
        second_msg.model_dump.return_value = {
            "role": "assistant",
            "content": "It's sunny in Tokyo!"
        }

        mock_client.chat.completions.create.side_effect = [
            Mock(choices=[Mock(message=first_msg)]),
            Mock(choices=[Mock(message=second_msg)])
        ]

        agent = AgentSync()
        agent.add_tool(get_weather)
        response = agent.run("What's the weather in Tokyo?")

        assert "sunny" in response.lower()
        assert "tokyo" in response.lower()
        assert mock_client.chat.completions.create.call_count == 2

    @patch('src.agent_sync.OpenAI')
    def test_max_turns_prevents_infinite_loop(self, mock_openai):
        """Test agent stops after max turns."""
        def search(q: str) -> str:
            """Search."""
            return "results"

        mock_client = Mock()
        mock_openai.return_value = mock_client

        # Mock LLM to always request tools
        mock_tool_call = Mock()
        mock_tool_call.id = "call_123"
        mock_tool_call.function.name = "search"
        mock_tool_call.function.arguments = '{"q": "test"}'

        mock_msg = Mock()
        mock_msg.content = None
        mock_msg.tool_calls = [mock_tool_call]
        mock_msg.model_dump.return_value = {
            "role": "assistant",
            "tool_calls": [{"id": "call_123"}]
        }

        mock_client.chat.completions.create.return_value = Mock(
            choices=[Mock(message=mock_msg)]
        )

        agent = AgentSync()
        agent.add_tool(search)

        with pytest.raises(RuntimeError, match="didn't finish"):
            agent.run("Search for something")


class TestStructuredOutput:
    """Test structured output with Pydantic models."""

    @patch('src.agent_sync.OpenAI')
    def test_structured_output_sync(self, mock_openai):
        """Test agent returns structured Pydantic model."""
        class Weather(BaseModel):
            city: str
            temperature: int

        mock_client = Mock()
        mock_openai.return_value = mock_client

        # Mock parsed response
        mock_parsed = Weather(city="Paris", temperature=22)
        mock_message = Mock()
        mock_message.parsed = mock_parsed
        mock_message.model_dump.return_value = {
            "role": "assistant",
            "content": None
        }

        mock_client.beta.chat.completions.parse.return_value = Mock(
            choices=[Mock(message=mock_message)]
        )

        agent = AgentSync()
        result = agent.run("Get weather", response_format=Weather)

        assert isinstance(result, Weather)
        assert result.city == "Paris"
        assert result.temperature == 22
        assert mock_client.beta.chat.completions.parse.called


class TestAgentLoopAsync:
    """Test async agent loop."""

    @pytest.mark.asyncio
    @patch('src.agent.AsyncOpenAI')
    async def test_simple_run_async(self, mock_openai):
        """Test async agent run."""
        mock_client = AsyncMock()
        mock_openai.return_value = mock_client

        mock_message = Mock()
        mock_message.content = "Hello!"
        mock_message.tool_calls = None
        mock_message.model_dump.return_value = {
            "role": "assistant",
            "content": "Hello!"
        }

        mock_client.chat.completions.create.return_value = Mock(
            choices=[Mock(message=mock_message)]
        )

        agent = Agent()
        response = await agent.run("Hi")

        assert "Hello" in response
        assert mock_client.chat.completions.create.called

    @pytest.mark.asyncio
    @patch('src.agent.AsyncOpenAI')
    async def test_parallel_tool_execution(self, mock_openai):
        """Test async tools run in parallel."""
        call_order = []

        async def tool1(x: str) -> str:
            """Tool 1."""
            call_order.append("tool1_start")
            await asyncio.sleep(0.01)
            call_order.append("tool1_end")
            return "result1"

        async def tool2(x: str) -> str:
            """Tool 2."""
            call_order.append("tool2_start")
            await asyncio.sleep(0.01)
            call_order.append("tool2_end")
            return "result2"

        agent = Agent()
        agent.add_tools(tool1, tool2)

        # Mock tool calls
        mock_call1 = Mock()
        mock_call1.id = "call_1"
        mock_call1.function.name = "tool1"
        mock_call1.function.arguments = '{"x": "a"}'

        mock_call2 = Mock()
        mock_call2.id = "call_2"
        mock_call2.function.name = "tool2"
        mock_call2.function.arguments = '{"x": "b"}'

        await agent._execute_tools([mock_call1, mock_call2])

        # Both tools should start before either ends (parallel execution)
        assert call_order.index("tool1_start") < call_order.index("tool1_end")
        assert call_order.index("tool2_start") < call_order.index("tool2_end")
        assert call_order.index("tool2_start") < call_order.index("tool1_end")
