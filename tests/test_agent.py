"""
Simplified tests for Agent and ConversationMemory.

These tests focus on core behaviors students need to understand:
1. How conversation memory works
2. How to register and use tools
3. How the agent loop executes
4. How errors are handled
"""

import pytest
from unittest.mock import Mock, patch
from src.agent import Agent, ConversationMemory


class TestConversationMemory:
    """Test conversation memory management."""

    def test_memory_basics(self):
        """Test basic memory operations."""
        # Empty memory
        memory = ConversationMemory()
        assert len(memory) == 0

        # Memory with system prompt persists it
        memory = ConversationMemory(system_prompt="You are helpful.")
        assert len(memory) == 1
        assert memory.messages[0]["role"] == "system"

    def test_conversation_flow(self):
        """Test a realistic conversation with multiple message types."""
        memory = ConversationMemory(system_prompt="You are helpful.")

        memory.add_message("user", "What's the weather?")
        memory.add_message("assistant", "Let me check.")
        memory.add_tool_result("call_1", '{"result": "Sunny"}')
        memory.add_message("assistant", "It's sunny!")

        # system + user + assistant + tool + assistant
        assert len(memory) == 5

    def test_clear_preserves_system_prompt(self):
        """Test that clear() keeps system prompt but removes other messages."""
        memory = ConversationMemory(system_prompt="You are helpful.")
        memory.add_message("user", "Hello")
        memory.add_message("assistant", "Hi!")

        memory.clear()

        assert len(memory) == 1
        assert memory.messages[0]["role"] == "system"


class TestAgentBasics:
    """Test agent initialization and configuration."""

    def test_agent_initialization(self):
        """Test agent can be created with various configurations."""
        # Default agent
        agent = Agent()
        assert agent.model == "gpt-4o-mini"
        assert agent.max_iterations == 5
        assert len(agent.tools) == 0

        # Configured agent
        agent = Agent(
            model="gpt-4",
            max_iterations=10,
            system_prompt="You are a bot."
        )
        assert agent.model == "gpt-4"
        assert agent.max_iterations == 10
        assert len(agent.memory) == 1

    def test_tool_registration(self):
        """Test registering tools with an agent."""
        def weather(city: str) -> str:
            """Get weather."""
            return f"Weather in {city}"

        def calculate(x: str, y: str) -> str:
            """Calculate."""
            return "result"

        agent = Agent()

        # Register single tool
        agent.register_tool(weather)
        assert "weather" in agent.tools
        assert len(agent.tool_schemas) == 1

        # Register multiple tools
        agent.register_tools(calculate)
        assert len(agent.tools) == 2

    def test_reset(self):
        """Test reset clears conversation but keeps tools."""
        def search(q: str) -> str:
            """Search."""
            return "results"

        agent = Agent(system_prompt="You are helpful.")
        agent.register_tool(search)
        agent.memory.add_message("user", "Hello")

        agent.reset()

        # Conversation cleared but system prompt remains
        assert len(agent.memory) == 1
        assert agent.memory.messages[0]["role"] == "system"

        # Tools still registered
        assert len(agent.tools) == 1


class TestToolExecution:
    """Test tool execution and error handling."""

    def test_successful_tool_execution(self):
        """Test tools execute successfully."""
        def get_weather(city: str) -> str:
            """Get weather."""
            return f"Sunny in {city}"

        agent = Agent()
        agent.register_tool(get_weather)

        # Mock a tool call
        mock_call = Mock()
        mock_call.function.name = "get_weather"
        mock_call.function.arguments = '{"city": "Paris"}'

        result = agent._execute_tool(mock_call)

        assert "Paris" in result
        assert "result" in result  # JSON format

    def test_tool_error_handling(self):
        """Test that tool errors are handled gracefully."""
        agent = Agent()

        # Tool not found
        mock_call = Mock()
        mock_call.function.name = "nonexistent"
        mock_call.function.arguments = '{}'
        result = agent._execute_tool(mock_call)
        assert "error" in result.lower()

        # Malformed JSON
        def weather(city: str) -> str:
            """Get weather."""
            return "Sunny"

        agent.register_tool(weather)
        mock_call.function.name = "weather"
        mock_call.function.arguments = 'not json'
        result = agent._execute_tool(mock_call)
        assert "error" in result.lower()


class TestAgentLoop:
    """Test the agent loop with mocked API calls."""

    @patch('src.agent.OpenAI')
    def test_simple_chat_no_tools(self, mock_openai):
        """Test a simple chat where LLM returns immediate answer."""
        # Mock API to return a direct response
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

        # Test
        agent = Agent()
        response = agent.chat("What's the capital of France?")

        assert "Paris" in response
        assert mock_client.chat.completions.create.call_count == 1

    @patch('src.agent.OpenAI')
    def test_chat_with_tool_call(self, mock_openai):
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

        # Test
        agent = Agent()
        agent.register_tool(get_weather)
        response = agent.chat("What's the weather in Tokyo?")

        assert "sunny" in response.lower()
        assert "tokyo" in response.lower()
        assert mock_client.chat.completions.create.call_count == 2

    @patch('src.agent.OpenAI')
    def test_max_iterations_prevents_infinite_loop(self, mock_openai):
        """Test agent stops after max iterations."""
        def search(q: str) -> str:
            """Search."""
            return "results"

        mock_client = Mock()
        mock_openai.return_value = mock_client

        # Mock LLM to always request tools (never give final answer)
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

        # Test
        agent = Agent(max_iterations=3)
        agent.register_tool(search)

        with pytest.raises(RuntimeError, match="max iterations"):
            agent.chat("Search for something")

        assert mock_client.chat.completions.create.call_count == 3
