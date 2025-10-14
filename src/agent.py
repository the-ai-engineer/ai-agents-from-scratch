"""
Agent module providing core agent functionality with tool-calling capabilities.

This module implements the fundamental agent pattern: a model using tools in a loop.
It's designed to be simple enough for learning while being robust enough for real use.
"""

import json
import logging
from typing import Callable, Optional
from openai import OpenAI

from .tool import Tool


# Configure logging
logger = logging.getLogger(__name__)


class ConversationMemory:
    """
    Manages conversation history for an agent.

    This class encapsulates message management, keeping system prompts persistent
    while allowing user/assistant messages to be added and cleared.

    Why separate class? Keeps concerns separate and makes memory strategies pluggable.
    """

    def __init__(self, system_prompt: Optional[str] = None):
        """
        Initialize conversation memory.

        Args:
            system_prompt: Optional system instructions for the agent
        """
        self.messages = []
        if system_prompt:
            self.messages.append({"role": "system", "content": system_prompt})
            logger.debug(f"Initialized memory with system prompt: {system_prompt[:50]}...")

    def add_message(self, role: str, content: str) -> None:
        """Add a message to conversation history."""
        self.messages.append({"role": role, "content": content})
        logger.debug(f"Added {role} message: {content[:100]}...")

    def add_tool_result(self, tool_call_id: str, content: str) -> None:
        """Add a tool execution result to history."""
        self.messages.append({
            "role": "tool",
            "tool_call_id": tool_call_id,
            "content": content
        })
        logger.debug(f"Added tool result for {tool_call_id}: {content[:100]}...")

    def get_history(self) -> list[dict]:
        """Get complete conversation history."""
        return self.messages

    def clear(self) -> None:
        """Clear conversation history, keeping system messages."""
        system_messages = [msg for msg in self.messages if msg["role"] == "system"]
        self.messages = system_messages
        logger.info("Cleared conversation history (kept system messages)")

    def __len__(self) -> int:
        """Return number of messages in history."""
        return len(self.messages)


class Agent:
    """
    Autonomous agent with tool-calling capabilities.

    This implements the core agent loop pattern:
    1. Send message + available tools to LLM
    2. LLM decides to call tools or provide final answer
    3. If tools called: execute them, add results to history, repeat
    4. If final answer: return response to user

    The agent continues until it provides a final answer or reaches max iterations.
    """

    def __init__(
        self,
        model: str = "gpt-4o-mini",
        max_iterations: int = 5,
        system_prompt: Optional[str] = None,
        api_key: Optional[str] = None,
    ):
        """
        Initialize the agent.

        Args:
            model: OpenAI model to use
            max_iterations: Maximum agent loop iterations (prevents infinite loops)
            system_prompt: Optional system instructions for the agent
            api_key: Optional OpenAI API key (defaults to OPENAI_API_KEY env var)
        """
        self.client = OpenAI(api_key=api_key) if api_key else OpenAI()
        self.model = model
        self.max_iterations = max_iterations
        self.memory = ConversationMemory(system_prompt=system_prompt)
        self.tools: dict[str, Callable] = {}
        self.tool_schemas: list[dict] = []

        logger.info(f"Initialized agent with model={model}, max_iterations={max_iterations}")

    def register_tool(self, func: Callable) -> None:
        """
        Register a tool function for the agent to use.

        Args:
            func: A Python function to register as a tool. Should have a docstring.

        Example:
            def get_weather(city: str) -> str:
                '''Get the current weather for a city.'''
                return f"Weather in {city}: Sunny, 72F"

            agent.register_tool(get_weather)
        """
        tool = Tool.from_function(func)
        self.tools[tool.name] = func
        self.tool_schemas.append(tool.to_dict())
        logger.info(f"Registered tool: {tool.name}")
        logger.debug(f"Tool schema: {tool.to_dict()}")

    def register_tools(self, *funcs: Callable) -> None:
        """Register multiple tools at once."""
        for func in funcs:
            self.register_tool(func)

    def chat(self, message: str) -> str:
        """
        Send a message to the agent and get a response.

        This is the main public interface. It handles the entire agent loop internally,
        executing tools as needed until the agent provides a final answer.

        Args:
            message: User message to send to the agent

        Returns:
            Agent's final text response

        Raises:
            RuntimeError: If max iterations reached without final answer
        """
        logger.info(f"User message: {message}")

        # Add user message to history
        self.memory.add_message("user", message)

        # Run the agent loop
        response = self._run_agent_loop()

        logger.info(f"Agent response: {response[:200]}...")
        return response

    def reset(self) -> None:
        """Clear conversation history (keeps system prompt)."""
        self.memory.clear()
        logger.info("Agent conversation reset")

    def _run_agent_loop(self) -> str:
        """
        Execute the agent loop: call LLM, execute tools, repeat until final answer.

        This is the heart of the agent. It implements the "model using tools in a loop" pattern.

        Returns:
            Final text response from the agent
        """
        iteration = 0

        while iteration < self.max_iterations:
            iteration += 1
            logger.info(f"Agent loop iteration {iteration}/{self.max_iterations}")

            # Call LLM with conversation history and available tools
            response = self.client.chat.completions.create(
                model=self.model,
                messages=self.memory.get_history(),
                tools=self.tool_schemas if self.tool_schemas else None,
            )

            message = response.choices[0].message

            # Add assistant message to history
            self.memory.messages.append(message.model_dump(exclude_none=True))

            # Check if LLM wants to call tools
            if message.tool_calls:
                logger.info(f"LLM requested {len(message.tool_calls)} tool call(s)")

                # Execute all requested tool calls
                for tool_call in message.tool_calls:
                    result = self._execute_tool(tool_call)
                    self.memory.add_tool_result(tool_call.id, result)

                # Continue loop - LLM will see tool results and decide next action
                continue

            # No tool calls - this is the final answer
            final_response = message.content or ""
            logger.info(f"Agent reached final answer after {iteration} iteration(s)")
            return final_response

        # Max iterations reached without final answer
        error_msg = f"Agent reached max iterations ({self.max_iterations}) without providing final answer"
        logger.error(error_msg)
        raise RuntimeError(error_msg)

    def _execute_tool(self, tool_call) -> str:
        """
        Execute a tool function call.

        Args:
            tool_call: OpenAI tool call object

        Returns:
            JSON string containing tool execution result or error
        """
        function_name = tool_call.function.name
        logger.info(f"Executing tool: {function_name}")

        try:
            # Parse arguments
            arguments = json.loads(tool_call.function.arguments)
            logger.debug(f"Tool arguments: {arguments}")

            # Get the actual function
            if function_name not in self.tools:
                error_msg = f"Tool '{function_name}' not found in registered tools"
                logger.error(error_msg)
                return json.dumps({"error": error_msg})

            func = self.tools[function_name]

            # Execute the function
            result = func(**arguments)
            logger.info(f"Tool {function_name} executed successfully")
            logger.debug(f"Tool result: {result}")

            # Return result as JSON
            return json.dumps({"result": result})

        except json.JSONDecodeError as e:
            error_msg = f"Failed to parse tool arguments: {e}"
            logger.error(error_msg)
            return json.dumps({"error": error_msg})

        except TypeError as e:
            error_msg = f"Invalid arguments for tool '{function_name}': {e}"
            logger.error(error_msg)
            return json.dumps({"error": error_msg})

        except Exception as e:
            error_msg = f"Tool execution failed: {e}"
            logger.error(error_msg, exc_info=True)
            return json.dumps({"error": error_msg})

    def __repr__(self) -> str:
        """String representation of the agent."""
        return (
            f"Agent(model={self.model}, "
            f"tools={len(self.tools)}, "
            f"messages={len(self.memory)})"
        )
