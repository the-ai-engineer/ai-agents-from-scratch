"""
Synchronous AI agent - simpler for initial teaching.

This version uses only sync code to make it easier to understand.
Once you understand this, the async version adds parallel execution.

Usage:
    agent = AgentSync()
    agent.add_tool(my_function)
    response = agent.run("Hello!")
"""

import json
from typing import Callable, Optional, Type, TypeVar
from openai import OpenAI
from pydantic import BaseModel
from .tool import Tool

T = TypeVar("T", bound=BaseModel)


class AgentSync:
    """
    A simple synchronous AI agent.

    This agent can:
    - Call tools (functions you define)
    - Remember conversation history
    - Return structured data (Pydantic models)

    Example:
        agent = AgentSync("You are helpful")
        agent.add_tool(get_weather)
        response = agent.run("What's the weather?")
    """

    def __init__(self, system_prompt: str = "", model: str = "gpt-4o-mini"):
        """Create a new agent."""
        self.client = OpenAI()
        self.model = model
        self.messages = []
        self.tools = {}

        if system_prompt:
            self.messages.append({"role": "system", "content": system_prompt})

    # ========================================================================
    # Adding Tools
    # ========================================================================

    def add_tool(self, func: Callable) -> "AgentSync":
        """Add a function the agent can call."""
        tool = Tool.from_function(func)
        self.tools[tool.name] = {
            "schema": tool.to_openai_format(),
            "func": func,
        }
        return self

    def add_tools(self, *funcs: Callable) -> "AgentSync":
        """Add multiple tools at once."""
        for func in funcs:
            self.add_tool(func)
        return self

    # ========================================================================
    # Running the Agent
    # ========================================================================

    def run(self, message: str, response_format: Optional[Type[T]] = None):
        """
        Run the agent with a message.

        Args:
            message: What to ask the agent
            response_format: Optional Pydantic model for structured output

        Returns:
            String response or structured Pydantic model
        """
        # Add user message to conversation
        self.messages.append({"role": "user", "content": message})

        # Structured output mode (no tools)
        if response_format:
            return self._structured_mode(response_format)

        # Regular mode (with tools)
        return self._agent_loop()

    def reset(self):
        """Clear conversation history (keeps system prompt)."""
        self.messages = [m for m in self.messages if m.get("role") == "system"]

    # ========================================================================
    # Internal Implementation
    # ========================================================================

    def _structured_mode(self, response_format: Type[T]) -> T:
        """Get a structured response (Pydantic model)."""
        response = self.client.beta.chat.completions.parse(
            model=self.model,
            messages=self.messages,
            response_format=response_format,
        )

        message = response.choices[0].message
        self.messages.append(message.model_dump(exclude_none=True))
        return message.parsed

    def _agent_loop(self, max_turns: int = 10) -> str:
        """
        Main loop: call model, execute tools, repeat.

        This is the core of how an agent works:
        1. Call the AI model
        2. If it wants to use tools, execute them
        3. Send results back to the model
        4. Repeat until the model gives a final answer
        """
        tool_schemas = [t["schema"] for t in self.tools.values()]

        for turn in range(max_turns):
            # Step 1: Call the model
            response = self.client.chat.completions.create(
                model=self.model,
                messages=self.messages,
                tools=tool_schemas if tool_schemas else None,
            )

            message = response.choices[0].message
            self.messages.append(message.model_dump(exclude_none=True))

            # Step 2: Check if we're done
            if not message.tool_calls:
                return message.content or ""

            # Step 3: Execute tools sequentially
            for call in message.tool_calls:
                result = self._call_tool(call)

                self.messages.append(
                    {
                        "role": "tool",
                        "tool_call_id": call.id,
                        "content": result,
                    }
                )

        raise RuntimeError(f"Agent didn't finish in {max_turns} turns")

    def _call_tool(self, call) -> str:
        """Call a single tool and return JSON result."""
        name = call.function.name
        tool = self.tools.get(name)

        if not tool:
            return json.dumps({"error": f"Tool '{name}' not found"})

        try:
            # Parse arguments from JSON
            args = json.loads(call.function.arguments or "{}")

            # Call the function
            result = tool["func"](**args)

            # Return result as JSON
            return json.dumps({"result": result})

        except Exception as e:
            # Return error as JSON
            return json.dumps({"error": f"{type(e).__name__}: {str(e)}"})

    def __repr__(self):
        return f"AgentSync(tools={len(self.tools)}, messages={len(self.messages)})"
