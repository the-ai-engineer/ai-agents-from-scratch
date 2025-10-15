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
            # Step 1: Call the model with Responses API
            response = self.client.responses.create(
                model=self.model,
                input=self.messages,  # Responses API uses 'input' not 'messages'
                tools=tool_schemas if tool_schemas else None,
            )

            # Step 2: Process output items
            # Responses API returns an array of 'output' items (not 'choices')
            has_tool_calls = False
            final_text = None

            for item in response.output:
                if item.type == "message":
                    # Extract text from message content
                    if item.content and len(item.content) > 0:
                        final_text = item.content[0].text

                    # Add to conversation history
                    self.messages.append({
                        "role": "assistant",
                        "content": final_text or ""
                    })

                elif item.type == "function_call":
                    # Mark that we have tool calls to execute
                    has_tool_calls = True

                    # First, add the function call itself to conversation history
                    self.messages.append({
                        "type": "function_call",
                        "call_id": item.call_id,
                        "name": item.name,
                        "arguments": item.arguments,
                    })

                    # Execute the tool call
                    result = self._call_tool(item)

                    # Then add the tool result to conversation
                    # Responses API uses "type" not "role", and "output" not "content"
                    self.messages.append({
                        "type": "function_call_output",
                        "call_id": item.call_id,
                        "output": result,
                    })

            # Step 3: If no tool calls, return the final answer
            if not has_tool_calls and final_text:
                return final_text

        raise RuntimeError(f"Agent didn't finish in {max_turns} turns")

    def _call_tool(self, call) -> str:
        """Call a single tool and return JSON result."""
        # Responses API has flat structure: call.name and call.arguments
        # (not nested under call.function like Chat Completions API)
        name = call.name
        tool = self.tools.get(name)

        if not tool:
            return json.dumps({"error": f"Tool '{name}' not found"})

        try:
            # Parse arguments from JSON
            args = json.loads(call.arguments or "{}")

            # Call the function
            result = tool["func"](**args)

            # Return result as JSON
            return json.dumps({"result": result})

        except Exception as e:
            # Return error as JSON
            return json.dumps({"error": f"{type(e).__name__}: {str(e)}"})

    def __repr__(self):
        return f"AgentSync(tools={len(self.tools)}, messages={len(self.messages)})"
