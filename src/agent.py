"""
This file shows the core concepts of building AI agents:
- Tool calling (functions the AI can use)
- Conversation memory
- Async/sync support
- Structured outputs

Usage:
    agent = Agent()
    agent.add_tool(my_function)
    response = await agent.run("Hello!")
"""

import asyncio
import json
from typing import Callable, Optional, Type, TypeVar
from openai import AsyncOpenAI
from pydantic import BaseModel
from .tool import Tool

T = TypeVar("T", bound=BaseModel)


class Agent:
    """
    A simple AI agent that can call tools and return structured data.

    Example:
        agent = Agent("You are helpful")
        agent.add_tool(get_weather)
        response = await agent.run("What's the weather?")
    """

    def __init__(self, system_prompt: str = "", model: str = "gpt-4o-mini"):
        """Create a new agent with optional instructions."""
        self.client = AsyncOpenAI()
        self.model = model
        self.messages = []
        self.tools = {}

        if system_prompt:
            self.messages.append({"role": "system", "content": system_prompt})

    # ========================================================================
    # Adding Tools
    # ========================================================================

    def add_tool(self, func: Callable) -> "Agent":
        """Add a function the agent can call."""
        tool = Tool.from_function(func)
        self.tools[tool.name] = {
            "schema": tool.to_openai_format(),
            "func": func,
            "is_async": asyncio.iscoroutinefunction(func),
        }
        return self

    def add_tools(self, *funcs: Callable) -> "Agent":
        """Add multiple tools at once."""
        for func in funcs:
            self.add_tool(func)
        return self

    # ========================================================================
    # Running the Agent
    # ========================================================================

    async def run(self, message: str, response_format: Optional[Type[T]] = None):
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
            return await self._structured_mode(response_format)

        # Regular mode (with tools)
        return await self._agent_loop()

    def run_sync(self, message: str, response_format: Optional[Type[T]] = None):
        """Sync version of run() for Jupyter/scripts."""
        return asyncio.run(self.run(message, response_format))

    def reset(self):
        """Clear conversation history (keeps system prompt)."""
        self.messages = [m for m in self.messages if m.get("role") == "system"]

    # ========================================================================
    # Internal Implementation
    # ========================================================================

    async def _structured_mode(self, response_format: Type[T]) -> T:
        """Get a structured response (Pydantic model)."""
        response = await self.client.beta.chat.completions.parse(
            model=self.model,
            messages=self.messages,
            response_format=response_format,
        )

        message = response.choices[0].message
        self.messages.append(message.model_dump(exclude_none=True))
        return message.parsed

    async def _agent_loop(self, max_turns: int = 10) -> str:
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
            response = await self.client.responses.create(
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
                    result = await self._call_tool(item)

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

    async def _execute_tools(self, tool_calls):
        """Execute all tool calls (in parallel if async)."""
        # Create tasks for all tools
        tasks = []
        for call in tool_calls:
            tasks.append(self._call_tool(call))

        # Run them all (async tools run in parallel!)
        results = await asyncio.gather(*tasks, return_exceptions=True)

        # Add results to conversation
        for call, result in zip(tool_calls, results):
            # Handle errors gracefully
            if isinstance(result, Exception):
                content = json.dumps({"error": str(result)})
            else:
                content = result

            self.messages.append(
                {
                    "role": "tool",
                    "tool_call_id": call.id,
                    "content": content,
                }
            )

    async def _call_tool(self, call) -> str:
        """Call a single tool and return JSON result."""
        # Responses API has flat structure: call.name and call.arguments
        # (not nested under call.function like Chat Completions API)
        name = call.name
        tool = self.tools.get(name)

        if not tool:
            return json.dumps({"error": f"Tool '{name}' not found"})

        try:
            # Parse arguments
            args = json.loads(call.arguments or "{}")

            # Call the function (async or sync)
            if tool["is_async"]:
                result = await tool["func"](**args)
            else:
                # Run sync functions in executor so they don't block
                loop = asyncio.get_event_loop()
                result = await loop.run_in_executor(None, lambda: tool["func"](**args))

            return json.dumps({"result": result})

        except Exception as e:
            return json.dumps({"error": f"{type(e).__name__}: {str(e)}"})

    def __repr__(self):
        return f"Agent(tools={len(self.tools)}, messages={len(self.messages)})"
