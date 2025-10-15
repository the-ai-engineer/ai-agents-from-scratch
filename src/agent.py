"""AI agent compatible with OpenAI's Responses API."""

import json
import logging
from typing import Callable, Optional, Any
from openai import OpenAI

logger = logging.getLogger(__name__)


class Agent:
    """AI agent using OpenAI's Responses API with function calling."""

    def __init__(
        self,
        model: str = "gpt-5",
        system_prompt: Optional[str] = None,
        max_iterations: int = 10,
    ):
        self.client = OpenAI()
        self.model = model
        self.instructions = system_prompt
        self.max_iterations = max_iterations

        # Store both schema and callable
        self.tools: dict[str, dict] = {}

        # Input list replaces messages
        self.input_list: list[dict] = []

    def register_tool(self, func: Callable) -> "Agent":
        """Register a function as a tool."""
        from tool import Tool

        tool = Tool.from_function(func)
        self.tools[tool.name] = {
            "schema": tool.to_openai_format(),
            "func": func,
        }
        logger.info(f"Registered tool: {tool.name}")
        return self

    def register_tools(self, *funcs: Callable) -> "Agent":
        """Register multiple tools."""
        for func in funcs:
            self.register_tool(func)
        return self

    def chat(self, message: str) -> str:
        """Send a message and get the agent's response."""
        if not message.strip():
            raise ValueError("Message cannot be empty")

        # Add user message to input
        self.input_list.append({"role": "user", "content": message})
        logger.info(f"User: {message[:100]}...")

        return self._run_agent_loop()

    def reset(self) -> None:
        """Clear conversation history."""
        self.input_list = []
        logger.info("Conversation reset")

    def _run_agent_loop(self) -> str:
        """Main agent loop: call model → execute tools → repeat."""
        tool_schemas = [t["schema"] for t in self.tools.values()]

        for iteration in range(self.max_iterations):
            logger.debug(f"Iteration {iteration + 1}/{self.max_iterations}")

            # Call the model using responses.create
            response = self.client.responses.create(
                model=self.model,
                input=self.input_list,
                instructions=self.instructions,
                tools=tool_schemas if tool_schemas else None,
            )

            # Save all output items for next request
            self.input_list.extend(response.output)

            # Check for function calls in output
            function_calls = [
                item for item in response.output if item.type == "function_call"
            ]

            if not function_calls:
                # No more tools to call - return final text
                return response.output_text or ""

            # Execute all function calls
            logger.debug(f"Executing {len(function_calls)} function calls")
            for call in function_calls:
                result = self._execute_tool(call)

                # Append result in OpenAI's format
                self.input_list.append(
                    {
                        "type": "function_call_output",
                        "call_id": call.call_id,
                        "output": result,
                    }
                )

        raise RuntimeError(
            f"Agent exceeded {self.max_iterations} iterations. "
            f"Last response had {len(function_calls)} function calls."
        )

    def _execute_tool(self, call: Any) -> str:
        """Execute a single function call and return JSON result."""
        name = call.name
        tool_entry = self.tools.get(name)

        if tool_entry is None:
            error_msg = f"Tool '{name}' not registered"
            logger.error(error_msg)
            return json.dumps({"error": error_msg})

        try:
            args = json.loads(call.arguments or "{}")
            result = tool_entry["func"](**args)
            return json.dumps({"result": result})

        except json.JSONDecodeError as e:
            logger.error(f"Invalid JSON in {name}: {call.arguments}")
            return json.dumps({"error": f"Invalid JSON: {e}"})

        except Exception as e:
            logger.exception(f"Tool '{name}' failed")
            return json.dumps({"error": f"{type(e).__name__}: {e}"})

    def __repr__(self) -> str:
        return (
            f"Agent(model={self.model}, "
            f"tools={len(self.tools)}, "
            f"inputs={len(self.input_list)})"
        )
