"""
Lesson 10: Planning with ReAct

Learn the ReAct (Reasoning + Acting) pattern - agents that think before they act.

This lesson uses the agents.py framework but adds a unique twist: structured output
for explicit reasoning. While the base Agent class handles tool calling automatically,
ReAct forces the LLM to articulate its thinking at each step using Pydantic models.

Key differences from base Agent:
- Base Agent: LLM decides tools to call implicitly
- ReAct Agent: LLM must provide explicit "thought" before each action
- Base Agent: Good for straightforward tasks
- ReAct Agent: Better for complex multi-step reasoning where you want visibility
"""

import sys
from pathlib import Path

# Add parent directory to path to import agents module
sys.path.insert(0, str(Path(__file__).parent.parent))

from typing import Literal
from pydantic import BaseModel, Field
from agents import Agent, tool, ConversationMemory
from dotenv import load_dotenv

load_dotenv()


class ReActStep(BaseModel):
    """Structured output for each ReAct step"""
    thought: str = Field(description="Your reasoning about what to do next")
    action: Literal["get_weather", "calculate", "FINISH"] = Field(
        description="Tool to use, or FINISH when complete"
    )
    action_input: dict = Field(
        description="Arguments for the tool (empty dict if FINISH)",
        default_factory=dict
    )


# Tools decorated with @tool
@tool
def get_weather(city: str) -> str:
    """Get weather for a city

    Args:
        city: City name (e.g., 'Paris', 'London', 'Tokyo')
    """
    weather_db = {
        "paris": "Sunny, 22°C",
        "london": "Cloudy, 15°C",
        "tokyo": "Rainy, 18°C",
    }
    return weather_db.get(city.lower(), f"Weather data not available for {city}")


@tool
def calculate(expression: str) -> str:
    """Calculate a mathematical expression

    Args:
        expression: Math expression to evaluate (e.g., '2+2', '15*24')
    """
    try:
        # Use eval with restricted builtins for simple math
        # In production, use a proper math parser like py-expression-eval
        result = eval(expression, {"__builtins__": {}}, {})
        return str(result)
    except Exception as e:
        return f"Error: {str(e)}"


class ReActAgent:
    """Agent that uses ReAct pattern: Think → Act → Observe → Repeat

    This extends the base Agent class with explicit reasoning steps using
    structured output to force the LLM to think before acting.
    """

    def __init__(self, model: str = "gpt-4o-mini", max_iterations: int = 10):
        # Use the Agent framework with tools
        self.agent = Agent(
            model=model,
            max_iterations=max_iterations,
            system_prompt="""You are a ReAct agent that solves problems by reasoning and acting.

For each step:
1. THINK: Reason about what to do next
2. ACT: Choose a tool, or FINISH if complete
3. OBSERVE: See the result

Available tools:
- get_weather(city: str) - Get weather
- calculate(expression: str) - Math calculations

Break complex tasks into simple steps. When you have enough info, use FINISH.""",
            tools=[get_weather, calculate]
        )

        # For structured output we need direct client access
        from openai import OpenAI
        self.client = OpenAI()
        self.model = model
        self.max_iterations = max_iterations
        self.tools = {"get_weather": get_weather, "calculate": calculate}

    def run(self, user_query: str) -> str:
        """Run the ReAct agent with explicit reasoning steps"""
        print(f"\nQuery: {user_query}\n")

        # Build messages list with system prompt
        system_prompt = self.agent.memory.get_items()[0]["content"]
        messages = [
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": user_query}
        ]

        for iteration in range(self.max_iterations):
            print(f"Step {iteration + 1}:")

            # Agent thinks and decides what to do using structured output
            response = self.client.beta.chat.completions.parse(
                model=self.model,
                messages=messages,
                response_format=ReActStep
            )

            step = response.choices[0].message.parsed

            print(f"  Thought: {step.thought}")
            print(f"  Action: {step.action}")

            if step.action == "FINISH":
                # Get final answer
                messages.append({
                    "role": "assistant",
                    "content": f"Thought: {step.thought}\nAction: FINISH"
                })
                messages.append({
                    "role": "user",
                    "content": "Provide your final answer."
                })

                final_response = self.client.chat.completions.create(
                    model=self.model,
                    messages=messages
                )

                return final_response.choices[0].message.content

            # Execute the tool
            tool_func = self.tools.get(step.action)
            if tool_func:
                try:
                    observation = tool_func(**step.action_input)
                except Exception as e:
                    observation = f"Error: {str(e)}"
            else:
                observation = f"Tool '{step.action}' not found"

            print(f"  Observation: {observation}\n")

            # Add to conversation
            conversation.append({
                "role": "assistant",
                "content": f"Thought: {step.thought}\nAction: {step.action}\nInput: {step.action_input}"
            })
            conversation.append({
                "role": "user",
                "content": f"Observation: {observation}\n\nWhat's next?"
            })

        return "Max iterations reached"


def example_single_step():
    """Example 1: Simple single-step task"""
    print("=== Example 1: Simple Task ===")
    agent = ReActAgent()
    answer = agent.run("What's the weather in Paris?")
    print(f"Final Answer: {answer}")


def example_multi_step():
    """Example 2: Multi-step task that requires planning"""
    print("\n\n=== Example 2: Multi-Step Task ===")
    agent = ReActAgent()
    answer = agent.run("What's the weather in Paris and London? Calculate the temperature difference.")
    print(f"Final Answer: {answer}")


if __name__ == "__main__":
    example_single_step()
    example_multi_step()
