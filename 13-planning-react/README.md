# Lesson 12: Make Your Agents Think with ReAct

## What You'll Learn

In this lesson, you'll learn how to make your agents more intelligent and reliable by teaching them to reason before acting. You'll implement the ReAct pattern—Reasoning + Acting—which forces agents to think through problems step-by-step before taking action.

ReAct transforms your agents from reactive tools into thoughtful problem-solvers. Instead of blindly calling functions, they explain their reasoning, plan their approach, and adapt when things don't go as expected.

By the end of this lesson, you'll understand what ReAct is and why it improves agent performance, how to implement the ReAct loop from scratch, when to use planning versus direct action, and how to debug agents by reading their thought process.

This pattern is critical for complex tasks where reliability matters. Simple questions don't need ReAct. But multi-step problems, tasks requiring judgment, and situations where mistakes are costly all benefit from systematic reasoning.

## The Problem

Basic agents jump straight to action. You ask "What's the weather in Paris and London?" and they immediately call get_weather twice. This works for simple tasks, but fails when problems require planning.

Consider this request: "Find the three best-rated restaurants in Paris that are open tonight and under $100 per person." A basic agent might search restaurants, get confused about filtering criteria, miss the availability check, or return results in the wrong format.

The core issue: agents that act without thinking can't break down complex problems, adapt when initial approaches fail, or explain their reasoning when things go wrong.

The solution is ReAct—a pattern that forces agents to articulate their thinking before every action. This simple change dramatically improves reliability on complex tasks.

## How ReAct Works

ReAct stands for **Reasoning + Acting**. Instead of going straight from question to tool call, the agent follows this cycle:

1. **Think**: Reason about what needs to be done ("I need to find the weather for both cities first")
2. **Act**: Call a tool to gather information or take action
3. **Observe**: Review the result and decide what to do next
4. **Repeat**: Continue until the task is complete

This explicit reasoning step changes everything. The agent can't just react—it has to plan.

### Basic Agent vs ReAct Agent

**Basic Agent (from previous lessons):**
```
User: "What's the weather in Paris and London?"
→ Call get_weather(Paris)
→ Call get_weather(London)
→ Compare and answer
```

The agent jumps to action. It works, but there's no visible planning.

**ReAct Agent:**
```
User: "What's the weather in Paris and London?"

Thought: "I need to get weather data for both cities before I can compare them."
Action: get_weather(Paris)
Observation: "Sunny, 22°C"

Thought: "Now I need London's weather to make the comparison."
Action: get_weather(London)
Observation: "Cloudy, 15°C"

Thought: "I have both temperatures. Paris is warmer. I can provide a complete answer now."
Answer: "Paris is warmer at 22°C and sunny, while London is cooler at 15°C and cloudy."
```

The agent explains each step. This makes it more reliable, easier to debug, and better at handling complex tasks.

## Why ReAct Improves Agent Performance

ReAct provides five key benefits:

1. **Better reasoning**: Forcing the agent to articulate thoughts improves its planning
2. **More reliable**: Planning reduces mistakes on multi-step tasks
3. **Easier to debug**: You can see exactly what the agent was thinking when it made mistakes
4. **Handles complexity**: Breaking tasks into explicit steps prevents the agent from getting overwhelmed
5. **Self-correcting**: Agents can recognize mistakes and adjust their approach

The research backs this up. The original ReAct paper showed 15-20% improvement in task success rates on complex reasoning tasks compared to basic agent loops.

## The ReAct Loop

Here's the core pattern:

```python
while not task_complete:
    # 1. THINK: What should I do next?
    thought = agent.reason_about_current_state()

    # 2. DECIDE: What action should I take?
    action = agent.choose_action(thought)

    # 3. ACT: Execute the tool
    observation = execute_tool(action)

    # 4. OBSERVE: Update state with the result
    current_state.add(observation)

    # 5. CHECK: Are we done?
    task_complete = agent.has_achieved_goal()
```

Each iteration requires explicit reasoning before action. This structure ensures the agent always knows why it's doing something.

## Implementation: Structured Thinking

Use Pydantic to force structured reasoning:

```python
from pydantic import BaseModel, Field
from typing import Literal

class ReActStep(BaseModel):
    thought: str = Field(
        description="Your reasoning about what to do next. Explain your thinking."
    )
    action: str = Field(
        description="The action to take: tool name or 'FINISH' if task is complete"
    )
    action_input: dict = Field(
        description="Arguments for the action",
        default={}
    )

class ReActFinish(BaseModel):
    thought: str = Field(description="Final reasoning")
    answer: str = Field(description="The final answer to return to the user")
```

This structure ensures every step includes explicit reasoning. The agent can't skip the "thought" field—it's required.

## Implementation: System Prompt for ReAct

Your system prompt must teach the ReAct pattern:

```python
system_prompt = """You are a ReAct agent that solves problems step-by-step.

For each step, you must:

1. THINK: Reason about what you need to do next. Consider:
   - What information do you have?
   - What information is still missing?
   - What's the best next step?

2. ACT: Choose a tool to call (or FINISH if you're done). Available tools:
   - search_web: Search for information online
   - calculate: Perform mathematical calculations
   - get_data: Retrieve data from databases

3. OBSERVE: Review the tool result and plan the next step

Always explain your reasoning in the 'thought' field before acting.

When you have enough information to answer completely, use action='FINISH'."""
```

The prompt explicitly teaches the think-act-observe cycle. This is critical—without it, the model won't follow the pattern consistently.

## Implementation: The ReAct Agent

Here's a complete ReAct implementation:

```python
from openai import OpenAI
import json

client = OpenAI()

def run_react_agent(user_query: str, tools: list, max_iterations: int = 10) -> str:
    """Run a ReAct agent that reasons before acting."""

    messages = [
        {"role": "system", "content": system_prompt},
        {"role": "user", "content": user_query}
    ]

    for iteration in range(max_iterations):
        # Get agent's next thought and action
        response = client.chat.completions.create(
            model="gpt-4o-mini",
            messages=messages,
            tools=tools,
            tool_choice="auto"
        )

        message = response.choices[0].message
        messages.append(message)

        # Agent wants to use a tool
        if message.tool_calls:
            for tool_call in message.tool_calls:
                # The LLM includes its reasoning in the message content
                thought = message.content or "Proceeding with action..."
                print(f"\n💭 Thought: {thought}")
                print(f"🔧 Action: {tool_call.function.name}({tool_call.function.arguments})")

                # Execute the tool
                tool_result = execute_tool(
                    tool_call.function.name,
                    json.loads(tool_call.function.arguments)
                )

                print(f"👁️ Observation: {tool_result}")

                # Add observation to conversation
                messages.append({
                    "role": "tool",
                    "tool_call_id": tool_call.id,
                    "content": tool_result
                })

        # Agent is finished
        else:
            print(f"\n💭 Final Thought: {message.content}")
            return message.content

    return "Maximum iterations reached. Task may be incomplete."

def execute_tool(tool_name: str, tool_input: dict) -> str:
    """Execute the appropriate tool."""
    if tool_name == "search_web":
        return f"Search results for '{tool_input['query']}': [Mock results]"
    elif tool_name == "calculate":
        try:
            result = eval(tool_input["expression"])
            return f"Result: {result}"
        except Exception as e:
            return f"Error: {e}"
    else:
        return "Unknown tool"
```

Notice the explicit printing of thoughts, actions, and observations. This makes the agent's reasoning visible.

## Implementation: Example with Weather

Here's a concrete example:

```python
tools = [
    {
        "type": "function",
        "function": {
            "name": "get_weather",
            "description": "Get current weather for a city",
            "parameters": {
                "type": "object",
                "properties": {
                    "city": {"type": "string", "description": "City name"}
                },
                "required": ["city"]
            }
        }
    }
]

result = run_react_agent(
    "What's the weather like in Paris and London? Which city is warmer?",
    tools=tools
)

# Output:
# 💭 Thought: I need weather data for both cities before I can compare them.
# 🔧 Action: get_weather({"city": "Paris"})
# 👁️ Observation: Sunny, 22°C
#
# 💭 Thought: Now I need London's weather to make the comparison.
# 🔧 Action: get_weather({"city": "London"})
# 👁️ Observation: Cloudy, 15°C
#
# 💭 Final Thought: Paris is warmer at 22°C compared to London at 15°C.
```

The agent's reasoning is completely transparent.

## When to Use ReAct

Not every task needs ReAct. Use it strategically:

| Task Complexity | Use ReAct? | Why |
|----------------|-----------|-----|
| Single tool call | ❌ No | Overkill. Basic agent is faster and cheaper. |
| 2-3 sequential tool calls | ⚠️ Maybe | Basic agent usually works. Use ReAct if reliability is critical. |
| 4+ dependent tool calls | ✅ Yes | Planning significantly improves success rate. |
| Complex multi-step tasks | ✅ Yes | Essential for breaking down and solving systematically. |
| Tasks requiring judgment | ✅ Yes | Explicit reasoning helps with ambiguous situations. |
| User asks to "think step by step" | ✅ Yes | User explicitly wants to see reasoning. |

The rule of thumb: if you'd need to think carefully about the steps yourself, use ReAct.

## Running the Example

This lesson includes a complete ReAct implementation:

```bash
cd 12-planning-react
uv run example.py
```

Try these queries to see ReAct in action:
- "What's 15% of 847, then multiply that by 3?"
- "Search for Python tutorials and tell me the best one for beginners"
- "Find the weather in Tokyo, then tell me if I need an umbrella"

Watch the thought-action-observation cycle in action.

## Key Takeaways

1. **ReAct = Reasoning + Acting**: Force agents to think before acting. This simple pattern dramatically improves reliability.

2. **Explicit thoughts improve performance**: Making the LLM articulate its reasoning actually helps it reason better.

3. **Better for complex tasks**: Simple tasks don't benefit much. Multi-step, dependent tasks see 15-20% improvement.

4. **Cost-accuracy tradeoff**: ReAct uses more tokens (more API calls, longer prompts). Balance cost against reliability needs.

5. **Debugging becomes easy**: When agents fail, you can see exactly where their reasoning went wrong.

6. **Not always necessary**: For simple, single-tool tasks, basic agents are faster and cheaper.

## Common Pitfalls

1. **Using ReAct for simple tasks**: Don't add complexity where it's not needed. Single tool calls don't benefit from ReAct.

2. **Vague thought prompts**: "Think about this" is too weak. Be specific: "Explain what information you have and what you still need."

3. **Not showing observations**: The agent needs to see tool results in the conversation to reason about them.

4. **No max iterations**: ReAct agents can loop forever if they never reach FINISH. Always set a limit.

5. **Forgetting to validate**: Just because the agent explained its reasoning doesn't mean the reasoning is correct. Always validate outputs.

## Real-World Impact

Companies use ReAct for high-stakes tasks where reliability matters more than speed. Due diligence workflows, medical diagnosis assistance, financial analysis, and legal research all benefit from systematic reasoning.

The business value comes from improved accuracy on complex tasks, reduced errors that would be costly to fix, transparent reasoning that humans can audit, and easier debugging when things go wrong.

ReAct isn't faster or cheaper than basic agents—it's more reliable. Use it when mistakes are expensive.

## Assignment

Take one of your previous agents (FAQ agent or research assistant) and convert it to use ReAct. Add explicit thought steps to the system prompt, implement the ReAct loop with visible thoughts, and test it on 3-5 complex, multi-step questions.

Compare the ReAct version to your original:
- Does it make better decisions?
- Is it easier to debug when things go wrong?
- How much more expensive is it (token usage)?
- When would you use each approach?

## Next Steps

You've mastered the ReAct pattern for intelligent agent planning. Next, move to [Lesson 13 - FastAPI Deployment](../13-fastapi-deployment) to learn how to deploy your AI agents as production web services.

## Resources

- [ReAct: Synergizing Reasoning and Acting in Language Models](https://arxiv.org/abs/2210.03629) - Original research paper
- [Chain-of-Thought Prompting Elicits Reasoning in Large Language Models](https://arxiv.org/abs/2201.11903) - Related research
- [LangChain ReAct Documentation](https://python.langchain.com/docs/modules/agents/agent_types/react)
- [OpenAI Best Practices for Agents](https://platform.openai.com/docs/guides/function-calling)
