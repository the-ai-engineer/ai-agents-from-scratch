# Lesson 11: Build a Complete Research Assistant Agent

## What You'll Build

In this lesson, you're building a production-ready research assistant that combines multiple tools to handle complex, multi-step tasks. This agent can search multiple sources, perform calculations, analyze data, save notes, and generate comprehensive reports—all autonomously.

This is your capstone project. It brings together everything you've learned: API calls, prompt engineering, structured outputs, tool calling, the agent loop, memory management, and RAG. By the end, you'll have a fully functional agent that demonstrates professional-level AI engineering.

Research assistants are valuable in countless business contexts. They gather market intelligence, analyze competitor data, investigate technical problems, compile reports from multiple sources, and answer complex questions that require synthesizing information. This pattern applies to competitive analysis, due diligence, technical research, and strategic planning.

## The Problem

Complex research tasks don't fit into simple workflows. When you ask "analyze the top three fintech companies in London," you need to search for companies, filter by criteria, research each one individually, analyze competitive positioning, and compile findings into a coherent report.

Traditional tools can't adapt. Workflows break when requirements change. Static scripts can't handle the variability and judgment calls required for real research tasks.

The solution is an autonomous agent that understands the goal, breaks it down into steps, uses the right tools for each step, adapts based on what it finds, and delivers a comprehensive result—all without requiring step-by-step instructions.

## What You'll Learn

You'll master multi-tool coordination by giving your agent 4-5 specialized tools and teaching it to choose the right one for each situation. You'll implement tool chaining where the output of one tool becomes the input to another. You'll build stateful agents that persist notes and findings across the conversation.

You'll also learn production patterns like clear tool descriptions that help the LLM select correctly, error handling in each tool so failures don't crash the system, and structured outputs for complex analysis tasks.

Most importantly, you'll understand when to use agents versus workflows. Agents excel at open-ended tasks requiring judgment. Workflows work better for fixed, repeatable processes.

This is the culmination of your learning. Master this and you can build production AI agents for any domain.

## How Research Agents Work

A research assistant combines multiple capabilities to solve complex problems:

1. **Search**: Find information from various sources (web, databases, APIs)
2. **Analysis**: Process and understand the information retrieved
3. **Calculation**: Perform mathematical or logical operations on data
4. **Memory**: Save findings and notes for future reference
5. **Synthesis**: Combine everything into a coherent answer or report

The agent doesn't follow a fixed script. It reasons about what to do next based on:
- The user's goal
- What it's discovered so far
- What tools are available
- What information is still missing

This adaptive behavior is what makes agents powerful for research tasks.

## System Architecture

Your research assistant follows this architecture:

```
User Request
    ↓
Research Agent
    ├─→ Search Web (find companies, articles, data)
    ├─→ Search Database (query structured data)
    ├─→ Calculate (analyze numbers, compare metrics)
    ├─→ Save Notes (persist findings)
    └─→ Generate Summary (synthesize everything)
```

The agent decides which tools to use, in what order, and how to combine their outputs. It might search first, then calculate, then search again based on what it learned, then save notes, then generate a final report.

## Use Cases

Research assistants excel at specific types of tasks:

**Market Research**: "Find the top 5 SaaS companies in healthcare and compare their pricing models."

**Competitive Analysis**: "Who are Stripe's main competitors and how do their fees compare?"

**Data Investigation**: "Analyze our Q4 sales data and identify which products drove growth."

**Technical Research**: "What are the best practices for deploying ML models at scale? Give me code examples."

**Due Diligence**: "Research this startup—funding history, key customers, team background."

These tasks require multiple steps, judgment calls, and synthesizing information from different sources. Perfect for agents.

## Implementation: Defining Tools

A research assistant needs multiple specialized tools:

```python
from pydantic import BaseModel, Field
from typing import Literal

class WebSearchInput(BaseModel):
    query: str = Field(description="Search query for finding information online")
    num_results: int = Field(default=5, description="Number of results to return")

def search_web(query: str, num_results: int = 5) -> str:
    """Search the web for information."""
    # In production, use actual search API (Google, Bing, etc.)
    # For demo, return mock results
    return f"Search results for '{query}': [Mock results 1-{num_results}]"

class CalculateInput(BaseModel):
    expression: str = Field(description="Mathematical expression to evaluate")

def calculate(expression: str) -> str:
    """Perform calculations on numerical data."""
    try:
        result = eval(expression)  # Use safe eval in production
        return f"Result: {result}"
    except Exception as e:
        return f"Calculation error: {str(e)}"

class SaveNoteInput(BaseModel):
    title: str = Field(description="Title or key for the note")
    content: str = Field(description="The note content to save")

# Persistent notes storage
notes_db = {}

def save_note(title: str, content: str) -> str:
    """Save a note for future reference."""
    notes_db[title] = content
    return f"Note '{title}' saved successfully."

class GetNotesInput(BaseModel):
    pass  # No parameters needed

def get_notes() -> str:
    """Retrieve all saved notes."""
    if not notes_db:
        return "No notes saved yet."

    formatted_notes = []
    for title, content in notes_db.items():
        formatted_notes.append(f"**{title}**:\n{content}")

    return "\n\n".join(formatted_notes)
```

Each tool has a single, clear purpose. The agent combines them to accomplish complex tasks.

## Implementation: Tool Definitions

Define the tools for the OpenAI API:

```python
tools = [
    {
        "type": "function",
        "function": {
            "name": "search_web",
            "description": "Search the web for information. Use this when you need to find facts, companies, articles, or general information online.",
            "parameters": WebSearchInput.model_json_schema()
        }
    },
    {
        "type": "function",
        "function": {
            "name": "calculate",
            "description": "Perform mathematical calculations. Use this for arithmetic, percentages, comparisons, or any numerical analysis.",
            "parameters": CalculateInput.model_json_schema()
        }
    },
    {
        "type": "function",
        "function": {
            "name": "save_note",
            "description": "Save important findings or notes for later reference. Use this to remember key information as you research.",
            "parameters": SaveNoteInput.model_json_schema()
        }
    },
    {
        "type": "function",
        "function": {
            "name": "get_notes",
            "description": "Retrieve all saved notes. Use this when you need to review what you've learned so far.",
            "parameters": GetNotesInput.model_json_schema()
        }
    }
]
```

Clear descriptions are critical. The LLM uses them to decide which tool to call.

## Implementation: Research Agent

Create the agent with a system prompt optimized for research:

```python
from openai import OpenAI
import os

client = OpenAI(api_key=os.getenv("OPENAI_API_KEY"))

system_prompt = """You are a professional research assistant.

Your job is to help users with complex research tasks by:
1. Breaking down questions into steps
2. Searching for information when needed
3. Saving important findings as you go
4. Performing calculations and analysis
5. Synthesizing everything into clear answers

Always:
- Use tools when you need information or need to calculate something
- Save key findings using save_note so you don't lose information
- Be thorough but concise
- Cite sources when possible
- Admit when you don't have information

Work step-by-step and show your reasoning."""

def execute_tool(tool_name: str, tool_input: dict) -> str:
    """Execute the appropriate tool based on the name."""
    if tool_name == "search_web":
        return search_web(tool_input["query"], tool_input.get("num_results", 5))
    elif tool_name == "calculate":
        return calculate(tool_input["expression"])
    elif tool_name == "save_note":
        return save_note(tool_input["title"], tool_input["content"])
    elif tool_name == "get_notes":
        return get_notes()
    else:
        return f"Unknown tool: {tool_name}"

def run_research_agent(user_message: str, max_iterations: int = 10) -> str:
    """Run the research agent with the user's query."""
    messages = [
        {"role": "system", "content": system_prompt},
        {"role": "user", "content": user_message}
    ]

    for iteration in range(max_iterations):
        # Get agent's next action
        response = client.chat.completions.create(
            model="gpt-4o-mini",
            messages=messages,
            tools=tools
        )

        message = response.choices[0].message
        messages.append(message)

        # Check if agent wants to call tools
        if message.tool_calls:
            for tool_call in message.tool_calls:
                # Execute the tool
                tool_result = execute_tool(
                    tool_call.function.name,
                    eval(tool_call.function.arguments)
                )

                # Add tool result to conversation
                messages.append({
                    "role": "tool",
                    "tool_call_id": tool_call.id,
                    "content": tool_result
                })
        else:
            # Agent is done, return final response
            return message.content

    return "Max iterations reached. The task may require more steps."
```

The agent loops until it's satisfied it has answered the question completely.

## Implementation: Putting It All Together

Now use your research assistant:

```python
# Example research task
query = """Research the top 3 cloud providers (AWS, Azure, GCP) and compare:
1. Their market share
2. Starting prices for basic compute
3. One unique feature of each

Save your findings and give me a summary."""

result = run_research_agent(query)
print(result)

# The agent will:
# 1. Search for cloud provider market share
# 2. Save that finding
# 3. Search for pricing information
# 4. Save pricing data
# 5. Search for unique features
# 6. Save feature information
# 7. Review all saved notes
# 8. Generate a comprehensive summary
```

The agent autonomously breaks down the task, gathers information, organizes findings, and synthesizes a final answer.

## Running the Example

This lesson includes a complete working implementation:

```bash
cd 11-example-research-assistant
uv run example.py
```

Try these research tasks:
- "Compare Python vs JavaScript for web development"
- "Find the top 3 AI companies by funding and compare their focus areas"
- "Calculate the compound annual growth rate if I invest $1000 at 8% for 10 years"

Watch how the agent uses multiple tools, saves notes, and builds up to a comprehensive answer.

## Key Takeaways

1. **Multiple tools = more capability**: Give your agent 4-5 well-designed tools. More than that and tool selection becomes unreliable. Fewer and it's too limited.

2. **Tool chaining is powerful**: The agent can use the output of one tool as input to another. This creates complex workflows dynamically.

3. **Clear tool descriptions matter**: The LLM relies entirely on your descriptions to choose the right tool. Be specific about when to use each one.

4. **Each tool handles its own errors**: Don't let errors crash the whole system. Return error messages as strings so the agent can adapt.

5. **State management is critical**: The notes database persists findings across tool calls. Without it, the agent forgets what it learned.

6. **Limit iterations**: Always set max_iterations to prevent infinite loops. 10-15 is usually sufficient for complex tasks.

## Common Pitfalls

1. **Too many tools**: More than 6-7 tools and the agent struggles to choose correctly. Keep your tool set focused.

2. **Vague tool descriptions**: "Searches for stuff" is useless. "Search the web for companies, articles, or facts when you need external information" is clear.

3. **Not persisting state**: Without notes or memory, the agent can't build on previous findings. Multi-step tasks fail.

4. **No max iterations**: Agents can loop forever. Always set a maximum to prevent runaway costs.

5. **Forgetting error handling**: Tools will fail. Network issues, rate limits, bad inputs. Return helpful error messages instead of crashing.

6. **Not testing edge cases**: What if search returns no results? What if calculation syntax is invalid? Test failure modes.

## Real-World Impact

Companies use research assistants to dramatically accelerate information gathering. Tasks that took analysts hours or days now take minutes. Due diligence that required reading dozens of documents is automated. Competitive research that needed multiple team members can be done by one person with an AI assistant.

The business impact is measurable: 10x faster research cycles, consistent quality across all investigations, junior team members can do work that previously required senior expertise, and teams focus on decision-making instead of information gathering.

Research assistants don't replace human judgment—they amplify it by handling the tedious work of finding and organizing information.

## Assignment

Build a research assistant for a domain you care about. Choose 4-5 tools relevant to your domain (could be APIs, databases, calculation tools, or specialized search). Design clear tool descriptions, implement the agent loop with max iterations, and test it on 3-5 complex, multi-step questions.

Pay attention to:
- Does it choose the right tools?
- Does it chain tools together effectively?
- Does it save and recall important information?
- Is the final output comprehensive and accurate?
- How many iterations does it typically take?

## Next Steps

Congratulations! You've built a complete, production-ready AI agent. Next, move to [Lesson 12 - Planning with ReAct](../12-planning-react) to learn how to make your agents reason more systematically using the ReAct pattern.

## Resources

- [OpenAI Function Calling Best Practices](https://platform.openai.com/docs/guides/function-calling)
- [Building Reliable AI Agents](https://www.anthropic.com/index/building-reliable-agents)
- [ReAct: Synergizing Reasoning and Acting](https://arxiv.org/abs/2210.03629)
