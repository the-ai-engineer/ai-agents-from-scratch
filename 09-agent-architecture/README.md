# Building an AI Agent from Scratch

## What You'll Learn

Build a complete autonomous AI agent from scratch in this hands-on tutorial.

By the end, you'll understand:
- The agent loop pattern that powers all autonomous agents
- How to build a reusable Agent class
- Why conversation history enables multi-step reasoning
- How agents make autonomous decisions

This is **the** foundational pattern. Every advanced agent system—from ChatGPT to coding assistants—uses this core concept.

## The Core Concept

> **"Agents are models using tools in a loop"**

That's it. This simple pattern is the foundation of agent autonomy.

## The Problem

Basic tool calling only works for single-step tasks. Ask "What's the weather in Paris and what time is it there?" and you're stuck—you can only call ONE tool per request.

Real tasks need multiple steps:
- Gathering information from multiple sources
- Using one result as input to another tool
- Making decisions based on what you learn
- Combining information into a final answer

You need **autonomous agents** that can work through problems step-by-step.

## The Solution: The Agent Loop

Instead of calling the LLM once, **loop until you have a final answer**:

```python
messages = [{"role": "user", "content": "What's the weather in Paris?"}]

for iteration in range(max_iterations):
    # 1. Call LLM with tools
    response = client.responses.create(
        model="gpt-4o-mini",
        input=messages,
        tools=tool_schemas
    )

    # 2. Process response
    for item in response.output:
        if item.type == "message":
            # Got final answer!
            return item.content[0].text

        elif item.type == "function_call":
            # LLM wants to use a tool
            # Add function call to history
            messages.append({
                "type": "function_call",
                "call_id": item.call_id,
                "name": item.name,
                "arguments": item.arguments,
            })

            # Execute tool
            result = execute_tool(item)

            # Add result to history
            messages.append({
                "type": "function_call_output",
                "call_id": item.call_id,
                "output": result,
            })
            # Loop continues...
```

**The magic:** After each tool execution, you add the results to conversation history. When the LLM sees the updated history, it knows what information it has and can decide the next step.

The LLM orchestrates everything—you just provide tools and run the loop.

## The Tutorial

`tutorial.py` walks you through building a complete Agent class from scratch:

**What You'll Build:**

```python
# Your Agent class
agent = Agent()
agent.add_tool(get_weather)
agent.add_tool(get_time)

# Use it - the loop happens automatically!
response = agent.run("What's the weather and time in Tokyo?")
```

**The tutorial covers:**
1. **Tool registration** - How to let agents use your functions
2. **Tool schemas** - Converting Python functions to OpenAI format
3. **The agent loop** - The core pattern in detail
4. **Conversation history** - Why it enables multi-step reasoning
5. **Error handling** - Graceful failures
6. **Four examples** - From simple to complex

**You'll build:**
- Complete Agent class (~180 lines)
- Tool registry system
- Conversation management
- Autonomous decision making

## Running the Tutorial

```bash
uv run python 09-agent-architecture/tutorial.py
```

You'll see 4 examples:
1. **Single tool** - Basic agent usage
2. **Multiple tools** - Calling several tools in one query
3. **Conversation** - Multi-turn dialogue with context
4. **Tool chaining** - Agent orchestrating multiple tool calls

## Key Concepts

### 1. The Agent Loop

```
User message → Call LLM → Tool calls?
                          ↓ Yes
                Execute tools → Add to history → Loop back
                          ↓ No
                Return answer ✓
```

### 2. Conversation History is Critical

Every interaction is stored:
- User messages
- Assistant responses
- **Function calls themselves**
- Function call outputs

The LLM sees the full history and decides what to do next based on what it already knows.

### 3. Function Call History Format

**CRITICAL:** You must add BOTH the function call AND its output:

```python
# Step 1: Add function call
messages.append({
    "type": "function_call",
    "call_id": item.call_id,
    "name": item.name,
    "arguments": item.arguments,
})

# Step 2: Execute function
result = execute_tool(item)

# Step 3: Add output
messages.append({
    "type": "function_call_output",
    "call_id": item.call_id,
    "output": result,
})
```

Skip step 1 and you get a 400 error from OpenAI.

### 4. Max Iterations Prevents Infinite Loops

Always set a limit (typically 5-10 iterations). If the agent can't solve the problem in N steps, fail gracefully.

### 5. The LLM Orchestrates

You don't write if-then logic. The LLM:
- Decides which tools to call
- Determines when it has enough information
- Chooses when to give a final answer

You just provide tools and run the loop.

## From Tutorial to Production

After completing `tutorial.py`, compare your Agent with `src/agent_sync.py`:

```bash
# Your tutorial agent
less 09-agent-architecture/tutorial.py

# Production agent
less src/agent_sync.py
```

You'll see they're nearly identical! The main differences:
- `AgentSync` has type hints and documentation
- Handles structured outputs (Pydantic models)
- More robust error handling
- Better tool schema generation

**The core loop is the same.** You just built a production-ready agent pattern!

## Common Pitfalls

1. **Forgetting max iterations** - Always set a limit
2. **Not adding function calls to history** - Must add call BEFORE output
3. **Not adding results to history** - LLM can't see tool results
4. **Using wrong message format** - Responses API needs specific structure
5. **Ignoring errors** - Always handle tool failures gracefully

## Assignment

Extend your agent with new capabilities:

1. **Add a calculation tool:**
   ```python
   def calculate(expression: str) -> str:
       """Evaluate a math expression."""
       return str(eval(expression))
   ```

2. **Add a search tool:**
   ```python
   def search(query: str) -> str:
       """Search for information."""
       return f"Search results for: {query}"
   ```

3. **Test multi-step queries:**
   - "What's the weather in Paris and what is 25 * 48?"
   - "Search for Python programming and tell me what time it is in Tokyo"
   - "Check weather in 3 cities and tell me which is warmest"

4. **Observe:**
   - How many iterations each query takes
   - Which tools get called
   - How the agent chains tools together

## Key Takeaways

1. **Agents are models using tools in a loop** - This is the entire pattern
2. **Conversation history enables autonomy** - Past results inform future decisions
3. **The LLM orchestrates** - You provide tools, LLM decides when to use them
4. **Max iterations are essential** - Prevent infinite loops
5. **Function call history is critical** - Must be added before outputs
6. **Simple API, complex behavior** - Hide loop complexity behind clean interface
7. **Production ready** - This pattern scales to real systems

## Next Steps

You've built an autonomous AI agent from scratch!

**Move to Lesson 10: Advanced Memory** to learn:
- Handling long conversations and token limits
- Automatic history trimming
- Conversation persistence
- Production memory management

## Resources

- [Anthropic: Building Effective Agents](https://www.anthropic.com/news/building-effective-agents) - Best practices for agent systems
- [OpenAI Responses API](https://platform.openai.com/docs/api-reference/responses) - Official API documentation
- [LLM Agents Survey](https://arxiv.org/abs/2309.07864) - Academic overview of agent architectures
