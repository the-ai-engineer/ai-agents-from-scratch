# Lesson 09 Simplification

## What Changed

Simplified Lesson 09 from two files to **one clear tutorial** that builds an Agent class from scratch.

## Before
- ❌ Two files: `loop.py` (raw loop) + `class.py` (abstraction)
- ❌ ~500 lines total
- ❌ Students had to understand raw implementation first, then abstraction
- ❌ Duplicated code between files

## After
- ✅ **One file: `tutorial.py`** - Build an Agent class step-by-step
- ✅ ~250 lines of clear, documented code
- ✅ Single learning path - build as you learn
- ✅ Four progressive examples showing capabilities

## Structure

### tutorial.py (~250 lines)

**Step 1: Define Tools**
- Simple Python functions (get_weather, get_time)
- No decorators or complexity

**Step 2: Build the Agent Class**
- `__init__()` - Initialize client, messages, tools
- `add_tool()` - Register functions
- `_create_tool_schema()` - Convert functions to OpenAI format
- `run()` - The agent loop (core concept)
- `_execute_tool()` - Safe tool execution
- `reset()` - Clear conversation

**Step 3: Four Examples**
1. Single tool usage
2. Multiple tools in one query
3. Conversation with memory
4. Multi-step tool chaining

**Summary section** explaining what students just built

## Key Teaching Points

1. **"Agents are models using tools in a loop"** - Core concept
2. **Build as you learn** - No theory first, just implementation
3. **Progressive complexity** - Examples get gradually more advanced
4. **Compare to production** - Tutorial agent → AgentSync (nearly identical!)
5. **Function call history is critical** - Emphasized with examples

## Benefits

- **Simpler**: One file instead of two
- **Clearer**: Build incrementally, not raw→abstraction
- **Shorter**: Half the code to read
- **More practical**: Focus on building, not comparing approaches
- **Better aligned**: Tutorial agent matches AgentSync closely

## Learning Flow

1. **Read README.md** - Understand why agents need loops
2. **Read tutorial.py** - Build Agent class step-by-step
3. **Run tutorial.py** - See 4 examples in action
4. **Compare to src/agent_sync.py** - See how it maps to production

Simple, clear, effective.
