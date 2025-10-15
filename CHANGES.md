# Changes Summary

## Updated to OpenAI Responses API

### What Changed

**Old:** Chat Completions API (`client.chat.completions.create()`)
**New:** Responses API (`client.responses.create()`)

### Key API Differences

| Aspect | Chat Completions | Responses API |
|--------|-----------------|---------------|
| Method call | `client.chat.completions.create()` | `client.responses.create()` |
| Input parameter | `messages=` | `input=` |
| Response structure | `response.choices[0].message` | `response.output` (array of items) |
| Tool schema | Nested with `function` key | Flat structure |
| Tool call format | `message.tool_calls` | `item.type == "function_call"` |
| Function results | `role="tool"` | `role="function_call_output"` |

### Tool Schema Format

The Responses API uses a **flat tool schema** format (not nested):

```python
# Responses API format (current)
{
    "type": "function",
    "name": "get_weather",
    "description": "Get weather for a city",
    "parameters": {
        "type": "object",
        "properties": {
            "city": {"type": "string"}
        },
        "required": ["city"]
    }
}

# OLD Chat Completions format (deprecated)
{
    "type": "function",
    "function": {  # <- Extra nesting, not used in Responses API
        "name": "get_weather",
        ...
    }
}
```

### Response Output Processing

Responses API returns an array of output items, not choices:

```python
# Old way (Chat Completions)
message = response.choices[0].message
if message.tool_calls:
    # Process tool calls

# New way (Responses API)
for item in response.output:
    if item.type == "message":
        final_text = item.content[0].text
    elif item.type == "function_call":
        # Process function call
        result = self._call_tool(item)
        # Note: Uses "type" not "role", and "output" not "content"
        self.messages.append({
            "type": "function_call_output",
            "call_id": item.call_id,
            "output": result,
        })
```

### Files Updated

1. **src/tool.py** - `to_openai_format()` returns flat structure (no nested "function" key)
2. **src/agent_sync.py** - Updated `_agent_loop()` to use Responses API:
   - Changed `messages=` to `input=`
   - Process `response.output` array instead of `response.choices[0].message`
   - Handle `item.type == "message"` and `item.type == "function_call"`
   - Use `role="function_call_output"` with `call_id` for tool results
3. **src/agent.py** - Same changes as sync version for async agent
4. **tests/test_tool.py** - Tests expect flat format
5. **tests/test_agent.py** - Tests work with both formats (internal mocking)

### Testing

```bash
# Run unit tests (all pass)
pytest tests/test_agent.py tests/test_tool.py -v
# 22 passed ✅

# Run integration tests (requires valid OPENAI_API_KEY)
pytest tests/ -m integration -v
# Requires valid API key in .env file
```

### API Compatibility

- ✅ Responses API (current)
- ❌ Chat Completions API (deprecated - use old nested format)

The course now exclusively uses the Responses API throughout.

### Migration Complete

All agent code (`Agent` and `AgentSync`) now uses:
- `client.responses.create()` with `input=` parameter
- Flat tool schemas
- Output item processing with `item.type` checks
- Proper `call_id` correlation for function calls and results

### Function Call History Requirement

**CRITICAL:** The Responses API requires that function calls be added to conversation history **before** their outputs:

```python
# Step 1: Add the function call itself
self.messages.append({
    "type": "function_call",
    "call_id": item.call_id,
    "name": item.name,
    "arguments": item.arguments,
})

# Step 2: Execute the function
result = self._call_tool(item)

# Step 3: Add the function call output
self.messages.append({
    "type": "function_call_output",
    "call_id": item.call_id,
    "output": result,
})
```

Without step 1, the API returns error:
```
Error code: 400 - No tool call found for function call output with call_id <id>
```

### Tool Call Structure Update

The `_call_tool()` method was updated to match Responses API structure:

```python
# Old way (Chat Completions API)
name = call.function.name
args = json.loads(call.function.arguments or "{}")

# New way (Responses API)
name = call.name  # Direct attribute, not nested
args = json.loads(call.arguments or "{}")
```

The Responses API returns `ResponseFunctionToolCall` objects with a flat structure:
- `call.name` (not `call.function.name`)
- `call.arguments` (not `call.function.arguments`)
- `call.call_id` for correlation

This required updating both:
- `src/agent_sync.py` - `_call_tool()` method
- `src/agent.py` - async `_call_tool()` method
- `tests/test_agent.py` - All mock tool calls updated to flat structure

### Function Call Output Format

Function call outputs in the Responses API use a different structure:

```python
# OLD format (Chat Completions)
{
    "role": "tool",
    "tool_call_id": call.id,
    "content": result
}

# NEW format (Responses API)
{
    "type": "function_call_output",  # NOT "role"
    "call_id": item.call_id,         # Different key name
    "output": result                 # NOT "content"
}
```

The Responses API only accepts these roles: `'assistant'`, `'system'`, `'developer'`, and `'user'`.
Function call outputs must use the `"type"` field instead.
