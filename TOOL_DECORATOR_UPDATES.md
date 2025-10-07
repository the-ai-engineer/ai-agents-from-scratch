# Tool Decorator Pattern Updates

## Summary

Updated lessons 05 and 06 to use the `@tool` decorator pattern from `code_samples.py` for cleaner, more maintainable tool definitions.

## Changes Made

### Lesson 05: Tool Calling Basics (`05-tool-calling-basics/`)

#### `example.py` Changes:
1. **Added @tool decorator implementation** (lines 19-93):
   - `extract_function_params()`: Extracts parameter metadata from function signatures
   - `create_json_schema()`: Generates JSON schema from function signature
   - `Tool` class: Pydantic model for tool metadata
   - `tool()` decorator: Main decorator that attaches tool metadata to functions

2. **Restructured examples** to show progression:
   - **Example 1**: `basic_tool_call_manual()` - Traditional manual JSON schema approach (kept for learning)
   - **Example 2**: `basic_tool_call_decorator()` - Single tool with @tool decorator
   - **Example 3**: `multiple_tools_decorator()` - Multiple tools with tool registry pattern

3. **Key improvements**:
   - Automatic schema generation from function signatures
   - Type hints drive parameter types
   - Docstrings become tool descriptions
   - Tool registry pattern for clean function lookup

#### `README.md` Changes:
1. Updated introduction to mention both manual and decorator approaches
2. Updated "Running the Example" section to describe three new examples
3. Enhanced "Key Takeaways" to emphasize @tool decorator benefits
4. Added tool registry pattern to key concepts

### Lesson 06: Tool Calling with Pydantic (`06-tool-calling-pydantic/`)

#### `example.py` Changes:
1. **Complete refactor** to use @tool decorator exclusively:
   - Removed manual Pydantic BaseModel classes (`GetWeatherArgs`, `SendEmailArgs`)
   - Removed custom Tool wrapper class
   - Added same @tool decorator implementation as lesson 05

2. **New examples**:
   - **Example 1**: `basic_tool_decorator()` - Single tool with automatic schema generation
   - **Example 2**: `multiple_tools_with_defaults()` - Multiple tools with optional parameters
   - **Example 3**: `tool_with_validation()` - Type validation and error handling

3. **Benefits**:
   - Simpler: No separate argument classes needed
   - Cleaner: Function signature is the source of truth
   - Automatic: Optional parameters detected from defaults
   - Consistent: Same pattern as lesson 05

#### `README.md` Changes:
1. Updated title to "Production-Ready Tool Calling with the @tool Decorator"
2. Completely rewrote "The Solution" section to focus on @tool decorator
3. Updated all examples to show @tool decorator usage
4. Revised "Key Takeaways" to emphasize decorator benefits
5. Updated "Common Pitfalls" with decorator-specific issues
6. Updated "Running the Example" section

## Benefits of @tool Decorator Pattern

### Before (Manual Schema):
```python
# Verbose, error-prone, manual synchronization
tool = {
    "type": "function",
    "function": {
        "name": "get_weather",
        "description": "Get weather",
        "parameters": {
            "type": "object",
            "properties": {
                "city": {"type": "string"}
            },
            "required": ["city"]
        }
    }
}
```

### After (@tool Decorator):
```python
# Clean, automatic, single source of truth
@tool
def get_weather(city: str) -> str:
    """Get the current weather for a given city.

    Args:
        city: City name, e.g., 'Paris', 'London', 'Tokyo'
    """
    return f"Weather in {city}"

# Schema generated automatically!
tools = [get_weather.tool.to_openai_format()]
```

## Key Features

1. **Automatic Schema Generation**: Function signature + docstring → OpenAI schema
2. **Type Safety**: Type hints define parameter types
3. **Optional Parameters**: Default values automatically make parameters optional
4. **Tool Registry Pattern**: Clean dictionary-based function lookup
5. **Maintainability**: Change function signature, schema updates automatically
6. **Less Boilerplate**: No separate Pydantic models or manual JSON schemas

## Migration Guide for Other Lessons

To adopt this pattern in other lessons:

1. Copy the @tool decorator code (lines 19-93 from lesson 05)
2. Replace manual tool schemas with decorated functions
3. Use tool registry pattern: `tool_registry = {"name": func}`
4. Generate schemas: `tools = [func.tool.to_openai_format() for func in registry.values()]`
5. Execute tools: `result = tool_registry[function_name](**args)`

## Testing

Created `test_tool_decorator.py` to verify:
- Decorator attaches `.tool` attribute correctly
- Schema generation includes all parameters
- Required vs optional parameters detected correctly
- Docstrings become tool descriptions
- Type hints map to JSON schema types

Test passed successfully with proper schema generation for both required and optional parameters.

## Next Steps

Consider applying this pattern to:
- Lesson 07: Workflow Patterns (if using tools)
- Lesson 08: Agent Loop (standardize tool handling)
- Lesson 09: Agent Class (integrate decorator pattern)
- Lessons 10-14: Update any tool definitions to use decorator

This creates a consistent tool definition pattern across the entire tutorial series.
