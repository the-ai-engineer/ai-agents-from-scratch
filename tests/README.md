# Tests for Core Primitives

This directory contains tests for the core agent and tool primitives in the `src/` folder. Tests are designed to be both educational and functional - they demonstrate how the components work while ensuring correctness.

## Philosophy: Keep Tests Simple

These tests follow a **"simple but sufficient"** approach:
- **14 focused tests** covering all critical behaviors
- **No unnecessary granularity** - tests demonstrate complete workflows
- **Educational** - each test shows realistic usage patterns
- **Fast** - runs in ~0.3 seconds with mocking

## Test Structure

```
tests/
├── __init__.py           # Makes tests a package
├── test_tool.py          # 3 tests for Tool class
├── test_agent.py         # 11 tests for Agent and ConversationMemory
└── README.md             # This file
```

## Running Tests

### Run all tests

```bash
uv run pytest tests/
```

### Run tests with verbose output

```bash
uv run pytest tests/ -v
```

### Run tests for a specific module

```bash
# Test only the Tool class
uv run pytest tests/test_tool.py -v

# Test only the Agent class
uv run pytest tests/test_agent.py -v
```

### Run a specific test

```bash
uv run pytest tests/test_tool.py::TestToolCreation::test_create_tool_from_simple_function -v
```

### Run tests with coverage

```bash
uv run pytest tests/ --cov=src --cov-report=term-missing
```

### Run tests with detailed output

```bash
# Show print statements
uv run pytest tests/ -v -s

# Show local variables on failure
uv run pytest tests/ -v -l
```

## Test Organization

### test_tool.py (3 tests)

Tests for the `Tool` class that converts Python functions to OpenAI tool schemas:

1. **test_tool_creation_basics**: Creating tools from functions with required/optional parameters
2. **test_tool_to_openai_format**: Converting tools to OpenAI API format
3. **test_tool_edge_cases**: Handling no parameters and missing docstrings

### test_agent.py (11 tests)

Tests for the `Agent` and `ConversationMemory` classes:

**TestConversationMemory** (3 tests)
- Memory initialization and message management
- Realistic conversation flow with all message types
- Clear behavior (preserves system prompt)

**TestAgentBasics** (3 tests)
- Agent initialization with default and custom configs
- Tool registration (single and multiple)
- Reset behavior (clears conversation, keeps tools)

**TestToolExecution** (2 tests)
- Successful tool execution
- Error handling (tool not found, malformed JSON)

**TestAgentLoop** (3 tests)
- Simple chat without tools
- Chat with tool call (full agent loop)
- Max iterations safety (prevents infinite loops)

## Testing Philosophy

### Simple but Sufficient

Tests are designed to be **educational without being overwhelming**:

- **Focused on core behaviors** - not every edge case
- **Realistic examples** - demonstrate how components actually work together
- **Clear test names** - you know what's being tested without reading code
- **Comprehensive enough** - 91% code coverage with just 14 tests

### Fast and Deterministic

- Tests use mocking to avoid real API calls
- No network dependencies
- Complete test suite runs in ~0.3 seconds
- Deterministic results (no flaky tests)

### What's Covered

- ✓ Tool creation from functions
- ✓ Required vs optional parameters
- ✓ OpenAI schema conversion
- ✓ Conversation memory management
- ✓ Agent initialization and configuration
- ✓ Tool registration and execution
- ✓ Agent loop behavior
- ✓ Error handling
- ✓ Max iterations safety

## Adding New Tests

When adding new functionality to `src/`, consider if you **really need a new test**:

### Before Adding a Test, Ask:

1. **Is this behavior already covered?** - Check existing tests first
2. **Is this too granular?** - Can it be combined with an existing test?
3. **Does this teach something valuable?** - If not, skip it
4. **Is this an obvious implementation detail?** - Probably don't test it

### When You Do Add a Test

```python
def test_my_new_feature():
    """
    Brief description of what this tests and why it matters.

    Tests should demonstrate realistic usage, not just exercise code.
    """
    # Setup
    agent = Agent()

    # Execute
    result = agent.do_something()

    # Verify
    assert result == expected_value
```

### Keep It Simple

- Prefer **fewer, more comprehensive tests** over many granular ones
- **Combine related assertions** in a single test
- **Use clear variable names** instead of excessive comments
- **Mock external calls** but keep test logic straightforward

## Troubleshooting

### Import Errors

If you see import errors, make sure you're running tests from the project root:

```bash
# From project root
uv run pytest tests/

# Not from tests directory
cd tests/
uv run pytest .  # This may fail with import errors
```

### Missing Dependencies

Install pytest if not already present:

```bash
uv add --dev pytest pytest-cov
```

### Tests Failing

If tests fail:

1. **Read the error message carefully** - pytest provides detailed output
2. **Check the test docstring** - understand what the test expects
3. **Run with -v flag** for verbose output
4. **Add print statements** and run with -s flag to see them
5. **Use pytest --pdb** to drop into debugger on failure

## Continuous Integration

These tests are designed to run in CI/CD pipelines:

```yaml
# Example GitHub Actions workflow
- name: Run tests
  run: |
    uv sync
    uv run pytest tests/ -v --cov=src
```

## Learning Resources

### Understanding the Tests

Start by reading tests in this order:

1. **test_tool.py::test_tool_creation_basics** - Simplest test showing tool creation
2. **test_agent.py::TestConversationMemory** - How memory works (3 tests)
3. **test_agent.py::TestAgentBasics** - Agent setup and tool registration (3 tests)
4. **test_agent.py::TestAgentLoop** - Full agent loop behavior (3 tests)

### pytest Documentation

- [pytest Getting Started](https://docs.pytest.org/en/stable/getting-started.html)
- [pytest Fixtures](https://docs.pytest.org/en/stable/fixture.html)
- [pytest Mocking](https://docs.pytest.org/en/stable/how-to/monkeypatch.html)

## Questions?

If you're stuck or have questions about the tests:

1. Read the test docstrings - they explain what's being tested
2. Look at the fixtures in conftest.py - they show realistic usage
3. Run tests with -v flag to see detailed output
4. Check the implementation in src/ to understand what's being tested
