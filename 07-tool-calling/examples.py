"""
Lesson 07: Tool Calling Examples

Learn how to give AI the ability to call functions and take actions.
This lesson shows the progression from manual JSON to automated schema generation.

Run from project root:
    uv run python 07-tool-calling/examples.py
"""

import json
import inspect
from dataclasses import dataclass
from openai import OpenAI
from dotenv import load_dotenv

load_dotenv()
client = OpenAI()


##=================================================##
## Example 1: Manual JSON tool definition
##=================================================##

def get_weather(city: str) -> str:
    """Get the current weather for a given city."""
    weather_data = {
        "paris": "Sunny, 22°C",
        "london": "Cloudy, 15°C",
        "tokyo": "Rainy, 18°C",
    }
    return weather_data.get(city.lower(), f"Weather data not available for {city}")


# Manual JSON schema definition (Responses API flat format)
# This is what OpenAI expects - you're writing the schema by hand
tools = [{
    "type": "function",
    "name": "get_weather",
    "description": "Get the current weather for a given city",
    "parameters": {
        "type": "object",
        "properties": {
            "city": {
                "type": "string",
                "description": "The city name, e.g., Paris, London"
            }
        },
        "required": ["city"]
    }
}]

user_message = "What's the weather like in Paris?"

# Step 1: LLM decides to call tool
response = client.responses.create(
    model="gpt-4o-mini",
    input=[{"role": "user", "content": user_message}],
    tools=tools
)

# Step 2: Execute the tool
for output in response.output:
    if output.type == "function_call":
        args = json.loads(output.arguments)
        result = get_weather(**args)

        # Step 3: Return result to LLM for final response
        input_with_result = [
            {"role": "user", "content": user_message},
            {"type": "function_call_output", "call_id": output.call_id, "output": result}
        ]

        final_response = client.responses.create(
            model="gpt-4o-mini",
            input=input_with_result,
            tools=tools
        )

        # final_response.output_text


##=================================================##
## Example 2: Using a dataclass to reduce boilerplate
##=================================================##

@dataclass
class Tool:
    """Simple dataclass to convert functions to OpenAI tool schemas."""
    name: str
    description: str
    parameters: dict

    def to_dict(self) -> dict:
        """Convert to OpenAI Responses API format."""
        return {
            "type": "function",
            "name": self.name,
            "description": self.description,
            "parameters": self.parameters,
        }

    @classmethod
    def from_function(cls, func) -> "Tool":
        """
        Create a Tool from a Python function.

        Automatically extracts name, docstring, and parameters.
        This saves you from writing the JSON by hand.
        """
        sig = inspect.signature(func)

        properties = {}
        required = []

        for name, param in sig.parameters.items():
            properties[name] = {"type": "string"}
            if param.default == inspect.Parameter.empty:
                required.append(name)

        return cls(
            name=func.__name__,
            description=inspect.getdoc(func) or "",
            parameters={
                "type": "object",
                "properties": properties,
                "required": required,
            },
        )


def list_files(directory: str = ".") -> str:
    """List all files in a directory.

    Args:
        directory: Path to directory (default: current directory)
    """
    try:
        import os
        files = os.listdir(directory)
        if not files:
            return f"Directory '{directory}' is empty"
        return f"Files in '{directory}': {', '.join(files)}"
    except FileNotFoundError:
        return f"Error: Directory '{directory}' not found"
    except Exception as e:
        return f"Error: {str(e)}"


# Create tool schema using the dataclass (much cleaner!)
list_files_tool = Tool.from_function(list_files)
tools = [list_files_tool.to_dict()]

user_message = "What files are in the current directory?"

response = client.responses.create(
    model="gpt-4o-mini",
    input=[{"role": "user", "content": user_message}],
    tools=tools
)

# Execute the tool call
for output in response.output:
    if output.type == "function_call":
        args = json.loads(output.arguments)
        result = list_files(**args)

        input_with_result = [
            {"role": "user", "content": user_message},
            {"type": "function_call_output", "call_id": output.call_id, "output": result}
        ]

        final_response = client.responses.create(
            model="gpt-4o-mini",
            input=input_with_result,
            tools=tools
        )

        # final_response.output_text


##=================================================##
## Example 3: Multiple tools with registry pattern
##=================================================##

def find_files(pattern: str, directory: str = ".") -> str:
    """Find files matching a pattern in a directory.

    Args:
        pattern: File pattern to match (e.g., '*.py', 'test_*')
        directory: Directory to search in (default: current directory)
    """
    try:
        import os
        import fnmatch
        matches = []
        for filename in os.listdir(directory):
            if fnmatch.fnmatch(filename, pattern):
                matches.append(filename)
        if not matches:
            return f"No files matching '{pattern}' found in '{directory}'"
        return f"Found {len(matches)} file(s): {', '.join(matches)}"
    except Exception as e:
        return f"Error: {str(e)}"


def get_file_size(filename: str) -> str:
    """Get the size of a file.

    Args:
        filename: Path to the file
    """
    try:
        import os
        size = os.path.getsize(filename)
        if size < 1024:
            return f"{filename}: {size} bytes"
        elif size < 1024 * 1024:
            return f"{filename}: {size / 1024:.1f} KB"
        else:
            return f"{filename}: {size / (1024 * 1024):.1f} MB"
    except FileNotFoundError:
        return f"Error: File '{filename}' not found"
    except Exception as e:
        return f"Error: {str(e)}"


# Create a registry to manage multiple tools
tool_registry = {
    "list_files": list_files,
    "find_files": find_files,
    "get_file_size": get_file_size,
}

# Generate all schemas at once
tools = [Tool.from_function(func).to_dict() for func in tool_registry.values()]

user_message = "What Python files are in the current directory?"

response = client.responses.create(
    model="gpt-4o-mini",
    input=[{"role": "user", "content": user_message}],
    tools=tools
)

# Execute all tool calls using the registry
for output in response.output:
    if output.type == "function_call":
        args = json.loads(output.arguments)
        func = tool_registry[output.name]
        result = func(**args)

        input_with_result = [
            {"role": "user", "content": user_message},
            {"type": "function_call_output", "call_id": output.call_id, "output": result}
        ]

        final_response = client.responses.create(
            model="gpt-4o-mini",
            input=input_with_result,
            tools=tools
        )

        # final_response.output_text


##=================================================##
## Example 4: Error handling in tools
##=================================================##

def read_file(filename: str) -> str:
    """Read the contents of a text file.

    Args:
        filename: Path to the file to read
    """
    try:
        with open(filename, 'r') as f:
            content = f.read()
        # Limit output to avoid overwhelming the LLM
        if len(content) > 500:
            return f"File content (first 500 chars): {content[:500]}..."
        return f"File content: {content}"
    except FileNotFoundError:
        # Return errors as strings so the LLM can explain them
        return f"Error: File '{filename}' not found"
    except PermissionError:
        return f"Error: Permission denied reading '{filename}'"
    except Exception as e:
        return f"Error: {str(e)}"


# Create tools
tools = [
    Tool.from_function(read_file).to_dict(),
    Tool.from_function(list_files).to_dict(),
]

tool_registry = {
    "read_file": read_file,
    "list_files": list_files,
}

user_message = "Read the file 'nonexistent.txt'"

response = client.responses.create(
    model="gpt-4o-mini",
    input=[{"role": "user", "content": user_message}],
    tools=tools
)

for output in response.output:
    if output.type == "function_call":
        args = json.loads(output.arguments)
        func = tool_registry[output.name]
        result = func(**args)  # This will return "Error: Cannot divide by zero"

        input_with_result = [
            {"role": "user", "content": user_message},
            {"type": "function_call_output", "call_id": output.call_id, "output": result}
        ]

        final_response = client.responses.create(
            model="gpt-4o-mini",
            input=input_with_result,
            tools=tools
        )

        # final_response.output_text
        # The LLM receives the error and explains it naturally to the user
