import inspect
from typing import Callable, get_type_hints, Any
from pydantic import BaseModel, create_model


def extract_function_params(func: Callable) -> tuple[dict, list[str]]:
    """Extract parameter fields and required list from a function.

    Returns:
        tuple: (fields dict for Pydantic, list of required param names)
    """
    sig = inspect.signature(func)
    type_hints = get_type_hints(func)

    fields = {}
    required = []

    for param_name, param in sig.parameters.items():
        if param_name == "self":
            continue

        param_type = type_hints.get(param_name, str)

        # Check if parameter has a default value
        if param.default == inspect.Parameter.empty:
            # Required parameter
            default = ...
            required.append(param_name)
        else:
            # Optional parameter
            default = None
            param_type = param_type | None

        fields[param_name] = (param_type, default)

    return fields, required


def create_json_schema(func: Callable) -> dict[str, Any]:
    """Create a JSON schema from a function signature."""
    fields, required = extract_function_params(func)

    # Create Pydantic model and convert to JSON schema
    args_schema = create_model(f"{func.__name__}_args", **fields)
    schema = args_schema.model_json_schema()

    # Add required fields list
    schema["required"] = required

    return schema


class Tool(BaseModel):
    name: str
    description: str
    input_schema: dict[str, Any]

    @classmethod
    def from_function(cls, func: Callable) -> "Tool":
        """Create a Tool from a function automatically using inspect."""
        return cls(
            name=func.__name__,
            description=inspect.getdoc(func) or "",
            input_schema=create_json_schema(func),
        )

    def to_anthropic_format(self) -> dict:
        """Convert to Anthropic/Claude tool format."""
        return {
            "name": self.name,
            "description": self.description,
            "input_schema": self.input_schema,
        }

    def to_openai_format(self) -> dict:
        """Convert to OpenAI tool format."""
        schema = self.input_schema.copy()
        schema["additionalProperties"] = False

        return {
            "type": "function",
            "name": self.name,
            "description": self.description,
            "parameters": schema,
            "strict": True,
        }


def tool(func: Callable) -> Callable:
    """Decorator to attach Tool schema to a function."""
    func.tool = Tool.from_function(func)
    return func


# Usage example:
@tool
def get_weather(location: str, units: str = "celsius") -> str:
    """Get the current weather in a given location.

    Args:
        location: The city and state, e.g. San Francisco, CA
        units: Temperature units (celsius or fahrenheit)
    """
    return f"Weather in {location}"


# Test both formats
print("Claude format:")
print(get_weather.tool.to_anthropic_format())

print("\nOpenAI format:")
print(get_weather.tool.to_openai_format())
