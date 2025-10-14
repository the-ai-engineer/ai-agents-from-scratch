from dataclasses import dataclass
from typing import Callable
import inspect


@dataclass
class Tool:
    name: str
    description: str
    parameters: dict

    def to_dict(self) -> dict:
        """Convert to OpenAI format."""
        return {
            "type": "function",
            "function": {
                "name": self.name,
                "description": self.description,
                "parameters": self.parameters,
            },
        }

    @classmethod
    def from_function(cls, func: Callable) -> "Tool":
        """Create Tool from a function."""
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
