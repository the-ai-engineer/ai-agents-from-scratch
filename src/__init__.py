"""
Core primitives for building AI agents from scratch.

This package provides the fundamental building blocks for creating AI agents:
- Tool: Convert Python functions to LLM-compatible tool schemas
- tool: Decorator for marking functions as tools
- Agent: Autonomous agent with tool-calling capabilities
- ConversationMemory: Manage conversation history

These components are designed to be simple enough for learning while being
robust enough for real use. They demonstrate the core patterns that power
production agent systems.
"""

from .tool import Tool, tool
from .agent import Agent, ConversationMemory

__all__ = ["Tool", "tool", "Agent", "ConversationMemory"]
__version__ = "0.1.0"
