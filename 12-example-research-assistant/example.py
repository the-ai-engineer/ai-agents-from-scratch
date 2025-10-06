"""
Lesson 11: Complete Research Assistant

A production-ready research assistant with multiple tools for information
gathering and knowledge management.
"""

import os
import json
from typing import Callable
from openai import OpenAI
from pydantic import BaseModel, Field
from dotenv import load_dotenv

load_dotenv()


class Agent:
    """Production-ready agent with tool calling"""

    def __init__(self, model: str = "gpt-4o-mini", max_iterations: int = 10, instructions: str = None):
        self.client = OpenAI(api_key=os.getenv("OPENAI_API_KEY"))
        self.model = model
        self.max_iterations = max_iterations
        self.tools = {}
        self.tool_schemas = []
        self.conversation_history = []
        self.instructions = instructions or "You are a helpful assistant."

    def register_tool(self, name: str, function: Callable, args_schema: type[BaseModel], description: str):
        """Register a tool"""
        self.tools[name] = function
        self.tool_schemas.append({
            "type": "function",
            "function": {
                "name": name,
                "description": description,
                "parameters": args_schema.model_json_schema()
            }
        })

    def chat(self, message: str) -> str:
        """Send a message and get a response

        Note: Uses Chat Completions API because tool calling requires it.
        """
        # Add system message if this is the first message
        if not self.conversation_history and self.instructions:
            self.conversation_history.append({"role": "system", "content": self.instructions})

        self.conversation_history.append({"role": "user", "content": message})

        for iteration in range(self.max_iterations):
            response = self.client.chat.completions.create(
                model=self.model,
                messages=self.conversation_history,
                tools=self.tool_schemas if self.tool_schemas else None
            )

            message_obj = response.choices[0].message

            if not message_obj.tool_calls:
                answer = message_obj.content or "No response generated"
                self.conversation_history.append({"role": "assistant", "content": answer})
                return answer

            # Handle tool calls
            self.conversation_history.append(message_obj)

            for tool_call in message_obj.tool_calls:
                if tool_call.function.name in self.tools:
                    args = json.loads(tool_call.function.arguments)
                    result = str(self.tools[tool_call.function.name](**args))
                else:
                    result = f"Tool not found: {tool_call.function.name}"

                self.conversation_history.append({
                    "role": "tool",
                    "tool_call_id": tool_call.id,
                    "content": result
                })

        return "Max iterations reached"


# Research tools
RESEARCH_NOTES = {}


def search_web(query: str) -> str:
    """Simulate web search"""
    mock_results = {
        "python": "Python is a high-level programming language known for simplicity and readability.",
        "machine learning": "Machine learning enables systems to learn from data without explicit programming.",
        "async": "Asynchronous programming allows concurrent operations without blocking.",
    }

    for key, value in mock_results.items():
        if key.lower() in query.lower():
            return f"Search results for '{query}':\n\n{value}"

    return f"Search results for '{query}':\n\nFound information about {query}."


def calculate(expression: str) -> str:
    """Evaluate mathematical expressions"""
    try:
        result = eval(expression, {"__builtins__": {}}, {})
        return f"Result: {result}"
    except Exception as e:
        return f"Error: {str(e)}"


def save_note(title: str, content: str) -> str:
    """Save a research note"""
    note_id = f"note_{len(RESEARCH_NOTES) + 1}"
    RESEARCH_NOTES[note_id] = {"title": title, "content": content}
    return f"Note saved: '{title}'"


class SearchWebArgs(BaseModel):
    query: str = Field(description="Search query")


class CalculateArgs(BaseModel):
    expression: str = Field(description="Mathematical expression")


class SaveNoteArgs(BaseModel):
    title: str = Field(description="Note title")
    content: str = Field(description="Note content")


def main():
    """Demo the research assistant"""
    print("Creating Research Assistant...\n")

    instructions = """You are an intelligent research assistant.
Use tools to gather information, perform calculations, and save findings.
Break down complex tasks into steps and provide comprehensive answers."""

    agent = Agent(model="gpt-4o-mini", max_iterations=10, instructions=instructions)

    agent.register_tool("search_web", search_web, SearchWebArgs, "Search the web")
    agent.register_tool("calculate", calculate, CalculateArgs, "Perform calculations")
    agent.register_tool("save_note", save_note, SaveNoteArgs, "Save research notes")

    print("Agent ready with 3 tools\n")

    # Test scenarios
    scenarios = [
        "What is Python and why is it popular?",
        "Research machine learning and calculate 15 * 8 study hours",
        "Search for async programming, save key points as a note titled 'Async Basics'"
    ]

    for i, query in enumerate(scenarios, 1):
        print(f"Query {i}: {query}")
        answer = agent.chat(query)
        print(f"Answer: {answer}\n")

    if RESEARCH_NOTES:
        print("\nSaved Notes:")
        for note_id, note in RESEARCH_NOTES.items():
            print(f"- {note['title']}: {note['content']}")


if __name__ == "__main__":
    main()
