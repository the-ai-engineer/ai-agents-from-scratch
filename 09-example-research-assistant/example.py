"""
Lesson 09: Complete Research Assistant

A production-ready research assistant with multiple tools for information
gathering and knowledge management.
"""

import os
import sys
from dotenv import load_dotenv

# Add parent directory to path to import agents framework
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from agents import Agent, tool

load_dotenv()


# Research tools - global storage for notes
RESEARCH_NOTES = {}


@tool
def search_web(query: str) -> str:
    """Search the web for information

    Args:
        query: Search query to find information online
    """
    mock_results = {
        "python": "Python is a high-level programming language known for simplicity and readability.",
        "machine learning": "Machine learning enables systems to learn from data without explicit programming.",
        "async": "Asynchronous programming allows concurrent operations without blocking.",
    }

    for key, value in mock_results.items():
        if key.lower() in query.lower():
            return f"Search results for '{query}':\n\n{value}"

    return f"Search results for '{query}':\n\nFound information about {query}."


@tool
def calculate(expression: str) -> str:
    """Evaluate mathematical expressions

    Args:
        expression: Mathematical expression to evaluate (e.g., '2+2', '15*8')
    """
    try:
        result = eval(expression, {"__builtins__": {}}, {})
        return f"Result: {result}"
    except Exception as e:
        return f"Error: {str(e)}"


@tool
def save_note(title: str, content: str) -> str:
    """Save a research note for future reference

    Args:
        title: Title or key for the note
        content: Content of the note to save
    """
    note_id = f"note_{len(RESEARCH_NOTES) + 1}"
    RESEARCH_NOTES[note_id] = {"title": title, "content": content}
    return f"Note saved: '{title}'"


def main():
    """Demo the research assistant"""
    print("Creating Research Assistant...\n")

    system_prompt = """You are an intelligent research assistant.
Use tools to gather information, perform calculations, and save findings.
Break down complex tasks into steps and provide comprehensive answers."""

    agent = Agent(
        model="gpt-4o-mini",
        max_iterations=10,
        system_prompt=system_prompt,
        tools=[search_web, calculate, save_note]
    )

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
