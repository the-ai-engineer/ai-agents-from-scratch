"""
Lesson 09: Memory and State Management

Learn how to manage conversation history, handle token limits, and persist state.
"""

import os
import json
import tiktoken
from typing import Callable
from openai import OpenAI
from pydantic import BaseModel, Field
from dotenv import load_dotenv

load_dotenv()


class AgentWithMemory:
    """Agent with token counting and history trimming"""

    def __init__(self, model: str = "gpt-4o-mini", max_iterations: int = 5, max_history_tokens: int = 4000):
        self.client = OpenAI(api_key=os.getenv("OPENAI_API_KEY"))
        self.model = model
        self.max_iterations = max_iterations
        self.max_history_tokens = max_history_tokens
        self.tools = {}
        self.tool_schemas = []
        self.conversation_history = []
        self.tokenizer = tiktoken.encoding_for_model(model)

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

    def _count_tokens(self, messages: list) -> int:
        """Count tokens in message history"""
        token_count = 0
        for message in messages:
            if isinstance(message.get("content"), str):
                token_count += len(self.tokenizer.encode(message["content"]))
            token_count += 4  # Overhead per message
        return token_count

    def _trim_history(self):
        """Trim old messages to stay under token limit"""
        current_tokens = self._count_tokens(self.conversation_history)

        if current_tokens <= self.max_history_tokens:
            return

        print(f"Trimming history: {current_tokens} tokens > {self.max_history_tokens} limit")

        # Keep system messages, remove oldest user/assistant messages
        system_messages = [msg for msg in self.conversation_history if msg["role"] == "system"]
        other_messages = [msg for msg in self.conversation_history if msg["role"] != "system"]

        while self._count_tokens(system_messages + other_messages) > self.max_history_tokens and other_messages:
            other_messages.pop(0)

        self.conversation_history = system_messages + other_messages

    def chat(self, message: str) -> str:
        """Send a message and get a response

        Note: Uses Chat Completions API because tool calling requires it.
        """
        self._trim_history()

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

            # Add assistant message with tool calls
            self.conversation_history.append(message_obj)

            # Execute tools
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

    def save_conversation(self, filepath: str):
        """Save conversation to JSON file"""
        with open(filepath, 'w') as f:
            json.dump(self.conversation_history, f, indent=2)
        print(f"Saved to {filepath}")

    def load_conversation(self, filepath: str):
        """Load conversation from JSON file"""
        with open(filepath, 'r') as f:
            self.conversation_history = json.load(f)
        print(f"Loaded from {filepath}")

    def get_token_count(self) -> int:
        """Get current token count"""
        return self._count_tokens(self.conversation_history)


def token_management_example():
    """Example 1: Token counting and automatic trimming"""
    print("=== Token Management ===\n")

    # Create agent with low token limit to trigger trimming
    agent = AgentWithMemory(max_history_tokens=500)

    # Have a long conversation
    for i in range(8):
        question = f"Tell me a fun fact about the number {i}"
        print(f"Turn {i + 1}: {question}")
        answer = agent.chat(question)
        print(f"Tokens: {agent.get_token_count()}/{agent.max_history_tokens}\n")


def persistence_example():
    """Example 2: Save and load conversation state"""
    print("\n=== Conversation Persistence ===\n")

    # Session 1: Have a conversation
    print("Session 1:")
    agent1 = AgentWithMemory()
    agent1.chat("My name is John")
    agent1.chat("I live in Paris")
    agent1.chat("I'm a software engineer")

    # Save
    agent1.save_conversation("/tmp/conversation.json")

    # Session 2: Load in new agent
    print("\nSession 2 (after restart):")
    agent2 = AgentWithMemory()
    agent2.load_conversation("/tmp/conversation.json")

    # Agent remembers context
    answer = agent2.chat("What's my name and where do I live?")
    print(f"User: What's my name and where do I live?")
    print(f"Assistant: {answer}")


if __name__ == "__main__":
    token_management_example()
    persistence_example()
