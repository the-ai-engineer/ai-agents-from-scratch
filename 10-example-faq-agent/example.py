"""
Lesson 10: Complete FAQ Agent with RAG

A production-ready FAQ agent that uses vector search to answer questions
from a knowledge base.
"""

import os
import json
from typing import Callable
from openai import OpenAI
from pydantic import BaseModel, Field
import chromadb
from chromadb.utils import embedding_functions
from dotenv import load_dotenv

load_dotenv()


class Agent:
    """Reusable agent with tool calling support"""

    def __init__(self, model: str = "gpt-4o-mini", max_iterations: int = 5, instructions: str = None):
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
        """Send a message and get a response"""
        self.conversation_history.append({"role": "user", "content": message})

        for iteration in range(self.max_iterations):
            response = self.client.responses.create(
                model=self.model,
                instructions=self.instructions,
                input=self.conversation_history,
                tools=self.tool_schemas if self.tool_schemas else None
            )

            if not hasattr(response, 'tool_calls') or not response.tool_calls:
                answer = response.output_text
                self.conversation_history.append({"role": "assistant", "content": answer})
                return answer

            # Handle tool calls
            self.conversation_history.append({
                "role": "assistant",
                "content": response.output_text or "",
                "tool_calls": [
                    {
                        "id": tc.id,
                        "type": "function",
                        "function": {"name": tc.function.name, "arguments": tc.function.arguments}
                    } for tc in response.tool_calls
                ]
            })

            for tool_call in response.tool_calls:
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


class FAQDatabase:
    """Vector database for FAQ storage and semantic search"""

    def __init__(self, collection_name: str = "faqs"):
        self.client = chromadb.Client()
        self.embedding_function = embedding_functions.OpenAIEmbeddingFunction(
            api_key=os.getenv("OPENAI_API_KEY"),
            model_name="text-embedding-3-small"
        )
        self.collection = self.client.create_collection(
            name=collection_name,
            embedding_function=self.embedding_function
        )

    def add_faqs_bulk(self, faqs: list[dict]):
        """Add multiple FAQs at once"""
        ids = [f"faq_{i+1}" for i in range(len(faqs))]
        documents = [f"Q: {faq['question']}\nA: {faq['answer']}" for faq in faqs]
        metadatas = [{"question": faq["question"]} for faq in faqs]

        self.collection.add(ids=ids, documents=documents, metadatas=metadatas)

    def search(self, query: str, top_k: int = 3) -> list[dict]:
        """Search for relevant FAQs using semantic search"""
        results = self.collection.query(query_texts=[query], n_results=top_k)

        faqs = []
        if results['documents'] and results['documents'][0]:
            for i, doc in enumerate(results['documents'][0]):
                parts = doc.split('\nA: ')
                question = parts[0].replace('Q: ', '') if parts else ""
                answer = parts[1] if len(parts) > 1 else ""

                faqs.append({
                    "question": question,
                    "answer": answer,
                    "score": 1 - results['distances'][0][i]
                })

        return faqs


class SearchFAQArgs(BaseModel):
    query: str = Field(description="Search query or question")
    top_k: int = Field(default=3, ge=1, le=5, description="Number of results")


def create_search_tool(faq_db: FAQDatabase):
    """Create a search function for the agent"""
    def search_faq(query: str, top_k: int = 3) -> str:
        results = faq_db.search(query, top_k)

        if not results:
            return "No relevant FAQs found."

        response = "Found the following FAQs:\n\n"
        for i, faq in enumerate(results, 1):
            response += f"{i}. Q: {faq['question']}\n   A: {faq['answer']}\n\n"

        return response

    return search_faq


def main():
    """Demo the FAQ agent"""
    print("Creating FAQ Database...")
    faq_db = FAQDatabase(collection_name="company_faqs")

    # Sample FAQs
    sample_faqs = [
        {
            "question": "What are your business hours?",
            "answer": "We are open Monday to Friday, 9 AM to 6 PM EST.",
        },
        {
            "question": "How do I reset my password?",
            "answer": "Click 'Forgot Password' on the login page and follow the email instructions.",
        },
        {
            "question": "What is your return policy?",
            "answer": "We accept returns within 30 days. Items must be unused in original packaging.",
        },
        {
            "question": "Do you offer international shipping?",
            "answer": "Yes, we ship to over 50 countries. Delivery takes 7-14 business days.",
        },
        {
            "question": "What payment methods do you accept?",
            "answer": "We accept all major credit cards, PayPal, Apple Pay, and Google Pay.",
        },
    ]

    faq_db.add_faqs_bulk(sample_faqs)
    print(f"Added {len(sample_faqs)} FAQs\n")

    # Create agent
    instructions = """You are a helpful customer support agent.
Search the FAQ database to answer questions.
Provide clear answers based on the FAQs.
If no relevant FAQ exists, say so politely."""

    agent = Agent(model="gpt-4o-mini", max_iterations=5, instructions=instructions)

    search_tool = create_search_tool(faq_db)
    agent.register_tool(
        name="search_faq",
        function=search_tool,
        args_schema=SearchFAQArgs,
        description="Search the FAQ database"
    )

    print("Agent created\n")

    # Test questions
    test_questions = [
        "What time are you open?",
        "I forgot my password",
        "Can you ship to Canada?",
    ]

    for question in test_questions:
        print(f"User: {question}")
        answer = agent.chat(question)
        print(f"Agent: {answer}\n")


if __name__ == "__main__":
    main()
