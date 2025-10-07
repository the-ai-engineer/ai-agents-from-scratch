"""
Lesson 08: Complete FAQ Agent with RAG

A production-ready FAQ agent that uses vector search to answer questions
from a knowledge base.
"""

import os
import sys
import chromadb
from chromadb.utils import embedding_functions
from dotenv import load_dotenv

# Add parent directory to path to import agents framework
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from agents import Agent, tool

load_dotenv()


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


# Global FAQ database instance
faq_db = None


def create_search_faq_tool():
    """Factory function to create the search_faq tool with access to the database"""
    @tool
    def search_faq(query: str, top_k: int = 3) -> str:
        """Search the FAQ database for relevant information

        Args:
            query: Search query or question to find relevant FAQs
            top_k: Number of results to return (1-5, default: 3)
        """
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
    global faq_db

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

    # Create search tool with access to database
    search_faq = create_search_faq_tool()

    # Create agent with tools
    system_prompt = """You are a helpful customer support agent.
Search the FAQ database to answer questions.
Provide clear answers based on the FAQs.
If no relevant FAQ exists, say so politely."""

    agent = Agent(
        model="gpt-4o-mini",
        max_iterations=5,
        system_prompt=system_prompt,
        tools=[search_faq]
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
