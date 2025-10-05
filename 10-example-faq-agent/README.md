# Lesson 10: Build a Production-Ready FAQ Agent with RAG

## What You'll Build

In this lesson, you're building a complete, production-ready FAQ agent powered by Retrieval-Augmented Generation (RAG). This isn't a simplified demo—it's a real system that searches a vector database, retrieves relevant information, and generates accurate answers with source citations.

You'll combine everything you've learned so far: the Agent class, tool calling, structured outputs, and semantic search. By the end, you'll have a working system that answers questions from a knowledge base intelligently and reliably.

FAQ agents are one of the most valuable AI applications in business. They reduce support ticket volume by 40-60%, provide instant answers 24/7, and free your team to focus on complex problems. This pattern applies to customer support bots, internal documentation systems, HR policy assistants, and technical runbooks.

## The Problem

Every company has knowledge trapped in documents—FAQs, policies, product docs, internal wikis. Finding answers requires searching multiple systems, reading entire documents, and often interrupting colleagues with questions.

Traditional search gives you matching keywords. Users still have to read through long documents to find specific answers. It doesn't scale, wastes time, and creates bottlenecks as teams grow.

The solution is a RAG-powered agent that understands questions semantically, retrieves only relevant information, and generates precise answers in natural language—all while citing its sources so users can verify accuracy.

## What You'll Learn

You'll master the complete RAG pipeline from end to end. This includes setting up ChromaDB for vector storage, creating embeddings to represent text semantically, implementing semantic search that finds meaning not just keywords, and integrating RAG capabilities into your Agent class.

You'll also learn production patterns like chunking documents intelligently, handling "no matches found" gracefully, providing source attribution, and balancing retrieval count for accuracy versus cost.

This is the foundation for any knowledge-based AI system. Master this pattern and you can build internal documentation assistants, customer support bots, engineering runbook systems, and more.

## How RAG Works

RAG stands for **Retrieval-Augmented Generation**. Instead of relying on the LLM's training data (which gets outdated), you give it access to your own knowledge base.

Here's the process:

1. **User asks a question**: "What's your refund policy?"
2. **Search knowledge base**: Find the 3-5 most relevant documents using semantic search
3. **Retrieve context**: Pull the actual text from those documents
4. **Pass to LLM**: Send the question plus retrieved documents as context
5. **Generate answer**: LLM answers based on your documents, not its training data
6. **Cite sources**: Include which documents were used so users can verify

The key insight: LLMs are excellent at answering questions when given relevant context. RAG provides that context dynamically from your own data.

## When to Use RAG

RAG solves specific problems. Use it when you have company-specific knowledge the LLM wasn't trained on, when information changes frequently and you need up-to-date answers, when factual accuracy is critical and you need source attribution, or when you're working in specialized domains with proprietary information.

Don't use RAG for general knowledge questions the LLM already knows, for creative tasks where facts don't matter, or when you don't have a well-organized knowledge base to search.

| Scenario | Use RAG? | Why |
|----------|----------|-----|
| Company FAQ / policies | ✅ Yes | Company-specific, needs to be accurate |
| Product documentation | ✅ Yes | Technical details, frequently updated |
| Code repositories | ✅ Yes | Proprietary code, specific to your system |
| General knowledge ("What is Python?") | ❌ No | LLM already knows this |
| Creative writing | ❌ No | Facts aren't needed |

## System Architecture

Your FAQ agent follows this architecture:

```
User Question
    ↓
Agent (with search_faq tool)
    ↓
Tool Call: search_faq(query)
    ↓
Vector Search (ChromaDB)
    ↓
Retrieve Top 3 FAQs
    ↓
Return to Agent
    ↓
LLM Generates Answer with Sources
    ↓
Response to User
```

The agent decides when to search (it's just another tool). The search tool converts the query to an embedding, finds similar documents in ChromaDB, and returns relevant FAQs. The agent then uses those FAQs to answer the user's question.

## Tech Stack

You're using four key technologies:

**ChromaDB**: A vector database optimized for semantic search. It stores documents as embeddings and retrieves them based on similarity.

**OpenAI Embeddings**: Convert text into high-dimensional vectors that capture semantic meaning. Similar concepts have similar embeddings.

**Agent Class**: Your reusable Agent from Lesson 07. It handles the conversation loop and tool orchestration.

**Pydantic**: Validates tool inputs and outputs to ensure type safety.

This stack is production-ready. Companies use these exact tools to build knowledge bases serving millions of users.

## Implementation: Setting Up ChromaDB

First, install the required packages (or UV will do this automatically when you run the example):

Initialize ChromaDB and create a collection:

```python
import chromadb
from chromadb.utils import embedding_functions

# Initialize ChromaDB (in-memory for development)
client = chromadb.Client()

# Use OpenAI embeddings
openai_ef = embedding_functions.OpenAIEmbeddingFunction(
    api_key=os.getenv("OPENAI_API_KEY"),
    model_name="text-embedding-3-small"
)

# Create a collection
collection = client.create_collection(
    name="faqs",
    embedding_function=openai_ef
)
```

In production, use persistent storage instead of in-memory mode.

## Implementation: Loading Documents

Load your FAQs into the vector database:

```python
# Example FAQs
faqs = [
    {
        "id": "faq1",
        "question": "What is your refund policy?",
        "answer": "We offer full refunds within 30 days of purchase. No questions asked.",
        "category": "billing"
    },
    {
        "id": "faq2",
        "question": "How long does shipping take?",
        "answer": "Standard shipping takes 5-7 business days. Express shipping takes 2-3 days.",
        "category": "shipping"
    },
    {
        "id": "faq3",
        "question": "Do you ship internationally?",
        "answer": "Yes, we ship to over 100 countries. International shipping takes 10-15 business days.",
        "category": "shipping"
    }
]

# Add to ChromaDB
for faq in faqs:
    collection.add(
        ids=[faq["id"]],
        documents=[f"Q: {faq['question']}\nA: {faq['answer']}"],
        metadatas=[{"category": faq["category"]}]
    )
```

The documents are automatically converted to embeddings and stored.

## Implementation: Search Tool

Create a tool that searches the FAQ database:

```python
from pydantic import BaseModel, Field

class SearchFAQInput(BaseModel):
    query: str = Field(description="The user's question to search for")

def search_faq(query: str) -> str:
    """Search the FAQ knowledge base for relevant information."""
    # Query the vector database
    results = collection.query(
        query_texts=[query],
        n_results=3  # Retrieve top 3 matches
    )

    # No results found
    if not results['documents'][0]:
        return "No relevant FAQs found. Please try rephrasing your question."

    # Format results
    formatted_results = []
    for doc, metadata in zip(results['documents'][0], results['metadatas'][0]):
        formatted_results.append(f"{doc}\nCategory: {metadata['category']}")

    return "\n\n".join(formatted_results)

# Tool definition for the agent
search_tool = {
    "type": "function",
    "function": {
        "name": "search_faq",
        "description": "Search the FAQ knowledge base for information to answer user questions",
        "parameters": SearchFAQInput.model_json_schema()
    }
}
```

The search tool handles the query, retrieves matches, formats results, and returns them to the agent.

## Implementation: Putting It All Together

Connect your agent with the search tool:

```python
from agent import Agent  # Your Agent class from Lesson 07

# Initialize agent with the search tool
system_prompt = """You are a helpful FAQ assistant.

When users ask questions:
1. Use the search_faq tool to find relevant information
2. Answer based on the retrieved FAQs
3. Always cite your sources by mentioning the category
4. If no relevant information is found, politely say you don't have that information

Be concise and helpful."""

agent = Agent(
    model="gpt-4o-mini",
    system_prompt=system_prompt,
    tools=[search_tool]
)

# Define tool executor
def execute_tool(tool_name: str, tool_input: dict) -> str:
    if tool_name == "search_faq":
        return search_faq(tool_input["query"])
    return "Unknown tool"

# Run the agent
response = agent.run(
    user_message="How long does shipping take?",
    tool_executor=execute_tool
)

print(response)
```

The agent automatically decides when to search, uses the retrieved context, and generates accurate answers.

## Running the Example

This lesson includes a complete working implementation:

```bash
cd 10-example-faq-agent
uv run example.py
```

Try these queries:
- "What's your refund policy?"
- "Do you ship to other countries?"
- "What payment methods do you accept?" (not in FAQs)

Notice how the agent searches when needed and admits when it doesn't have information.

## Key Takeaways

1. **RAG as a tool**: Treat retrieval as just another tool the agent can use. This keeps your architecture clean and flexible.

2. **Top K = 3-5 is optimal**: Retrieving more documents doesn't always improve answers. Too much context confuses the LLM and increases costs.

3. **Always include sources**: Users need to verify information. Always cite which documents or categories you used.

4. **Chunk documents intelligently**: Respect document structure when splitting text. A good FAQ is already one semantic chunk.

5. **Handle "not found" gracefully**: When search returns no matches, return a helpful message. Don't let the LLM hallucinate answers.

6. **Embeddings are cheap, storage is cheap**: The main cost is LLM generation, not embeddings or storage. Don't over-optimize the wrong thing.

## Common Pitfalls

1. **Retrieving too many documents**: More isn't better. Stick to 3-5 most relevant matches. Too much context degrades quality.

2. **Poor chunking**: Breaking documents mid-sentence or splitting related information hurts retrieval quality.

3. **Not handling empty results**: Always check if search returned matches before passing to the LLM.

4. **Forgetting metadata**: Store categories, timestamps, source URLs. You'll need them for citations and debugging.

5. **Using in-memory ChromaDB in production**: Always use persistent storage in production so data survives restarts.

## Real-World Impact

Companies using RAG-powered FAQ agents see measurable results. Support ticket volume drops by 40-60% as common questions get answered instantly. First response time goes from minutes or hours to seconds. Support teams spend less time on repetitive questions and more on complex issues that need human judgment.

For internal tools, employees find answers faster, reducing interruptions to managers and subject matter experts. Onboarding new hires becomes faster because information is accessible on demand.

The business value is clear: lower support costs, faster response times, higher customer satisfaction, and better knowledge retention.

## Assignment

Build your own FAQ agent for a domain you care about. Create 10-15 FAQs on a topic (your business, a hobby, anything with questions people ask repeatedly). Load them into ChromaDB, build the search tool, connect it to your Agent, and test it with 5-10 questions.

Pay attention to:
- Does it retrieve the right FAQs?
- Are the answers accurate?
- Does it handle questions outside the FAQ gracefully?
- Are sources properly cited?

## Next Steps

You've built a complete RAG-powered agent. Next, move to [Lesson 11 - Research Assistant](../11-example-research-assistant) to see a multi-tool agent that combines search, analysis, and note-taking into a comprehensive research workflow.

## Resources

- [ChromaDB Documentation](https://docs.trychroma.com/)
- [OpenAI Embeddings Guide](https://platform.openai.com/docs/guides/embeddings)
- [RAG Best Practices by Pinecone](https://www.pinecone.io/learn/retrieval-augmented-generation/)
- [Vector Databases Explained](https://www.pinecone.io/learn/vector-database/)
