# Lesson 01: Your First OpenAI API Call

Every AI application you'll ever build starts here: making API calls to Large Language Models. Before you touch any framework or library, you need to understand how to communicate with models directly using an API.

In this lesson, you'll make your first API call to OpenAI, learn how to control model behavior with simple parameters, and understand the fundamental patterns that power everything from chatbots to autonomous agents. You'll see how to stream responses for better user experience, make concurrent requests for speed, and handle the inevitable errors that occur in production.

Note: We'll use the **Responses API**, OpenAI's recommended interface released in March 2025. It's an evolution of the older Chat Completions API that simplifies common patterns and adds powerful capabilities like server-side conversation management and built-in tools. Everything you learn here will transfer to more advanced use cases later.

## Setting Up Your Environment

Before you make your first API call, get your API key from [platform.openai.com](https://platform.openai.com/api-keys). Create a `.env` file in the repository root with your key:

```
OPENAI_API_KEY=your-key-here
```

Never hardcode API keys in your source code. Always use environment variables to keep credentials secure and separate from your codebase.

## Your First API Call

Making an API call is surprisingly straightforward. You initialize a client with your API key, call `responses.create()` with your model and input, and access the response text. Here's the minimal version:

```python
from openai import OpenAI
from dotenv import load_dotenv

load_dotenv()
client = OpenAI()

response = client.responses.create(
    model="gpt-4o-mini",
    input="Explain async/await in Python in one paragraph"
)

print(response.output_text)
```

Run this code. If you see a response explaining async/await, congratulations - you just communicated with an AI model.

To control the model's behavior, add instructions (system prompt):

```python
response = client.responses.create(
    model="gpt-4o-mini",
    instructions="You are a helpful Python tutor. Explain concepts clearly with examples.",
    input="Explain async/await in Python"
)
```

The instructions shape how the model interprets and responds to your input. Experiment with different instructions to see how the same input can yield very different responses.

## Parameters That Control Behavior

Three main parameters control how the model generates responses:

**Model** determines which model version processes your request. Each models has pros/cons. Smaller models tend to be faster and cheaper and larger models tend to be smarter. 

**Temperature** controls how creative the model is in it's responses. Set it to 0 for deterministic, factual tasks like code generation or data extraction. The model will consistently choose the most likely tokens, giving you repeatable results. Set it higher (0.5-1.0) for creative tasks like brainstorming or writing. 

**Stream** determines whether you get the full response at once or token-by-token as it's generated. Streaming creates a better user experience for longer responses—users see text appearing progressively, like in ChatGPT.

## Understanding Responses and Costs

Every response includes more than just the generated text. The response object contains usage information that's critical for cost management:

- `response.output_text` gives you the generated text
- `response.usage.input_tokens` shows tokens in your prompt
- `response.usage.output_tokens` shows tokens in the response
- `response.usage.total_tokens` shows combined token count
- `response.model` confirms which model processed your request

Tokens determine your costs. OpenAI charges based on input and output tokens, with prices varying by model. Production systems must track token usage to avoid surprises. A chatbot that processes thousands of conversations without monitoring could rack up significant costs quickly.

## Streaming for Better UX

When responses might take a few seconds, streaming dramatically improves user experience. Instead of waiting for the complete response, users see text appearing progressively—exactly like ChatGPT's typing effect.

```python
stream = client.responses.create(
    model="gpt-4o-mini",
    input="Write a haiku about coding",
    stream=True
)

for event in stream:
    if hasattr(event, "type") and "text.delta" in event.type:
        print(event.delta, end="", flush=True)
```

The difference is psychological but powerful. A 5-second wait feels much longer than watching text appear over 5 seconds. For customer-facing applications, chatbots, or any interactive interface, streaming is essential.

## Async Client for Speed

When you need to make multiple API calls, doing them sequentially is wasteful. While one request waits for a response, your code sits idle. The `AsyncOpenAI` client lets you make concurrent requests, dramatically reducing total execution time.

```python
from openai import AsyncOpenAI
import asyncio

async_client = AsyncOpenAI()

async def process_multiple_prompts():
    prompts = ["What is Python?", "What is JavaScript?", "What is Rust?"]

    tasks = [
        async_client.responses.create(
            model="gpt-4o-mini",
            input=prompt,
            temperature=0
        )
        for prompt in prompts
    ]

    responses = await asyncio.gather(*tasks)
    return responses
```

This pattern is essential for production systems. If you're building a bulk content generator, a multi-step research assistant, or processing user requests in a web application, async execution prevents bottlenecks.

The example code includes a timing comparison (`02-async.py`) showing that parallel execution is typically 2-3x faster than sequential. For applications processing dozens or hundreds of requests, this difference compounds quickly.

## Error Handling

API calls fail more often than you'd expect. Networks drop, rate limits hit, services have outages. Production systems must handle these failures gracefully.

The OpenAI client provides specific exception types for different failure modes. Catch `RateLimitError` when you exceed your quota, `APIConnectionError` for network issues, and the general `APIError` as a catchall. Your code should handle these exceptions appropriately—perhaps by showing users an error message, queuing the request for retry, or logging the failure for investigation.

Beyond exception handling, configure automatic retries at the client level:

```python
from openai import OpenAI

client = OpenAI(
    max_retries=3,
    timeout=60.0
)
```

The client will automatically retry failed requests up to three times with exponential backoff. This handles transient network issues and temporary service disruptions without additional code. The timeout prevents requests from hanging indefinitely.

In production, you'll want detailed logging around failures to understand patterns and identify issues before they impact users.

## Assignment

Make your first call to OpenAI using the Python SDK. Experiment with temperature and system prompt (instructions).

## Resources

- [Responses API Reference](https://platform.openai.com/docs/api-reference/responses)
- [Responses API Quickstart](https://platform.openai.com/docs/quickstart?api-mode=responses)
- [Migrate to Responses API](https://platform.openai.com/docs/guides/migrate-to-responses)
- [OpenAI Best Practices](https://platform.openai.com/docs/guides/prompt-engineering)
