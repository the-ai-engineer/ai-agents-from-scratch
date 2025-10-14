"""
Lesson 01: OpenAI API Basics

Learn how to make your first API call to OpenAI.
"""

from openai import OpenAI
from dotenv import load_dotenv

load_dotenv()

client = OpenAI()

##=================================================##
## Example 1: Basic API call using Responses API
##=================================================##

response = client.responses.create(
    model="gpt-4o-mini",
    instructions="You are a helpful Python programming assistant.",
    input="Explain what async/await does in Python in 2 sentences.",
    temperature=0,
    store=True,
)

# response.output_text)

##=================================================##
## Example 2: Understanding responses and costs
##=================================================##

response.model_dump()

##=================================================##
## Example 2: Streaming responses for better UX
##=================================================##

stream = client.responses.create(
    model="gpt-4o-mini",
    input="Write a long haiku about coding.",
    stream=True,
)

for event in stream:
    if hasattr(event, "type") and "text.delta" in event.type:
        print(event.delta, end="", flush=True)
