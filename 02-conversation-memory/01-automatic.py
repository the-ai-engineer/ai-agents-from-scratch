"""
Lesson 02: Conversation Memory - Automatic Management

Learn how OpenAI manages conversation state server-side automatically.

Run from project root:
    uv run python 02-conversation-memory/01-automatic.py
"""

from openai import OpenAI
from openai import conversations
from dotenv import load_dotenv

load_dotenv()
client = OpenAI()


##=================================================##
## Automatic conversation management with OpenAI
##=================================================##

# Create a conversation - OpenAI will manage state server-side
conversation = conversations.create()

# Turn 1
response = client.responses.create(
    model="gpt-4o-mini",
    input="My name is Alice.",
    store=True,
    conversation=conversation.id,
)

# response.output_text

# Turn 2 - automatically remembers Turn 1
response = client.responses.create(
    model="gpt-4o-mini",
    input="What's my name?",
    store=True,
    conversation=conversation.id,
)

# response.output_text  # "Your name is Alice."

# Turn 3 - still remembers
response = client.responses.create(
    model="gpt-4o-mini",
    input="Can you remind me what we discussed?",
    store=True,
    conversation=conversation.id,
)

# response.output_text
