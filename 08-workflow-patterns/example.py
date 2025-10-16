"""
Lesson 03: Structured Output with Pydantic

Learn how to get type-safe, validated JSON responses using Pydantic models.
"""

from openai import OpenAI
from pydantic import BaseModel, Field
from typing import Literal, Optional
from dotenv import load_dotenv

load_dotenv()

client = OpenAI()

##=================================================##
## Example 1: Basic Extraction
##=================================================##


class CalendarEvent(BaseModel):
    title: str
    date: str  # YYYY-MM-DD
    time: str  # HH:MM
    attendees: list[str]


# Natural language → Structured data
text = "Schedule Q4 planning with John and Sarah next Friday at 2pm"

response = client.responses.parse(
    model="gpt-4o-mini", input=text, text_format=CalendarEvent
)
event = response.output_parsed

# Access fields: event.title, event.date, event.attendees


##=================================================##
## Example 2: Classification with Validation
##=================================================##


class SupportTicket(BaseModel):
    customer_name: str
    issue_type: Literal["technical", "billing", "shipping", "other"]
    priority: Literal["low", "medium", "high", "urgent"]
    order_id: Optional[str] = None
    summary: str
    next_action: str


# Parse customer email into ticket
email = """
From: Jane Smith
Order #12345 hasn't arrived and I need it for tomorrow's presentation!
Please overnight a replacement or refund me immediately.
"""

response = client.responses.parse(
    model="gpt-4o-mini", input=email, text_format=SupportTicket
)
ticket = response.output_parsed

# Automatic validation and categorization
# ticket.priority → "urgent"
# ticket.issue_type → "shipping"


##=================================================##
## Example 3: Nested Structures
##=================================================##


class Sentiment(BaseModel):
    score: float  # -1 to 1
    label: Literal["negative", "neutral", "positive"]


class ProductReview(BaseModel):
    product: str
    rating: int = Field(ge=1, le=5)
    sentiment: Sentiment
    pros: list[str]
    cons: list[str]
    would_recommend: bool


# Analyze complex review
review = """
iPhone 15 Pro: 4/5 stars
Amazing camera and build quality. Photos are incredible!
But battery life is disappointing and it's overpriced.
Would recommend if you can afford it.
"""

response = client.responses.parse(
    model="gpt-4o-mini", input=review, text_format=ProductReview
)
analysis = response.output_parsed

# Access nested data
# analysis.sentiment.label
# analysis.pros
# analysis.rating
