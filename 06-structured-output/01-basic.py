"""
Lesson 03: Structured Output with Pydantic

Learn how to get type-safe, validated JSON responses using Pydantic models.
"""

from openai import OpenAI
from pydantic import BaseModel, Field
from typing import Literal
from dotenv import load_dotenv

load_dotenv()

client = OpenAI()


def call_llm_structured(input, text_format: BaseModel):
    """Helper to get structured output from LLM"""
    response = client.responses.parse(
        model="gpt-4o-mini", input=input, text_format=text_format
    )
    return response.output_parsed


##=================================================##
## Example 1: Extract event details
##=================================================##


class CalendarEvent(BaseModel):
    title: str = Field(description="Event title")
    date: str = Field(description="Event date in YYYY-MM-DD format")
    time: str = Field(description="Event time in HH:MM format")
    attendees: list[str] = Field(description="List of attendee names")


user_input = "Schedule a Q4 planning meeting with John and Sarah next Friday at 2pm"
event = call_llm_structured(user_input, CalendarEvent)

print(f"Input: {user_input}\n")
print(f"Title: {event.title}")
print(f"Date: {event.date}")
print(f"Time: {event.time}")
print(f"Attendees: {', '.join(event.attendees)}\n")


##=================================================##
## Example 2: Extract email metadata and tasks
##=================================================##


class EmailData(BaseModel):
    sender: str = Field(description="Sender name")
    subject: str = Field(description="Email subject")
    action_items: list[str] = Field(description="Action items or tasks")
    deadline: str | None = Field(
        description="Deadline in YYYY-MM-DD format", default=None
    )
    priority: Literal["high", "medium", "low"]


email_text = """
From: Mike Johnson
Subject: Urgent - API Integration Issues

We need to fix the authentication bug by end of day tomorrow.
Please also update the documentation and notify the client.
This is blocking their production deployment.
"""

email = call_llm_structured(f"Extract key information:\n{email_text}", EmailData)

print(f"From: {email.sender}")
print(f"Subject: {email.subject}")
print(f"Priority: {email.priority}")
print(f"Deadline: {email.deadline or 'Not specified'}")
print("Action Items:")
for item in email.action_items:
    print(f"  - {item}")
print()


##=================================================##
## Example 3: Nested data structures
##=================================================##


class Address(BaseModel):
    street: str
    city: str
    country: str


class Contact(BaseModel):
    phone: str
    email: str


class Company(BaseModel):
    name: str
    address: Address
    contact: Contact
    employees: int = Field(gt=0)
    founded_year: int


company_text = """
TechCorp is a software company with 150 employees, founded in 2010.
Located at 123 Main Street, San Francisco, USA.
Contact: info@techcorp.com or 415-555-0123.
"""

company = call_llm_structured(f"Extract company info:\n{company_text}", Company)

print(f"Name: {company.name}")
print(
    f"Address: {company.address.street}, {company.address.city}, {company.address.country}"
)
print(f"Email: {company.contact.email}")
print(f"Phone: {company.contact.phone}")
print(f"Employees: {company.employees}")
print(f"Founded: {company.founded_year}")
