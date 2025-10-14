# Stop Parsing AI Responses: Get Type-Safe JSON Automatically

## What You'll Learn

In this lesson, you'll master structured output—the technique that transforms messy LLM text responses into clean, validated data structures you can use directly in your code.

No more regex parsing. No more brittle string manipulation. No more hoping the AI formats its response correctly. You'll learn how to get type-safe, validated JSON every single time.

This is the foundation for building reliable AI systems that extract data, classify content, and generate structured information at scale.

## The Problem

When you ask an LLM to extract information, you get unstructured text back:

```python
response = "The user's email is john@example.com and they want 3 tickets"
```

Now what? You need to write regex patterns, handle edge cases, deal with formatting variations, and pray the LLM doesn't change its response format. Every response requires parsing code that breaks easily.

In production systems, this doesn't scale. You need data extraction you can trust. You need type safety. You need validation. You need structured output.

## The Solution: Structured Output with Pydantic

Structured output forces the LLM to return valid JSON matching a schema you define. You specify exactly what fields you want, their types, and validation rules. The LLM generates a response that matches your schema perfectly.

Here's the difference:

**Without structured output:**
```python
# Get text response
response = "The event is on 2024-03-15 at 2:00 PM with John and Sarah"

# Write fragile parsing code
import re
date_match = re.search(r'\d{4}-\d{2}-\d{2}', response)
# More regex for time, attendees... 😱
```

**With structured output:**
```python
class CalendarEvent(BaseModel):
    title: str
    date: str
    time: str
    attendees: list[str]

# Get validated, typed response automatically
event = response.parsed  # Type-safe Pydantic object ✅
print(event.date)  # "2024-03-15"
print(event.attendees)  # ["John", "Sarah"]
```

No parsing. No regex. No fragile string manipulation. Just clean, validated data you can use immediately.

## How It Works

Structured output works through three simple steps:

### Step 1: Define Your Schema

Create a Pydantic model describing the data structure you want:

```python
from pydantic import BaseModel, Field

class CalendarEvent(BaseModel):
    title: str = Field(description="Event title")
    date: str = Field(description="Event date in YYYY-MM-DD format")
    time: str = Field(description="Event time in HH:MM format")
    attendees: list[str] = Field(description="List of attendee names")
```

The `Field()` descriptions help the LLM understand what you want. Think of them as instructions for the AI.

### Step 2: Use the Schema in Your API Call

Pass your Pydantic model as the `response_format`:

```python
from openai import OpenAI

client = OpenAI()

response = client.beta.chat.completions.parse(
    model="gpt-4o-mini",
    messages=[
        {"role": "user", "content": "Schedule a meeting with John and Sarah on Friday at 2pm"}
    ],
    response_format=CalendarEvent
)

event = response.choices[0].message.parsed
```

### Step 3: Access Validated Data

The response is automatically validated and typed:

```python
print(event.title)      # "Meeting with John and Sarah"
print(event.date)       # "2024-03-15"
print(event.attendees)  # ["John", "Sarah"]

# IDE autocomplete works! ✅
# Type checking works! ✅
# Validation is automatic! ✅
```

## Code Examples

### Example 1: Extract Email Data

Extract structured information from unstructured email text:

```python
from pydantic import BaseModel, Field
from typing import Literal

class EmailData(BaseModel):
    sender: str = Field(description="Sender name")
    subject: str = Field(description="Email subject")
    action_items: list[str] = Field(description="List of action items mentioned")
    deadline: str | None = Field(description="Deadline if mentioned", default=None)
    priority: Literal["high", "medium", "low"] = Field(description="Priority level")

email_text = """
From: Mike Johnson
Subject: Urgent - API Integration Issues

We need to fix the authentication bug by end of day tomorrow.
Please also update the documentation and notify the client.
"""

response = client.beta.chat.completions.parse(
    model="gpt-4o-mini",
    messages=[
        {"role": "user", "content": f"Extract key information: {email_text}"}
    ],
    response_format=EmailData
)

email = response.choices[0].message.parsed

print(email.sender)        # "Mike Johnson"
print(email.priority)      # "high"
print(email.action_items)  # ["Fix authentication bug", "Update documentation", ...]
```

### Example 2: Sentiment Analysis with Classification

Use `Literal` types to force classification into specific categories:

```python
from pydantic import BaseModel, Field
from typing import Literal

class SentimentAnalysis(BaseModel):
    sentiment: Literal["positive", "negative", "neutral"]
    confidence: float = Field(ge=0, le=1, description="Confidence score 0-1")
    key_phrases: list[str] = Field(description="Key phrases that influenced sentiment")
    emotion: Literal["joy", "anger", "sadness", "fear", "surprise", "neutral"]

review = "This product is absolutely amazing! Best purchase ever!"

response = client.beta.chat.completions.parse(
    model="gpt-4o-mini",
    messages=[
        {"role": "user", "content": f"Analyze sentiment: {review}"}
    ],
    response_format=SentimentAnalysis
)

analysis = response.choices[0].message.parsed

print(analysis.sentiment)    # "positive"
print(analysis.confidence)   # 0.95
print(analysis.emotion)      # "joy"
```

The LLM must choose from your predefined categories. No more getting back "super positive" when you expected "positive".

### Example 3: Complex Nested Structures

Handle nested data with multiple Pydantic models:

```python
class Address(BaseModel):
    street: str
    city: str
    country: str

class Contact(BaseModel):
    phone: str
    email: str

class Company(BaseModel):
    name: str = Field(description="Company name")
    address: Address = Field(description="Company address")
    contact: Contact = Field(description="Contact information")
    employees: int = Field(description="Number of employees", gt=0)

company_text = """
TechCorp is a software company with 150 employees.
Located at 123 Main Street, San Francisco, USA.
Contact: info@techcorp.com or 415-555-0123.
"""

response = client.beta.chat.completions.parse(
    model="gpt-4o-mini",
    messages=[
        {"role": "user", "content": f"Extract company info: {company_text}"}
    ],
    response_format=Company
)

company = response.choices[0].message.parsed

print(company.name)                 # "TechCorp"
print(company.address.city)         # "San Francisco"
print(company.contact.email)        # "info@techcorp.com"
print(company.employees)            # 150
```

### Example 4: Validation with Pydantic

Pydantic validates data automatically using Field constraints:

```python
from pydantic import BaseModel, Field, EmailStr

class User(BaseModel):
    name: str = Field(min_length=1, max_length=100, description="User's full name")
    email: EmailStr = Field(description="Valid email address")
    age: int = Field(gt=0, lt=150, description="User age")
    role: Literal["admin", "user", "guest"] = Field(description="User role")

# The LLM will return data matching these constraints:
# - name: 1-100 characters
# - email: valid email format
# - age: between 0 and 150
# - role: one of the three allowed values
```

If the LLM tries to return invalid data, the request fails and retries automatically.

## Running the Example

The `example.py` file demonstrates all seven structured output patterns:

```bash
cd 06-structured-output
uv run example.py
```

You'll see examples of:
1. Basic structured output (calendar events)
2. Email data extraction
3. Validation in action
4. Nested data structures
5. Sentiment analysis
6. Code review analysis
7. Optional fields handling

## Key Takeaways

1. **Use Field() descriptions**: The LLM uses these to understand what you want. "Date in YYYY-MM-DD format" is better than just "date".

2. **Use Literal types for classification**: Force the LLM to choose from specific categories instead of making up its own.

3. **Optional fields use `| None`**: Use `field: str | None = None` for optional data the LLM might not find.

4. **Validation is automatic**: Pydantic validates types, constraints, and formats without extra code.

5. **Keep schemas simple**: Start with 3-5 fields. Add complexity only when needed. Complex schemas can confuse the LLM.

6. **Nested models work**: Use separate Pydantic models for nested structures like addresses or contacts.

7. **Type safety everywhere**: Get IDE autocomplete, type checking, and compile-time safety for free.

## Common Pitfalls

1. **Over-complicated schemas**: Don't nest 5 levels deep. Keep it simple, especially when learning. The LLM struggles with complex structures.

2. **Missing Field() descriptions**: The LLM needs context. "email" alone is vague. "User's email address in standard format" is clear.

3. **Forgetting optional fields**: If a field might not exist in the input, mark it optional with `| None`. Otherwise the LLM will make up data.

4. **Unclear field names**: Use descriptive names like `user_email` not `e`. The field name is documentation.

5. **Not testing edge cases**: Test with incomplete data, ambiguous inputs, and missing information to see how the LLM handles it.

6. **Wrong use case**: Don't use structured output for creative writing or explanations. Use it for data extraction and classification.

## When to Use Structured Output

| Use Case | Use Structured Output? |
|----------|----------------------|
| Extract emails, dates, names from text | ✅ Yes |
| Classify sentiment, categories, intent | ✅ Yes |
| Parse invoices, receipts, documents | ✅ Yes |
| Generate forms, surveys, data entry | ✅ Yes |
| Creative writing or explanations | ❌ No |
| Code generation | ❌ No (use plain text) |
| Chat conversations | ❌ No (unless extracting data) |

Use structured output when you need reliable, typed data. Use plain text when you need creativity and flexibility.

## Real-World Impact

Structured output is how production AI systems work. Here's what it enables:

**Data Extraction**: Process thousands of invoices, receipts, or contracts automatically with validated output you can load directly into databases.

**Classification Systems**: Build content moderation, sentiment analysis, or intent detection that returns consistent, typed results.

**Form Generation**: Extract information from conversations or documents and populate forms automatically with validated data.

**API Reliability**: Return consistent JSON responses from AI endpoints that frontend developers can depend on.

**Quality Assurance**: Eliminate parsing errors, type mismatches, and format inconsistencies that plague text-based systems.

Companies use structured output to process thousands of documents daily, classify millions of messages, and extract data at scale—all with type safety and validation.

## Assignment

Build a job posting parser that extracts structured information from unstructured job descriptions.

Create a Pydantic model with these fields:
- `title`: Job title (required)
- `company`: Company name (required)
- `location`: Job location (optional, None if remote)
- `salary_min` and `salary_max`: Salary range (optional)
- `remote`: Boolean indicating if remote
- `required_skills`: List of required skills (required)

Test it with at least 3 different job postings that include varying levels of information. Handle cases where salary isn't mentioned or location is ambiguous.

Bonus: Add validation constraints (salary must be positive, skills list can't be empty, etc.)

## Next Steps

You've mastered structured output for data extraction and classification. Now it's time to give your AI the ability to take actions in the real world.

Move to [07-tool-calling-basics](/Users/owainlewis/Projects/the-ai-engineer/ai-agents-from-scratch/07-tool-calling-basics) to learn how to let the LLM call functions, query APIs, and interact with external systems.

## Resources

- [Pydantic Documentation](https://docs.pydantic.dev) - Complete guide to Pydantic models and validation
- [OpenAI Structured Outputs Guide](https://platform.openai.com/docs/guides/structured-outputs) - Official documentation for structured output
- [JSON Schema](https://json-schema.org/) - Understanding the schema format under the hood
- [Python Type Hints](https://docs.python.org/3/library/typing.html) - Master Python typing for better schemas
