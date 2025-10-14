"""
Lesson 02: Prompt Engineering - Practical Examples

Demonstrates the six core prompting techniques for reliable outputs.

Run from project root:
    uv run python 02-prompt-engineering/examples.py
"""

import os
from openai import OpenAI
from dotenv import load_dotenv

load_dotenv()
client = OpenAI(api_key=os.getenv("OPENAI_API_KEY"))


##=================================================##
## Example 1: Few-Shot Learning
##=================================================##

# Wrong approach - vague instructions
vague_prompt = "Write a friendly welcome email for new customers. Make it warm but professional."

# Better approach - show examples
few_shot_prompt = """Write a welcome email following these examples:

Example 1:
Subject: Welcome to Acme!
Hey Sarah, we're excited to have you. Your account is ready at dashboard.acme.com.
Reply to this email if you need anything.

Example 2:
Subject: You're all set!
Hi Marcus, thanks for signing up. Log in at dashboard.acme.com to get started.
Questions? Just reply.

Now write one for: Jessica, signed up for Pro plan"""

response = client.responses.create(
    model="gpt-4o-mini",
    input=few_shot_prompt,
    temperature=0
)

# response.output_text


##=================================================##
## Example 2: Clear Delimiters
##=================================================##

article_text = """
Artificial intelligence is transforming how businesses operate.
Companies are adopting AI for customer service, data analysis, and automation.
However, challenges remain around data privacy and implementation costs.
"""

delimited_prompt = f"""<instructions>
Summarize the article in 100 words or less.
Output JSON matching this schema: {{"summary": "text"}}
</instructions>

<article>
{article_text}
</article>

<output_format>
{{"summary": "<your summary here>"}}
</output_format>"""

response = client.responses.create(
    model="gpt-4o-mini",
    input=delimited_prompt,
    temperature=0
)

# response.output_text


##=================================================##
## Example 3: Role and Constraints
##=================================================##

role_prompt = """You are a software engineer writing a root cause analysis (RCA).

Constraints:
- Input will contain logs, metrics, and incident notes
- Output must be a single sentence root cause summary
- Use clear, technical language (no speculation)
- If evidence is missing, output {"rca": "insufficient data"}

Input:
<incident_data>
2024-10-14 12:34:56 ERROR Database connection timeout after 30s
2024-10-14 12:34:57 INFO Retry attempt 1/3 failed
2024-10-14 12:35:00 ERROR Max retries exceeded
</incident_data>

Output:
Return only valid JSON matching this schema:
{
  "rca": "string"
}"""

response = client.responses.create(
    model="gpt-4o-mini",
    input=role_prompt,
    temperature=0
)

# response.output_text


##=================================================##
## Example 4: Control Output Format
##=================================================##

format_prompt = """Summarize the root cause in one sentence.

Return only valid JSON using this exact schema:
{
  "incident_id": "string",
  "rca_summary": "string",
  "severity": "low | medium | high | critical",
  "timestamp": "ISO-8601 datetime"
}

Incident: Database connection pool exhausted during peak traffic at 2PM."""

response = client.responses.create(
    model="gpt-4o-mini",
    input=format_prompt,
    temperature=0
)

# response.output_text


##=================================================##
## Example 5: Temperature Control
##=================================================##

deterministic_prompt = "Extract the email address from: Contact John at john@example.com for details"

# Deterministic (temperature=0) for consistent outputs
response_deterministic = client.responses.create(
    model="gpt-4o-mini",
    input=deterministic_prompt,
    temperature=0  # Always same output
)

# response_deterministic.output_text

# Creative (temperature=0.8) for varied outputs
creative_prompt = "Write a creative tagline for an AI startup"

response_creative = client.responses.create(
    model="gpt-4o-mini",
    input=creative_prompt,
    temperature=0.8  # More varied, creative
)

# response_creative.output_text


##=================================================##
## Example 6: Grounding in Context (RAG Preview)
##=================================================##

context = """
Our refund policy:
- Digital products: 30-day money-back guarantee
- Physical products: 60-day return window
- Opened software: No refunds due to licensing
- Shipping costs: Non-refundable
"""

rag_prompt = f"""<instructions>
Answer the question using only information from the provided context.
If the answer is not in the context, say "I don't have that information."
Cite the specific section you used.
</instructions>

<context>
{context}
</context>

<question>
What is the refund policy for digital products?
</question>"""

response = client.responses.create(
    model="gpt-4o-mini",
    input=rag_prompt,
    temperature=0
)

# response.output_text


##=================================================##
## Example 7: Meta-Prompting (Basic)
##=================================================##

original_prompt = "Summarize customer feedback and list key issues"

meta_prompt = f"""Analyze this prompt and suggest specific improvements:

<prompt>
{original_prompt}
</prompt>

Focus on:
- Clarity of instructions
- Completeness of requirements
- Structured output format
- Potential edge cases"""

response = client.responses.create(
    model="gpt-4o-mini",
    input=meta_prompt,
    temperature=0
)

# response.output_text
