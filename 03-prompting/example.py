"""
Lesson 02: Prompt Engineering

Learn how to write effective prompts that get consistent, high-quality results.
"""

import os
from openai import OpenAI
from dotenv import load_dotenv

load_dotenv()
client = OpenAI(api_key=os.getenv("OPENAI_API_KEY"))


def basic_vs_detailed_prompt():
    """Example 1: Vague vs specific prompts"""
    code = """
def calculate(x, y):
    return x / y
"""

    # Vague prompt
    print("Vague prompt: 'Review this code'")
    response = client.responses.create(
        model="gpt-4o-mini",
        input=f"Review this code:\n{code}",
        temperature=0
    )
    print(response.output_text)

    # Specific prompt with structure
    print("\n\nSpecific prompt with structure:")
    system_prompt = """You are a code review assistant.

Instructions:
1. Identify bugs and security issues
2. Rate severity: CRITICAL, HIGH, MEDIUM, LOW
3. Format as: [SEVERITY] Issue description"""

    response = client.responses.create(
        model="gpt-4o-mini",
        instructions=system_prompt,
        input=f"Review this code:\n{code}",
        temperature=0
    )
    print(response.output_text)


def few_shot_learning():
    """Example 2: Few-shot learning for consistent output"""
    system_prompt = """You are a sentiment analyzer. Respond with ONLY: positive, negative, or neutral.

Examples:
Input: "I absolutely love this product!"
Output: positive

Input: "This is terrible. Waste of money."
Output: negative

Input: "It's okay. Nothing special."
Output: neutral
"""

    test_inputs = [
        "This movie was amazing! 10/10",
        "Meh, could be better",
        "Worst experience ever."
    ]

    print("\nSentiment classification:")
    for text in test_inputs:
        response = client.responses.create(
            model="gpt-4o-mini",
            instructions=system_prompt,
            input=text,
            temperature=0
        )
        sentiment = response.output_text.strip()
        print(f"  '{text}' → {sentiment}")


def chain_of_thought():
    """Example 3: Chain-of-thought for complex reasoning"""
    problem = """A company has 100 employees. 60% are engineers, and 40% of
the engineers are senior engineers. How many senior engineers are there?"""

    print("\nWithout chain-of-thought:")
    response = client.responses.create(
        model="gpt-4o-mini",
        input=problem,
        temperature=0
    )
    print(response.output_text)

    print("\n\nWith chain-of-thought:")
    cot_prompt = f"{problem}\n\nLet's solve this step by step:"

    response = client.responses.create(
        model="gpt-4o-mini",
        input=cot_prompt,
        temperature=0
    )
    print(response.output_text)


if __name__ == "__main__":
    basic_vs_detailed_prompt()
    few_shot_learning()
    chain_of_thought()
