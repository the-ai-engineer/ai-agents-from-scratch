# Lesson 02: Prompt Engineering That Gets Results

## What You'll Learn

In this lesson, you'll learn how to write prompts that get consistent, high-quality results from LLMs.

Prompt engineering is the skill that separates AI hobbyists from professionals. Anyone can make an API call. But getting reliable, production-quality results requires knowing how to communicate precisely with language models.

By the end of this lesson, you'll master system prompts that control behavior, few-shot learning that teaches patterns with examples, chain-of-thought prompting for complex reasoning, and reusable prompt templates. You'll also know when prompting is enough and when you need fine-tuning.

This skill directly impacts your system's reliability. Poor prompts give inconsistent results that break in production. Good prompts give predictable, trustworthy outputs you can build systems on.

## The Problem

Language models are powerful, but they're not mind readers. Vague instructions get vague results. Asking "help me with this code" could mean anything—review it? Debug it? Refactor it? The model guesses, and you get unpredictable output.

In production, unpredictability is unacceptable. You need outputs formatted consistently, edge cases handled gracefully, and behavior that doesn't surprise users. That requires intentional prompt engineering.

## How Prompt Engineering Works

Think of prompts as programming the AI's behavior. Just like you write code to control software, you write prompts to control language models.

The system message is your most powerful tool. It sets the assistant's role, expertise, output format, constraints, and examples. Get the system prompt right and everything else becomes easier.

### The Anatomy of a Good Prompt

Effective prompts follow this structure:

1. **Role**: Define who the assistant is (e.g., "You are an expert Python developer")
2. **Task**: State clearly what it should do (e.g., "Review code for bugs and security issues")
3. **Format**: Specify exactly how to format output (e.g., "Provide numbered list with severity ratings")
4. **Examples**: Show desired behavior (optional but powerful)
5. **Constraints**: Define what to avoid (e.g., "Do not suggest libraries not in Python's standard lib")

Following this structure eliminates ambiguity and dramatically improves consistency.

## Pattern 1: Clear, Specific Instructions

The difference between mediocre and excellent results often comes down to specificity.

Bad prompt:
```python
"Help me with this code"
```

Good prompt:
```python
"""You are an expert code reviewer specializing in Python.

Review the following code for:
1. Bugs and logic errors
2. Security vulnerabilities
3. Performance issues
4. Code style problems

Format your response as a numbered list. For each issue:
- Describe the problem
- Explain the risk
- Suggest a fix

If no issues found, respond with: "No issues detected."
"""
```

The second prompt leaves no room for confusion. The model knows exactly what to analyze, what to look for, and how to format results.

## Pattern 2: Few-Shot Learning with Examples

Sometimes showing is better than telling. Few-shot learning teaches patterns through 2-5 examples.

```python
system_prompt = """You are a sentiment classifier.

Examples:

Input: "I love this product! Best purchase ever."
Output: positive

Input: "This is completely broken and unusable."
Output: negative

Input: "It works fine, nothing special."
Output: neutral

Now classify the following input."""
```

The model learns the pattern from examples and applies it consistently. This works exceptionally well for classification, extraction, and formatting tasks.

Keep examples to 3-5. More than that and you're wasting tokens. Fewer and the pattern might not be clear.

## Pattern 3: Chain-of-Thought Prompting

For complex reasoning tasks, ask the model to think step-by-step. This dramatically improves accuracy on multi-step problems.

```python
user_prompt = """Let's solve this step by step:

1. First, analyze the problem
2. Then, identify possible solutions
3. Evaluate each solution
4. Finally, recommend the best approach

Problem: Our API is timing out under load. We're seeing 30% of requests fail when traffic exceeds 1000 req/sec."""
```

Chain-of-thought prompting forces the model to reason explicitly rather than jumping to conclusions. Use this for debugging, planning, strategy, and any task requiring logical reasoning.

## Pattern 4: Output Format Control

Controlling output format is critical for production systems that parse responses.

```python
system_prompt = """Extract contact information from text.

Output format (JSON):
{
  "name": "John Doe",
  "email": "john@example.com",
  "phone": "+1-555-0123"
}

If a field is not found, use null.
Always return valid JSON."""
```

Specifying format in the system prompt ensures consistent structure. For even stronger guarantees, you'll use structured outputs in the next lesson.

## Pattern 5: Reusable Prompt Templates

Production systems use templates with variables for reusability:

```python
def create_code_review_prompt(language: str, focus_areas: list[str]) -> str:
    areas = "\n".join(f"- {area}" for area in focus_areas)

    return f"""You are an expert {language} code reviewer.

Review the following code focusing on:
{areas}

Provide a numbered list of issues with severity levels (LOW, MEDIUM, HIGH).
If no issues found, respond: "Code review passed."
"""

# Usage
prompt = create_code_review_prompt(
    language="Python",
    focus_areas=["Security vulnerabilities", "Performance issues", "Type safety"]
)
```

Templates make your prompts consistent, maintainable, and easy to improve over time.

## When Prompting Isn't Enough

Prompting works for most use cases, but sometimes you need fine-tuning:

- **Custom behavior not achievable with prompts**: You need responses in a completely unique style
- **Reducing token usage**: Fine-tuned models can handle shorter prompts
- **Proprietary knowledge**: You're training on company-specific data

For 95% of applications, prompting is sufficient. Don't fine-tune unless you've exhausted prompt engineering.

## Running the Example Code

This lesson includes practical examples of all these patterns:

```bash
cd 02-prompting
uv run example.py
```

The examples show sentiment analysis with few-shot learning, code review with structured output, and chain-of-thought reasoning.

## Key Takeaways

1. **Be specific, not vague.** "Review this code for bugs" beats "help with code."
2. **Use 3-5 examples to teach patterns.** Few-shot learning is powerful and token-efficient.
3. **Specify output format explicitly.** Don't make the model guess how to structure responses.
4. **Break complex tasks into steps.** Chain-of-thought improves reasoning dramatically.
5. **Test edge cases thoroughly.** Your prompt should handle empty inputs, weird formatting, and unexpected content.

## Common Pitfalls

1. **Too vague**: "Be helpful" tells the model nothing. Specific instructions get specific results.
2. **Conflicting instructions**: Don't say "be concise" then ask for "detailed explanations."
3. **Too many examples**: More than 5-7 examples wastes tokens without improving quality.
4. **Not testing edge cases**: What happens with empty input? Very long input? Special characters?
5. **Prompt injection vulnerabilities**: Users can potentially override your system prompt. In production, validate and sanitize user inputs.

## Real-World Impact

Companies that master prompt engineering save thousands in API costs, reduce error rates by 50-80%, and ship AI features faster.

Poor prompts mean unpredictable outputs, endless debugging, and users losing trust in your system. Good prompts mean reliable features you can deploy confidently.

Every advanced pattern you'll learn—RAG systems, tool calling, autonomous agents—depends on strong prompt engineering. This skill compounds throughout your career.

## Assignment

Take the following vague prompt and improve it using what you learned:

**Vague prompt**: "Analyze this customer review"

Rewrite it following the structure:
1. Define the role
2. State the specific task
3. Specify output format
4. Add 2-3 examples
5. Define constraints

Then test your improved prompt with 3-5 different customer reviews and verify you get consistent, useful output.

## Next Steps

Now that you can write effective prompts, move to [Lesson 03 - Structured Output](../03-structured-output) to learn how to get type-safe, validated responses with guaranteed formats.

## Resources

- [OpenAI Prompt Engineering Guide](https://platform.openai.com/docs/guides/prompt-engineering)
- [Learn Prompting](https://learnprompting.org/)
- [Prompt Engineering Guide by DAIR.AI](https://www.promptingguide.ai/)
