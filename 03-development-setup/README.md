# Development Setup: Getting Ready to Build

## What You'll Learn

This is a hands-on course. You'll write code and build real projects, so let's set up a clean environment for AI development in Python.

By the end of this lesson, you'll have:
- Python 3.10+ installed
- UV for dependency management
- A code editor configured
- Your OpenAI API key set up
- Understanding of tokenization and costs

We'll assume you're on macOS. If you're using Windows or Linux, the same tools apply, though installation steps may differ slightly.

## 1. Install Python

We use Python 3.10+ throughout this course. Check if you already have it:

```bash
python3 --version
```

If it's missing or outdated, download from [python.org](https://www.python.org/) and install.

Once installed, verify:

```bash
python3 --version
# Should show 3.10 or higher
```

## 2. Choose a Code Editor

You need a reliable editor with Python and AI tooling support.

### Recommended Options

**VS Code** - Lightweight, extensible, great all-rounder
- Download: https://code.visualstudio.com/
- Install Python extension from marketplace

**Cursor** - Editor built with AI integration
- Download: https://cursor.com/
- Built on VS Code with AI features

Pick whichever you're most comfortable with.

## 3. Install UV (Python Project Manager)

UV is a fast, modern Python project manager. It handles dependencies and virtual environments better than pip + venv.

### Install UV

```bash
curl -LsSf https://astral.sh/uv/install.sh | sh
```

Or visit: https://docs.astral.sh/uv/getting-started/installation/

### Verify Installation

```bash
uv --version
```

### Common UV Workflow

```bash
# Create a new project
uv init my-ai-project
cd my-ai-project

# Create virtual environment
uv venv
source .venv/bin/activate  # macOS/Linux
# or: .venv\Scripts\activate  # Windows

# Add dependencies
uv add openai
uv add python-dotenv

# Sync dependencies
uv sync

# Run your code
uv run python main.py
```

Throughout this course, you'll use UV to manage dependencies in each lesson folder.

## 4. Get Your OpenAI API Key

### Create Account and Key

1. Go to [platform.openai.com](https://platform.openai.com/)
2. Sign up or log in
3. Navigate to API keys section
4. Click "Create new secret key"
5. Copy the key immediately (you can't see it again)

### Add Billing

OpenAI requires a payment method on file. Add a credit card to your account and set a reasonable usage limit (e.g., $10-20 for learning).

### Cost Management

- Set usage limits in your OpenAI dashboard
- Start with small limits while learning
- Monitor usage regularly
- Use `gpt-4o-mini` (cheap) instead of `gpt-4` (expensive) for development

## 5. Configure Environment Variables

Never hardcode API keys in your code. Use environment variables.

### Create .env File

In each project folder, create a `.env` file:

```bash
OPENAI_API_KEY=sk-your-actual-key-here
```

Add `.env` to your `.gitignore`:

```bash
echo ".env" >> .gitignore
```

### Load Environment Variables in Python

Install python-dotenv:

```bash
uv add python-dotenv
```

Use in your code:

```python
import os
from dotenv import load_dotenv
from openai import OpenAI

load_dotenv()  # Loads .env file

client = OpenAI(api_key=os.getenv("OPENAI_API_KEY"))
```

## 6. Understanding Tokenization

Before you start making API calls, you need to understand how LLMs process text and how you're charged.

### What Are Tokens?

When you send text to an LLM, your human-readable text gets broken down into **tokens** - the basic units the model processes.

Sometimes one word equals one token, but often it's split into smaller pieces.

### Examples

- "Hello world" → 2 tokens: "Hello" and " world"
- "tokenization" → 2 tokens: "token" and "ization"
- "AI" → 1 token
- "ChatGPT" → 2 tokens: "Chat" and "GPT"

The model learns relationships between token sequences, not individual letters or whole words. That's why LLMs sometimes struggle with tasks like counting letters - they don't "see" text the same way humans do.

### Types of Tokens

**Input tokens**: Everything you send to the model (prompts, context, conversation history)

**Output tokens**: What the model generates in response

### Why Tokenization Matters

Providers charge by tokens, not words:
- Input and output tokens often have different prices
- Longer prompts and responses cost more
- Token limits can cause truncation
- Understanding tokenization helps you optimize costs

### Counting Tokens in Python

OpenAI provides the `tiktoken` library for exact token counting:

```bash
uv add tiktoken
```

Example usage:

```python
import tiktoken

def count_tokens(text: str, model: str = "gpt-4o-mini") -> int:
    """Count tokens for a given text and model."""
    encoding = tiktoken.encoding_for_model(model)
    return len(encoding.encode(text))

# Example
text = "Hello, how are you today?"
tokens = count_tokens(text)
print(f"Token count: {tokens}")
```

### Estimating Costs

With token counts, you can estimate API costs:

```python
def estimate_cost(input_tokens: int, output_tokens: int, model: str = "gpt-4o-mini") -> float:
    """
    Estimate cost for a request.

    Prices as of 2024 (check platform.openai.com for current rates):
    - gpt-4o-mini: $0.15 per 1M input, $0.60 per 1M output
    - gpt-4o: $2.50 per 1M input, $10.00 per 1M output
    """
    if model == "gpt-4o-mini":
        input_cost = (input_tokens / 1_000_000) * 0.15
        output_cost = (output_tokens / 1_000_000) * 0.60
    elif model == "gpt-4o":
        input_cost = (input_tokens / 1_000_000) * 2.50
        output_cost = (output_tokens / 1_000_000) * 10.00
    else:
        return 0.0

    return input_cost + output_cost

# Example
cost = estimate_cost(input_tokens=100, output_tokens=50, model="gpt-4o-mini")
print(f"Estimated cost: ${cost:.6f}")
```

### Tools for Tokenization

**OpenAI Tokenizer**: Visual tool to see how text is tokenized
- https://platform.openai.com/tokenizer

Use this to understand how your prompts are being processed and optimize for token usage.

## 7. AI Coding Assistants (Optional)

AI-assisted coding can speed you up significantly while learning. These are optional but helpful.

### Options

**Claude Code** - Terminal-based AI assistant
- https://claude.ai/code
- Great for command-line workflows

**Cursor** - AI-native editor
- https://cursor.com/
- Built-in AI pair programming

**GitHub Copilot** - AI code completion
- Works with VS Code
- Auto-completes code as you type

You don't need these tools to complete the course, but they'll help you move faster.

## 8. Verify Your Setup

Create a test script to verify everything works:

```python
# test_setup.py
import os
from dotenv import load_dotenv
from openai import OpenAI
import tiktoken

# Load environment variables
load_dotenv()

# Initialize client
client = OpenAI(api_key=os.getenv("OPENAI_API_KEY"))

# Test API connection
print("Testing OpenAI API connection...")
try:
    response = client.responses.create(
        model="gpt-4o-mini",
        input="Say 'Hello, AI Engineer!' if you can read this.",
        temperature=0
    )
    print(f"✓ API working: {response.output_text}")
except Exception as e:
    print(f"✗ API error: {e}")

# Test tokenization
print("\nTesting tokenization...")
text = "This is a test message for tokenization."
encoding = tiktoken.encoding_for_model("gpt-4o-mini")
tokens = len(encoding.encode(text))
print(f"✓ Tokenization working: '{text}' = {tokens} tokens")

print("\n✓ Setup complete! You're ready to build.")
```

Run it:

```bash
uv run python test_setup.py
```

If you see success messages, your environment is ready.

## Troubleshooting

### API Key Not Found

If you get `OPENAI_API_KEY not set`:
- Check `.env` file exists
- Check `load_dotenv()` is called before using `OpenAI()`
- Check `.env` format: `OPENAI_API_KEY=sk-...` (no quotes)

### Import Errors

If you get `ModuleNotFoundError`:
- Make sure virtual environment is activated
- Run `uv sync` to install dependencies
- Check you're in the correct directory

### Permission Errors

If UV installation fails:
- Try with `sudo` (Unix/Linux)
- Check you have write permissions
- Refer to UV documentation for platform-specific issues

## Key Takeaway

Your development environment is now configured for AI engineering:
- Python 3.10+ for running code
- UV for dependency management
- OpenAI API key for model access
- Environment variables for security
- Tokenization understanding for cost management

This setup will serve you throughout the course and in production projects.

## Next Steps

With your environment ready, the next lesson makes your first API call and teaches you to interact with LLMs programmatically.

## Resources

- [Python Installation Guide](https://docs.python.org/3/using/index.html)
- [UV Documentation](https://docs.astral.sh/uv/)
- [OpenAI Platform](https://platform.openai.com/)
- [Tiktoken Library](https://github.com/openai/tiktoken)
- [OpenAI Tokenizer Tool](https://platform.openai.com/tokenizer)
