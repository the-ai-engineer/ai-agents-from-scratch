# Build AI Agents from First Principles

This is a comprehensive course that shows how to build AI agents and AI systems from scratch in pure Python.

Why learn from scratch?

When you understand first principles, you can:
- Build most AI systems without needing frameworks (just simple code)
- Build with **any** framework (or create your own)
- Build deeper understanding of how things actually work

## Course Structure

### Getting Started (00-03)

**[00-introduction](./00-introduction)** - Course overview, philosophy, and what you'll build

**[01-fundamentals](./01-fundamentals)** - What LLMs are, how they work, and how to choose models

**[02-prompt-engineering](./02-prompt-engineering)** - Techniques for reliable outputs: few-shot learning, delimiters, output formatting, meta-prompting

**[03-development-setup](./03-development-setup)** - Python setup, UV, API keys, tokenization, and cost management

### Foundations (04-06)

**[04-api-basics](./04-api-basics)** - Making API calls, streaming responses, async operations

**[05-conversation-memory](./05-conversation-memory)** - Maintaining context across multiple turns (automatic and manual)

**[06-structured-output](./06-structured-output)** - Type-safe responses with Pydantic validation

### Tool Calling & Workflows (07-08)

**[07-tool-calling](./07-tool-calling)** - Give AI the ability to call functions and take actions (manual schemas → @tool decorator)

**[08-workflow-patterns](./08-workflow-patterns)** - Five fundamental patterns: chaining, routing, parallelization, orchestrator-workers, evaluator-optimizer

### Autonomous Agents (09-10)

**[09-agent-architecture](./09-agent-architecture)** - Building agents from scratch: the agent loop pattern and reusable Agent class

**[10-advanced-memory](./10-advanced-memory)** - Production memory management: token limits, trimming strategies, Redis persistence, mem0

## 🚀 Getting Started

### Prerequisites

- Python 3.10+
- [UV package manager](https://docs.astral.sh/uv/) (recommended) or pip
- OpenAI API key from [platform.openai.com](https://platform.openai.com)
- Basic Python knowledge (functions, classes, type hints)

### Installation

This repository uses UV for fast, reliable dependency management. Each lesson is self-contained.

#### Install UV

```bash
# macOS/Linux
curl -LsSf https://astral.sh/uv/install.sh | sh

# Windows
powershell -c "irm https://astral.sh/uv/install.ps1 | iex"
```

Or visit: https://docs.astral.sh/uv/getting-started/installation/

#### Set Up Your API Key

```bash
# Create .env file in the root directory
echo "OPENAI_API_KEY=your-key-here" > .env
```

**Important:** Add `.env` to your `.gitignore` to avoid committing your API key.

#### Verify Setup

```bash
# Navigate to setup lesson
cd 03-development-setup

# Run test script
uv run python test_setup.py
```

If all tests pass, you're ready to build!

## 💡 Learning Path

### If you're brand new to AI

Start at **[00-introduction](./00-introduction)** and work through each lesson in order. The course builds progressively from concepts to code to production patterns.

### If you know LLM basics

- Skip to **[04-api-basics](./04-api-basics)** to start coding
- Jump to **[07-tool-calling](./07-tool-calling)** for function calling
- Start at **[08-workflow-patterns](./08-workflow-patterns)** for orchestration patterns
- Begin with **[09-agent-architecture](./09-agent-architecture)** for autonomous agents

### If you've used frameworks before

Start at **[07-tool-calling](./07-tool-calling)** to see tool calling from first principles, then move to **[09-agent-architecture](./09-agent-architecture)** to understand the agent loop that powers every framework.

## 🎯 What You'll Build

By the end of this course, you'll have built:

- **Conversational AI** with memory management
- **Structured data extraction** with type validation
- **Tool-calling systems** that execute functions
- **Workflow orchestrations** with 5 fundamental patterns
- **Autonomous agents** with the agent loop
- **Production-ready systems** with proper memory management

## 📖 Course Philosophy

This course teaches first principles:

1. **Manual before automatic** - Understand the raw format before using abstractions
2. **Build before using frameworks** - Know what LangChain/AutoGPT are doing under the hood
3. **Production patterns** - Learn techniques that work in real systems
4. **Type safety** - Use Python type hints and Pydantic for reliability

Every abstraction is earned through understanding.

## 🛠️ Running Examples

Each lesson includes runnable code examples:

```bash
# Navigate to any lesson
cd 04-api-basics

# Run examples with UV
uv run python 01-basic.py

# Or activate virtual environment first
uv venv
source .venv/bin/activate  # macOS/Linux
# .venv\Scripts\activate  # Windows
python 01-basic.py
```

## 📚 Additional Resources

- [OpenAI API Documentation](https://platform.openai.com/docs)
- [Anthropic Claude Documentation](https://docs.anthropic.com)
- [UV Documentation](https://docs.astral.sh/uv/)
- [Pydantic Documentation](https://docs.pydantic.dev)

## 🤝 Contributing

Found an issue or want to improve an example? PRs welcome!

## 📄 License

MIT License - feel free to use this code for learning and in your own projects.

---

**Ready to start?** Head to [00-introduction](./00-introduction) to begin your journey to AI engineering mastery.
