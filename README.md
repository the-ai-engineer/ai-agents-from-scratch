# Build AI Agents from First Principles

This is a short course that shows how to build AI agents and AI systems from scratch in pure Python.

Why learn from scratch?

When you understand first principles, you can:
- Build most AI systems without needing frameworks (just simple code)
- Build with **any** framework (or create your own)
- Build deeper understanding of how things actually work

## Course Structure

### Foundations (1-4)
1. **[01-api-basics](./01-api-basics)** - Master making API calls to LLM providers
2. **[02-conversation-memory](./02-conversation-memory)** - Maintaining context across turns
3. **[03-structured-output](./03-structured-output)** - Type-safe responses with Pydantic
4. **[04-tool-calling](./04-tool-calling)** - Give AI the ability to do real work

### Workflows & Agents (5-6)
5. **[05-workflow-patterns](./05-workflow-patterns)** - 5 fundamental patterns (chaining, routing, parallelization, orchestrator-workers, evaluator-optimizer) 📊
6. **[06-agent-architecture](./06-agent-architecture)** - Agent loop & reusable Agent class (loop.py and class.py)

### Advanced (7-8)
7. **[07-advanced-memory](./07-advanced-memory)** - Token limits, trimming, Redis persistence, and managed alternatives (mem0)

## 🚀 Getting Started

### Prerequisites
- Python 3
- [UV package manager](https://docs.astral.sh/uv/) (recommended) or pip
- OpenAI API key
- Basic Python knowledge (functions, classes, type hints)

### Installation

This repository uses UV for fast, reliable dependency management. Each lesson is self-contained with its own isolated environment.

#### Installing UV

```bash
# macOS/Linux
curl -LsSf https://astral.sh/uv/install.sh | sh
```

#### Set up your API key

```bash
# Create .env file in the root directory
echo "OPENAI_API_KEY=your-key-here" > .env
```

## 💡 Learning Path

### If you're brand new to AI:
Start at lesson 01 and work through each lesson in order.

### If you know the basics:
- Skip to lesson 04 for tool calling
- Jump to lesson 05 for workflow patterns
- Jump to lesson 06 for agent architecture with reusable Agent class

## Contributing

Found an issue or want to improve an example? PRs welcome!

## License

MIT License - feel free to use this code for learning and in your own projects.
