# Building AI Agents from Scratch 

A comprehensive, step-by-step guide to building production-ready AI agents using Python. No frameworks—just fundamentals.

## Why? 

Because so much of the online content focusses on frameworks which hide much of the details and often add complexity and overwhelm. This course focusses on the basics in simple Python so you can learn without getting lost in framework hell (spending more time learning frameworks than the concepts).

## 🎯 What You'll Learn

- How to call the OpenAI API and work with chat completions
- Prompt engineering techniques that actually work
- Structured output with Pydantic for type-safe responses
- Tool calling: giving AI the ability to take actions
- Building production-ready tools with validation
- Workflow patterns: chaining, routing, and parallelization
- Multi-step agent reasoning loops
- Creating reusable Agent classes
- Managing conversation memory and state
- ReAct pattern for planning and reasoning
- Deploying agents with FastAPI
- Building complete, production-ready AI agents

## 📚 Course Structure

### Foundations
1. **[01-api-basics](./01-api-basics)** - Your first API call to OpenAI
2. **[02-prompting](./02-prompting)** - Prompt engineering essentials
3. **[03-structured-output](./03-structured-output)** - Type-safe responses with Pydantic

### Tool Calling
4. **[04-tool-calling-basics](./04-tool-calling-basics)** - Give AI the ability to use tools
5. **[05-tool-calling-pydantic](./05-tool-calling-pydantic)** - Production tool calling with validation

### Workflow Patterns
6. **[06-workflow-patterns](./06-workflow-patterns)** - Chaining, routing, and parallelization

### Building Agents
7. **[07-agent-loop](./07-agent-loop)** - Multi-step reasoning and tool chaining
8. **[08-agent-class](./08-agent-class)** - Reusable Agent abstraction
9. **[09-memory](./09-memory)** - Conversation history and state management

### Complete Examples
10. **[10-example-faq-agent](./10-example-faq-agent)** - FAQ agent with RAG (vector search)
11. **[11-example-research-assistant](./11-example-research-assistant)** - Multi-tool research agent

### Advanced Patterns
12. **[12-planning-react](./12-planning-react)** - ReAct pattern (Reasoning + Acting)
13. **[13-fastapi-deployment](./13-fastapi-deployment)** - Deploy your agent with FastAPI

## 🚀 Getting Started

### Prerequisites
- Python 3.10+
- [UV package manager](https://docs.astral.sh/uv/) (recommended) or pip
- OpenAI API key
- Basic Python knowledge (functions, classes, type hints)

### Installation

This repository uses UV for fast, reliable dependency management. Each lesson is self-contained with its own isolated environment.

#### Installing UV

```bash
# macOS/Linux
curl -LsSf https://astral.sh/uv/install.sh | sh

# Windows
powershell -c "irm https://astral.sh/uv/install.ps1 | iex"

# Or with pip
pip install uv
```

#### Set up your API key

```bash
# Create .env file in the root directory
echo "OPENAI_API_KEY=your-key-here" > .env
```

### Usage

Each lesson is self-contained with:
- **README.md** - Tutorial and explanations
- **example.py** - Working code you can run
- **pyproject.toml** - Dependencies and project metadata
- **requirements.txt** - Legacy pip dependencies (for reference)

Start with lesson 01 and work your way through sequentially.

```bash
# Navigate to any lesson
cd 01-api-basics

# Install dependencies and run (UV creates isolated environment automatically)
uv run example.py

# Or install dependencies explicitly first
uv sync
uv run python example.py
```

Each lesson has its own isolated environment, so you can work through them independently without dependency conflicts.

## 💡 Learning Path

### If you're brand new to AI:
Start at lesson 01 and work through each lesson in order.

### If you know the basics:
- Skip to lesson 04 for tool calling
- Jump to lesson 06 for workflow patterns
- Jump to lesson 08 for agent architecture
- Go to lesson 10 for complete examples

### If you want to see working agents:
Check out lessons 10 and 11 for complete, production-ready examples.

## 🎓 Key Concepts

### Why No Frameworks?
Frameworks like LangChain are great, but they hide the fundamentals. This course teaches you what's actually happening under the hood. Once you understand these patterns, you can:
- Use frameworks intelligently
- Debug when things break
- Build custom solutions when frameworks don't fit

### Production-Ready from Day 1
Every example includes:
- Error handling
- Type validation with Pydantic
- Proper logging
- Token counting
- Cost estimation

### When to Use What

| Pattern | Use Case | Example |
|---------|----------|---------|
| **Simple Prompting** | One-off tasks, no external data | Code review, summarization |
| **Structured Output** | Extract data, need type safety | Parse emails, extract entities |
| **Tool Calling** | Need to take actions, call APIs | Weather bot, calculator |
| **Prompt Chaining** | Multi-step transformations | Content pipelines, data enrichment |
| **Routing** | Different queries need different handling | Customer support, specialized prompts |
| **Parallelization** | Independent concurrent tasks | Bulk processing, multi-perspective analysis |
| **Agent Loop** | Multi-step reasoning, tool chaining | Research assistant, booking agent |
| **RAG** | Need to search knowledge base | FAQ bot, docs assistant |

## 📖 Course Philosophy

1. **Fundamentals First**: Understand the OpenAI API before abstractions
2. **Production Patterns**: Real error handling, not just happy paths
3. **Progressive Complexity**: Each lesson builds on the previous
4. **Practical Examples**: Real use cases, not toy demos
5. **Know When to Use What**: Decision frameworks, not just code

## 🛠️ Tech Stack

- **UV**: Fast, modern Python package manager
- **OpenAI API**: GPT-4 and GPT-3.5 models
- **Pydantic**: Type validation and schema generation
- **ChromaDB**: Vector database for RAG
- **Python-dotenv**: Environment variable management

No LangChain, no LlamaIndex, no agent frameworks—just the fundamentals.

## 📝 What You'll Build

By the end of this course, you'll have built:
- ✅ 9 foundational examples (lessons 1-9)
- ✅ Workflow orchestration patterns (chaining, routing, parallelization)
- ✅ 2 complete production agents (lessons 10-11)
- ✅ ReAct agent with planning capabilities (lesson 12)
- ✅ Production FastAPI web service (lesson 13)
- ✅ Understanding of when to use each pattern
- ✅ Skills to build any AI agent from scratch

## 🤝 Contributing

Found an issue or want to improve an example? PRs welcome!

## 📄 License

MIT License - feel free to use this code for learning and in your own projects.

## 🔗 Resources

- [OpenAI API Documentation](https://platform.openai.com/docs)
- [Pydantic Documentation](https://docs.pydantic.dev)
- [ChromaDB Documentation](https://docs.trychroma.com)

---

**Ready to start?** Head to [01-api-basics](./01-api-basics) for your first lesson.
