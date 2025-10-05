# Build AI Agents from First Principles

> *Learn to build autonomous AI agents from scratch using pure Python and the OpenAI API*

**Stop copying framework code. Start understanding how agents actually work.**

## 🚀 What Makes This Course Different?

Most AI courses teach you to use LangChain or LlamaIndex. You copy-paste code that works until it doesn't. When things break, you're stuck because **you never learned the fundamentals**.

This course is different:

✅ **No frameworks** - Pure Python + OpenAI API (understand what's really happening)
✅ **Industry-standard patterns** - Based on [Anthropic's "Building Effective Agents"](https://www.anthropic.com/engineering/building-effective-agents) research
✅ **Production-ready** - Real error handling, cost tracking, optimization strategies
✅ **Visual learning** - Mermaid diagrams throughout every key concept
✅ **Self-contained lessons** - Jump to any topic or follow sequentially

**[📖 Read the full landing page →](./LANDING.md)**

## 💡 Why Learn from Scratch?

When you understand first principles, you can:
- Build with **any** framework (or create your own)
- Debug mysterious issues at 2am with confidence
- Make architecture decisions based on understanding, not guessing
- Never be blocked by incomplete documentation

Frameworks hide complexity. We embrace transparency.

## 🎯 What You'll Learn

### Foundations
- **OpenAI Responses API** (the modern way to call LLMs)
- **ConversationMemory** helper for managing context
- **Prompt engineering** techniques that actually work
- **Structured outputs** with Pydantic for type-safe responses

### The 5 Workflow Patterns (from Anthropic)
1. **Prompt Chaining** - Sequential pipelines with `WorkflowState`
2. **Routing** - Conditional branching to specialized handlers
3. **Parallelization** - Concurrent execution for speed
4. **Orchestrator-Workers** - Dynamic task decomposition
5. **Evaluator-Optimizer** - Generate-evaluate-refine loops

### Agent Architecture
- **Tool calling** - Give AI the ability to take actions
- **Agent loops** - Multi-step reasoning that decides what to do next
- **Memory management** - Conversation history and context
- **RAG systems** - Vector search with ChromaDB
- **Production deployment** - FastAPI with streaming

### When to Use What
- **Workflows vs Agents** - The critical architectural decision
- **WorkflowState vs ConversationMemory** - Managing different types of state
- **Cost optimization** - Token counting and model selection

## 📚 Course Structure

### Foundations
1. **[01-api-basics](./01-api-basics)** - Your first API call to OpenAI
2. **[02-prompting](./02-prompting)** - Prompt engineering essentials
3. **[03-structured-output](./03-structured-output)** - Type-safe responses with Pydantic

### Tool Calling
4. **[04-tool-calling-basics](./04-tool-calling-basics)** - Give AI the ability to use tools
5. **[05-tool-calling-pydantic](./05-tool-calling-pydantic)** - Production tool calling with validation

### Workflow Patterns
6. **[06-workflow-patterns](./06-workflow-patterns)** - 5 fundamental patterns (chaining, routing, parallelization, orchestrator-workers, evaluator-optimizer) 📊

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

### Workflows vs Agents: The Critical Distinction

**Workflows (Lesson 06):**
- **YOU** define the execution flow
- Deterministic paths (outline → draft → polish)
- Use `WorkflowState` dataclass for intermediate results
- Like assembly lines with fixed stations

**Agents (Lessons 07+):**
- **LLM** decides the execution flow dynamically
- Non-deterministic reasoning (agent chooses tools)
- Use `ConversationMemory` for conversational context
- Like consultants who decide their approach

📊 **[See visual comparison diagram →](./06-workflow-patterns/README.md)**

### Production-Ready from Day 1
Every example includes:
- Error handling (rate limits, network failures)
- Type validation with Pydantic
- Token counting for cost tracking
- Performance optimization strategies
- Clear visual diagrams explaining flows

### When to Use What

| Pattern | Use Case | Example | Lesson |
|---------|----------|---------|--------|
| **Simple Prompting** | One-off tasks, no external data | Code review, summarization | 01-02 |
| **Structured Output** | Extract data, need type safety | Parse emails, extract entities | 03 |
| **Tool Calling** | Need to take actions, call APIs | Weather bot, calculator | 04-05 |
| **Prompt Chaining** | Sequential transformations | Content pipeline (outline → draft → polish) | 06 |
| **Routing** | Different queries need different handling | Customer support triage | 06 |
| **Parallelization** | Independent concurrent tasks | Bulk email classification | 06 |
| **Orchestrator-Workers** | Complex tasks needing decomposition | Research assistant breaking down topics | 06 |
| **Evaluator-Optimizer** | Quality control & iterative refinement | Code generation with review | 06 |
| **Agent Loop** | Multi-step reasoning, tool chaining | Autonomous research, booking agent | 07-08 |
| **RAG** | Need to search knowledge base | FAQ bot, documentation assistant | 10 |

## 📖 Course Philosophy

1. **First Principles**: Understand the "why" behind every pattern
2. **Industry-Standard**: Aligned with [Anthropic's research](https://www.anthropic.com/engineering/building-effective-agents)
3. **Visual Learning**: Mermaid diagrams for every key concept 📊
4. **Production-Ready**: Real error handling, cost tracking, optimization
5. **Progressive Complexity**: Each lesson builds on the previous
6. **Self-Contained**: Jump to any lesson or follow sequentially
7. **Framework-Agnostic**: Once you know fundamentals, use any framework

**Cost to complete:** ~$2 in OpenAI credits (using `gpt-4o-mini`)
**Time investment:** 8-12 hours total
**Value:** Understanding that will serve your entire AI career

## 🛠️ Tech Stack

- **UV**: Fast, modern Python package manager
- **OpenAI API**: GPT-4 and GPT-3.5 models
- **Pydantic**: Type validation and schema generation
- **ChromaDB**: Vector database for RAG
- **Python-dotenv**: Environment variable management

No LangChain, no LlamaIndex, no agent frameworks—just the fundamentals.

## 📝 What You'll Build

By the end of this course, you'll have built:

**Workflow Patterns:**
- ✅ Content generation pipeline (chaining)
- ✅ Customer support router (routing)
- ✅ Bulk email classifier (parallelization)
- ✅ Research assistant (orchestrator-workers)
- ✅ Code generator with review (evaluator-optimizer)

**Agent Systems:**
- ✅ Multi-tool weather agent (tool calling)
- ✅ Autonomous reasoning agent (agent loop)
- ✅ FAQ bot with RAG (vector search + ChromaDB)
- ✅ Research assistant (multi-tool coordination)
- ✅ ReAct agent (planning + reasoning)

**Production Deployment:**
- ✅ FastAPI web service with streaming
- ✅ Stateful conversation management
- ✅ Error handling and cost tracking

**Most Importantly:**
- ✅ Understanding of workflows vs agents
- ✅ Knowledge of when to use each pattern
- ✅ Skills to build any AI agent from scratch
- ✅ Confidence to debug and optimize agents

## 🤝 Contributing

Found an issue or want to improve an example? PRs welcome!

## 📄 License

MIT License - feel free to use this code for learning and in your own projects.

## 🔗 Resources

**Course Materials:**
- [📖 Full Landing Page](./LANDING.md) - Complete course overview
- [📣 Promotional Materials](./PROMO.md) - Social media posts, email templates, etc.
- [🤖 CLAUDE.md](./CLAUDE.md) - Architecture guide for AI assistants

**External Documentation:**
- [OpenAI API Documentation](https://platform.openai.com/docs)
- [Anthropic: Building Effective Agents](https://www.anthropic.com/engineering/building-effective-agents)
- [Pydantic Documentation](https://docs.pydantic.dev)
- [ChromaDB Documentation](https://docs.trychroma.com)

## ⭐ Support This Project

If you find this course valuable:
- ⭐ **Star this repository** to show your support
- 🐛 **Report issues** or contribute improvements
- 📣 **Share with your network** - help others discover it
- 💬 **Give feedback** - what worked? what didn't?

## 🎓 Who This Is For

✅ Backend engineers adding AI to their stack
✅ ML engineers building production systems
✅ Technical leaders evaluating AI architectures
✅ Career switchers transitioning to AI engineering

**Prerequisites:** Python proficiency + basic API knowledge. No ML/AI experience required.

---

**Ready to master AI agents?**

🚀 **[Start with Lesson 01: API Basics →](./01-api-basics)**
📖 **[Read the full landing page →](./LANDING.md)**

*Stop copying framework code. Start understanding how agents actually work.*
