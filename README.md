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

## 🧠 The Core Insight

> **"Agents are models using tools in a loop."**

This memorable definition (from Anthropic research) captures everything:
- **Models**: LLMs provide reasoning and decision-making
- **Tools**: Functions that interact with the real world
- **Loop**: Continuous observe → decide → act → update cycle

Everything else—memory, planning, multi-agent systems—is refinement of this core pattern.

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
2. **[02-conversation-memory](./02-conversation-memory)** - Maintaining context across turns
3. **[03-prompting](./03-prompting)** - Prompt engineering essentials
4. **[04-structured-output](./04-structured-output)** - Type-safe responses with Pydantic

### Tool Calling
5. **[05-tool-calling-basics](./05-tool-calling-basics)** - Give AI the ability to use tools
6. **[06-tool-calling-pydantic](./06-tool-calling-pydantic)** - Production tool calling with validation

### Workflow Patterns
7. **[07-workflow-patterns](./07-workflow-patterns)** - 5 fundamental patterns (chaining, routing, parallelization, orchestrator-workers, evaluator-optimizer) 📊

### Building Agents
8. **[08-agent-loop](./08-agent-loop)** - Multi-step reasoning and tool chaining
9. **[09-agent-class](./09-agent-class)** - Reusable Agent abstraction
10. **[10-advanced-memory](./10-advanced-memory)** - Token limits, trimming, and persistence

### Complete Examples
11. **[11-example-faq-agent](./11-example-faq-agent)** - FAQ agent with RAG (vector search)
12. **[12-example-research-assistant](./12-example-research-assistant)** - Multi-tool research agent

### Advanced Patterns
13. **[13-planning-react](./13-planning-react)** - ReAct pattern (Reasoning + Acting)
14. **[14-fastapi-deployment](./14-fastapi-deployment)** - Deploy your agent with FastAPI

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
- Skip to lesson 05 for tool calling
- Jump to lesson 07 for workflow patterns
- Jump to lesson 09 for agent architecture
- Go to lesson 11 for complete examples

### If you want to see working agents:
Check out lessons 11 and 12 for complete, production-ready examples.

## 🎓 Key Concepts

### Workflows vs Agents: The Critical Distinction

**Workflows (Lesson 07):**
- **YOU** define the execution flow
- Deterministic paths (outline → draft → polish)
- Use `WorkflowState` dataclass for intermediate results
- Like assembly lines with fixed stations

**Agents (Lessons 08+):**
- **LLM** decides the execution flow dynamically
- Non-deterministic reasoning (agent chooses tools)
- Use `ConversationMemory` for conversational context
- Like consultants who decide their approach

📊 **[See visual comparison diagram →](./07-workflow-patterns/README.md)**

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
| **Simple Prompting** | One-off tasks, no external data | Code review, summarization | 01-03 |
| **Conversation Memory** | Multi-turn dialogue | Chatbots, assistants | 02 |
| **Structured Output** | Extract data, need type safety | Parse emails, extract entities | 04 |
| **Tool Calling** | Need to take actions, call APIs | Weather bot, calculator | 05-06 |
| **Prompt Chaining** | Sequential transformations | Content pipeline (outline → draft → polish) | 07 |
| **Routing** | Different queries need different handling | Customer support triage | 07 |
| **Parallelization** | Independent concurrent tasks | Bulk email classification | 07 |
| **Orchestrator-Workers** | Complex tasks needing decomposition | Research assistant breaking down topics | 07 |
| **Evaluator-Optimizer** | Quality control & iterative refinement | Code generation with review | 07 |
| **Agent Loop** | Multi-step reasoning, tool chaining | Autonomous research, booking agent | 08-09 |
| **Advanced Memory** | Token limits, persistence | Long conversations, production apps | 10 |
| **RAG** | Need to search knowledge base | FAQ bot, documentation assistant | 11 |

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
