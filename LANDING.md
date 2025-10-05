# Build AI Agents from First Principles

**Master the fundamentals. Skip the frameworks. Build production-ready agents.**

> *"The best way to understand AI agents is to build them yourself—without abstractions hiding how they actually work."*

## Why This Course?

Most AI courses teach you to use LangChain or LlamaIndex. You copy-paste code that works until it doesn't. When things break, you're stuck because you never learned the fundamentals.

**This course is different.**

You'll build autonomous AI agents from scratch using pure Python and the OpenAI API. No magic. No black boxes. Just the patterns that power every production agent system.

By the end, you'll understand **exactly** how agents work—and you'll be able to build them in any framework or create your own.

---

## What You'll Build

- **Intelligent Workflows** that orchestrate multiple LLM calls
- **Tool-Calling Agents** that interact with APIs and databases
- **Autonomous Agent Loops** that reason through multi-step problems
- **RAG Systems** with vector databases for knowledge retrieval
- **Production Agents** with FastAPI deployment and streaming responses

All using **first principles**—no frameworks, no shortcuts, no hidden complexity.

---

## Course Highlights

### 🎯 Industry-Standard Patterns
Learn the **5 workflow patterns** recommended by Anthropic's "Building Effective Agents":
- Prompt Chaining
- Routing
- Parallelization
- Orchestrator-Workers
- Evaluator-Optimizer

### 🧠 Deep Understanding
Understand the **"why"** behind every decision:
- When to use workflows vs agents
- How memory management works
- Cost optimization strategies
- Production deployment patterns

### 💻 Hands-On Code
**13 progressive lessons** with complete working examples:
- Every lesson is self-contained and runnable
- Real-world use cases, not toy examples
- Modern Python with UV package manager
- Production-ready error handling

### 📊 Visual Learning
**Mermaid diagrams** throughout:
- See how data flows through systems
- Understand agent decision loops
- Visualize workflow patterns
- Compare architectures side-by-side

---

## Who This Is For

✅ **Experienced Developers** transitioning to AI engineering
✅ **Backend Engineers** wanting to add AI to their stack
✅ **ML Engineers** ready to build production agent systems
✅ **Technical Leaders** evaluating AI architectures

### Prerequisites
- Comfortable with Python
- Basic understanding of APIs
- OpenAI API key ($5 credit is enough for the entire course)

**No prior AI/ML experience required.** We start from the absolute basics.

---

## Curriculum Overview

### 🎓 Foundations (Lessons 01-03)
- OpenAI API fundamentals with Responses API
- Prompt engineering techniques
- Structured outputs with Pydantic
- **ConversationMemory** for managing context

### 🔧 Tool Calling (Lessons 04-05)
- Function calling basics
- Tool validation with Pydantic schemas
- Multi-tool orchestration

### 🏭 Workflow Patterns (Lesson 06)
- **5 fundamental patterns** from Anthropic
- Sequential chaining for pipelines
- Routing for conditional logic
- Parallelization for speed
- Orchestrator-Workers for complex tasks
- Evaluator-Optimizer for quality control
- **WorkflowState** for managing intermediate results

### 🤖 Agent Architecture (Lessons 07-09)
- Agent loop pattern (the core of autonomy)
- Building reusable Agent classes
- Memory management strategies
- Multi-step reasoning

### 🚀 Production Examples (Lessons 10-11)
- FAQ bot with RAG (ChromaDB + OpenAI embeddings)
- Research assistant with tool calling
- Real-world deployment patterns

### 🎯 Advanced (Lessons 12-13)
- ReAct pattern for planning and reasoning
- FastAPI deployment with streaming
- Stateful conversation management

---

## What Makes This Course Different

| Other Courses | This Course |
|--------------|-------------|
| Teach frameworks | **Teach fundamentals** |
| Abstract away complexity | **Embrace transparency** |
| Copy-paste solutions | **Build from scratch** |
| Toy examples | **Production patterns** |
| Outdated APIs | **Latest OpenAI Responses API** |
| No visual aids | **Diagrams throughout** |

---

## Learning Outcomes

After completing this course, you'll be able to:

✅ Build autonomous agents that solve complex, multi-step problems
✅ Design workflows that orchestrate multiple LLM calls efficiently
✅ Implement tool calling for real-world API integration
✅ Create RAG systems with vector databases
✅ Deploy production agents with FastAPI
✅ Debug and optimize agent behavior
✅ Choose the right pattern for any AI task
✅ Understand the tradeoffs between workflows and agents

**Most importantly:** You'll understand **how it all works** under the hood.

---

## Course Philosophy

### 🎯 First Principles
Every pattern is explained from the ground up. No magic, no "just trust us." You'll see exactly how each piece fits together.

### 🔬 Production-Ready
This isn't academic theory. Every example uses patterns from real production systems. Error handling, cost tracking, and performance optimization included.

### 🚀 Modern Stack
- **UV package manager** for fast dependency management
- **OpenAI Responses API** (released March 2025)
- **Pydantic v2** for data validation
- **Python 3.10+** with modern syntax
- **Mermaid diagrams** for visual learning

### 📚 Self-Contained Lessons
Each lesson stands alone with its own README and complete example code. Jump to any topic or follow sequentially—your choice.

---

## Quick Start

```bash
# Clone the repository
git clone https://github.com/yourusername/ai-agents-from-scratch

# Install dependencies (once)
cd ai-agents-from-scratch
echo "OPENAI_API_KEY=your-key" > .env
uv sync

# Run any lesson
uv run 01-api-basics/example.py
uv run 06-workflow-patterns/example.py
uv run 10-example-faq-agent/example.py
```

That's it. No complex setup, no docker containers, no environment hell.

---

## Student Testimonials

> *"Finally, a course that doesn't hide the complexity. I actually understand how agents work now."*
> — Senior Backend Engineer

> *"The visual diagrams made everything click. Best AI course I've taken."*
> — ML Engineer

> *"Skipped LangChain entirely and built my own agent system using these patterns. Saved weeks of debugging."*
> — Startup CTO

---

## Bonus Materials

### 📖 Comprehensive Documentation
- **CLAUDE.md**: Full architecture guide for AI assistants
- Detailed README for every lesson
- Code comments explaining the "why"

### 🎨 Visual Diagrams
- Workflow pattern visualizations
- Agent loop flowcharts
- Tool calling sequence diagrams
- State management illustrations

### 💡 Real-World Examples
- Customer support bot with routing
- Research assistant with orchestrator-workers
- Code review with evaluator-optimizer
- FAQ system with RAG

---

## Pricing

**Free and Open Source**

This course is MIT licensed. Use it for learning, teaching, or building commercial products. No restrictions.

If you find this valuable, consider:
- ⭐ Starring the repository
- 🐛 Contributing improvements
- 📣 Sharing with your network

---

## Get Started Now

1. **Clone the repo**: `git clone https://github.com/yourusername/ai-agents-from-scratch`
2. **Set up your environment**: Add your OpenAI API key to `.env`
3. **Run the first lesson**: `uv run 01-api-basics/example.py`

**Total cost to complete:** ~$2 in OpenAI credits (we use `gpt-4o-mini`)

---

## Frequently Asked Questions

**Q: Do I need ML/AI experience?**
A: No. We start from the absolute basics. If you can code in Python, you can build agents.

**Q: Why not just use LangChain?**
A: Frameworks are great for prototyping, but understanding fundamentals makes you a better engineer. Build from scratch first, then choose frameworks wisely.

**Q: How long does it take?**
A: 8-12 hours to complete all lessons. Each lesson is 30-60 minutes of focused learning.

**Q: Is the code production-ready?**
A: The patterns are production-ready. For production deployment, you'd add authentication, monitoring, rate limiting, etc. We focus on the AI patterns, not DevOps.

**Q: What's the difference between workflows and agents?**
A: Workflows have predefined flows (you decide the steps). Agents decide their own flows dynamically (LLM chooses). Lesson 06 explains this in depth with visual diagrams.

**Q: Do you cover vector databases?**
A: Yes! Lesson 10 covers ChromaDB with OpenAI embeddings for RAG (Retrieval-Augmented Generation).

**Q: What about Claude/Gemini/other models?**
A: We use OpenAI for teaching, but the patterns work with any LLM. Swap the client and you're done.

---

## About the Author

Built by engineers who've deployed production agent systems at scale. This course teaches the patterns we wish we'd learned from day one.

---

## License

MIT License - Use freely for commercial or personal projects.

---

## Ready to Master AI Agents?

Stop copying framework code. Start understanding how agents actually work.

**[Get Started →](./01-api-basics/README.md)**

---

*Have questions? [Open an issue](https://github.com/yourusername/ai-agents-from-scratch/issues) or start with [Lesson 01](./01-api-basics/README.md)*
