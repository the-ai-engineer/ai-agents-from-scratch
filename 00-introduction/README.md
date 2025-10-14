# AI Agents From First Principles

Welcome to a course that teaches you how AI agents actually work - not just how to use them.

## The Problem with Most AI Courses

Most AI courses teach you to use frameworks like LangChain or AutoGPT. You copy-paste code that works until it doesn't. When things break, you're stuck reading documentation and waiting for fixes because you never learned how it actually works.

This course is different.

## What You'll Learn

You'll build AI agents from scratch using pure Python and the OpenAI API. No frameworks. No magic. Just the fundamental patterns that power every production agent system.

By the end, you'll be able to:

- Build autonomous agents that use tools and reason through problems
- Understand exactly how LangChain, AutoGPT, and other frameworks work internally
- Debug agent systems confidently because you understand the mechanics
- Choose the right patterns for your use case (or build your own)
- Evaluate AI frameworks critically instead of blindly adopting them

## Course Philosophy: First Principles

First principles thinking means understanding the fundamentals before adding abstractions.

Most courses start with frameworks where everything is hidden. We start by showing you exactly what the LLM sees, how to execute tools manually, and how to send results back. Then we progress to building your own abstractions.

You build the patterns yourself. You know how they work.

## What You'll Build

### Core Primitives (Lessons 01-03)

- API communication and streaming
- Conversation memory systems
- Structured output with Pydantic

### Tool Systems (Lesson 04)

- Manual tool definitions to understand the format
- Automatic schema generation
- Tool registries for managing multiple tools

### Workflows (Lesson 05)

Five fundamental patterns for orchestrating LLM calls:

- Chaining: Sequential pipelines where each step feeds into the next
- Routing: Conditional branching based on classification
- Parallelization: Concurrent execution for speed
- Orchestrator-Workers: Dynamic task decomposition
- Evaluator-Optimizer: Iterative refinement loops

### Autonomous Agents (Lessons 06-07)

- The agent loop: "A model using tools in a loop"
- Building a reusable Agent class
- Advanced memory management with token counting, trimming, and persistence

Each lesson builds on previous concepts. Each abstraction is earned through understanding.

## Course Roadmap

```
┌─────────────────────────────────────────────────────────────┐
│                    FOUNDATIONS (01-03)                       │
│  • API Basics      • Memory         • Structured Output     │
└─────────────────────┬───────────────────────────────────────┘
                      │
┌─────────────────────┴───────────────────────────────────────┐
│                    TOOL CALLING (04)                         │
│  Manual → Automatic → @tool decorator → Production patterns │
└─────────────────────┬───────────────────────────────────────┘
                      │
         ┌────────────┴────────────┐
         │                         │
┌────────┴──────────┐    ┌────────┴──────────┐
│  WORKFLOWS (05)   │    │   AGENTS (06-07)  │
│  • Chaining       │    │   • Agent Loop    │
│  • Routing        │    │   • Agent Class   │
│  • Parallel       │    │   • Memory Mgmt   │
│  • Orchestrator   │    │                   │
│  • Evaluator      │    │                   │
└───────────────────┘    └───────────────────┘
         │                         │
         └────────────┬────────────┘
                      │
         ┌────────────┴────────────┐
         │   PRODUCTION (08+)       │
         │   • RAG & Vector DBs    │
         │   • Testing & Debugging │
         │   • Deployment          │
         └─────────────────────────┘
```

## Key Distinction: Workflows vs Agents

One of the most important concepts you'll learn is the difference between workflows and agents.

Workflows are deterministic. You (the developer) decide the execution flow. Think assembly line: outline → draft → polish. Always executes in this order.

Agents are adaptive. The LLM decides what to do based on reasoning. Think problem-solver: the agent decides whether to search the web or calculate based on the user's question.

Understanding this distinction is crucial for choosing the right architecture.

## Prerequisites

### Required

- Solid Python knowledge: Classes, type hints, decorators, async/await
- API basics: Making HTTP requests, handling responses
- OpenAI API key from platform.openai.com

### Helpful but not required

- Experience with REST APIs
- Basic understanding of LLMs
- Familiarity with JSON

### Not required

- Machine learning expertise
- Prior AI framework experience (we build from scratch!)
- Advanced computer science knowledge

## Time Commitment

- Each lesson: 30-60 minutes to read and code
- Total course: 8-12 hours over 1-2 weeks
- Recommended pace: One lesson per day
- Assignments at the end of each lesson are optional but recommended

## Learning Path

### If you're brand new to AI

Start at Lesson 01 and work through sequentially. Each lesson builds on previous concepts.

### If you know the basics

- Already familiar with API calls? Skip to Lesson 02 (Conversation Memory)
- Know memory management? Jump to Lesson 04 (Tool Calling)
- Understand tools? Start at Lesson 05 (Workflows) or Lesson 06 (Agents)

### If you've used frameworks before

You'll appreciate seeing how they work internally. Start at Lesson 04 to see tool calling from first principles, then move to Lesson 06 for the agent loop that powers every framework.

## Why First Principles Matter

### Deep Understanding

When you build from scratch, you understand every component. No black boxes.

### Framework Independence

Once you know the patterns, you can use any framework (or none at all).

### Better Debugging

When agents misbehave, you know exactly where to look.

### Informed Decisions

You can evaluate whether you need a framework, which one, and when to build custom solutions.

### Career Advantage

Most developers copy-paste. You'll understand the fundamentals. That's valuable.

## What This Course Is NOT

This course is NOT:

- A framework tutorial: We don't teach LangChain, AutoGPT, or LlamaIndex
- Production deployment: We focus on fundamentals, not DevOps
- ML theory: No neural networks, no training, no backpropagation
- A comprehensive AI course: This is specifically about agent systems

## Course Structure

Each lesson includes:

- README with conceptual explanation and diagrams
- Clean, executable Python code examples
- Assignment to practice and reinforce learning
- Resources for deeper exploration

All code is type-annotated, uses production-ready patterns, heavily commented, and self-contained.

## Getting Started

Ready to begin? Head to Lesson 01: API Basics to make your first API call and start building.

Remember: The goal isn't to memorize code. The goal is to understand the patterns so deeply that you can recreate them in any language, with any API, for any use case.

Let's build something from first principles.

## FAQ

**Q: Do I need to know LangChain or LlamaIndex first?**

No! In fact, it's better if you don't. You'll understand what these frameworks are doing under the hood.

**Q: Will I be able to use frameworks after this course?**

Absolutely. You'll use them more effectively because you understand the underlying patterns.

**Q: Is this course for beginners?**

It's for developers who want to understand fundamentals. You need Python knowledge, but not AI expertise.

**Q: How is this different from OpenAI's documentation?**

OpenAI docs teach API usage. We teach patterns, architectures, and production practices.

**Q: Why Python?**

Python is the lingua franca of AI. But the patterns apply to any language.

**Q: Can I build production systems with this knowledge?**

Yes! These are the same patterns used in production. You might add frameworks for convenience, but you'll understand what they're doing.

**Q: What if I get stuck?**

Each lesson is self-contained. You can ask questions, review code, and take your time.
