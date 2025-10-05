# Lesson 06: Workflow Patterns

Learn how to orchestrate multiple LLM calls to build powerful multi-step workflows.

## What You'll Learn

Before building autonomous agents, you need to understand how to coordinate multiple LLM calls. This lesson covers three fundamental workflow patterns:

1. **Prompt Chaining** - Sequential workflows where outputs feed into next steps
2. **Routing** - Conditional branching to specialized prompts or models
3. **Parallelization** - Concurrent LLM calls for speed and efficiency

## Why Workflow Patterns Matter

Single LLM calls are limited. Real applications need:
- Breaking complex tasks into manageable steps
- Different specialized prompts for different subtasks
- Processing multiple items concurrently
- Routing queries to the right model/prompt

These patterns are building blocks for agents.

## Pattern 1: Prompt Chaining

**Use Case:** Multi-step processing where each step depends on the previous one.

**Example:** Content generation pipeline
```
User input → Generate outline → Write sections → Edit/polish → Final content
```

**Key Characteristics:**
- Sequential execution (step 2 needs step 1's output)
- Each step has a specialized prompt
- State passes through the chain
- Can validate/transform between steps

**When to Use:**
- Multi-step transformations
- Pipelines with clear stages
- When later steps need earlier outputs

## Pattern 2: Routing

**Use Case:** Direct different types of queries to specialized handlers.

**Example:** Customer support system
```
User query → Classify intent → Route to:
  - Technical support prompt
  - Billing prompt
  - General inquiry prompt
```

**Key Characteristics:**
- Conditional branching based on classification
- Different prompts/models for different cases
- Router decides which path to take
- Can route to different models (GPT-4 vs GPT-3.5)

**When to Use:**
- Multiple specialized domains
- Cost optimization (simple queries → cheaper model)
- Different expertise levels needed
- A/B testing different prompts

## Pattern 3: Parallelization

**Use Case:** Process multiple items concurrently or gather multiple perspectives.

**Example 1:** Bulk processing
```
[Item 1, Item 2, Item 3, Item 4, Item 5]
  ↓ (parallel calls)
[Result 1, Result 2, Result 3, Result 4, Result 5]
  ↓ (aggregate)
Final output
```

**Example 2:** Multi-perspective analysis
```
Document → [Summarize, Extract entities, Sentiment analysis] (parallel)
           ↓
         Combined insights
```

**Key Characteristics:**
- Independent LLM calls execute concurrently
- Results aggregated/combined
- Much faster than sequential
- Requires async/parallel execution

**When to Use:**
- Processing lists of items
- Multiple independent analyses
- Time-sensitive applications
- When tasks don't depend on each other

## Combining Patterns

Real applications often combine these patterns:

```
User query
  ↓
Router (classify query type)
  ↓
├─ Technical: Chain (diagnose → solution → verification)
├─ Billing: Parallel (check account + lookup policy) → combine
└─ General: Single call
```

## Cost and Performance

| Pattern | API Calls | Latency | Cost | Complexity |
|---------|-----------|---------|------|------------|
| **Chaining** | N sequential | N × latency | N × cost | Low |
| **Routing** | 1 + routed path | Router + path | Router + path | Medium |
| **Parallelization** | N concurrent | Max(latencies) | N × cost | Medium |

## What's Next?

These workflow patterns give you precise control over LLM orchestration. In the next lesson, we'll build **autonomous agents** that can decide their own workflows dynamically using tool calling and reasoning loops.

**Key Difference:**
- **Workflows (this lesson):** You define the flow (chain, route, parallelize)
- **Agents (next lesson):** LLM decides the flow dynamically

## Code Structure

This lesson includes three examples:

1. `example.py` - All three patterns with practical use cases
   - Content generation pipeline (chaining)
   - Customer support router (routing)
   - Bulk email classifier (parallelization)
   - Multi-perspective document analysis (parallelization)

**Note:** This lesson uses OpenAI's Responses API (`client.responses.create()`) instead of the Chat Completions API. The Responses API provides a simpler interface with `instructions` + `input` parameters and cleaner access to outputs via `response.output_text`.

Run the example:
```bash
cd 06-workflow-patterns
uv run example.py
```

## Key Takeaways

- Workflow patterns orchestrate multiple LLM calls
- **Chain** for sequential dependencies
- **Route** for conditional branching
- **Parallelize** for independent concurrent tasks
- These patterns are foundational for building agents
- Choose pattern based on dependencies between tasks

**Next:** [Lesson 07 - Agent Loop](../07-agent-loop) - Build agents that decide their own workflows
