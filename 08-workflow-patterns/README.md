# Workflow Patterns

Learn how to orchestrate multiple LLM calls to solve complex problems.

## What You'll Learn

Five fundamental patterns for coordinating LLM calls:

1. **Prompt Chaining** - Sequential steps where each builds on the previous
2. **Routing** - Classify and route to specialized handlers
3. **Parallelization** - Process multiple items concurrently
4. **Orchestrator-Workers** - Dynamically decompose and delegate tasks
5. **Evaluator-Optimizer** - Generate-evaluate-refine loops

These patterns are the building blocks for autonomous agents.

## Workflows vs Agents

**This lesson (Workflows):**
- **You** control the execution flow
- Deterministic: step 1 → step 2 → step 3
- Example: `outline → draft → polish` (always this order)

**Next lesson (Agents):**
- **LLM** decides the execution flow
- Non-deterministic: agent chooses actions dynamically
- Example: Agent decides whether to call `search()` or `calculate()`

Think: Workflows = assembly lines (fixed steps), Agents = consultants (adaptive)

## Pattern 1: Prompt Chaining

**When:** Sequential steps where each depends on the previous output

**Example:** Content generation pipeline

```python
async def prompt_chaining(topic: str):
    # Step 1: Create outline
    outline = await llm(f"Create a 3-point outline for: {topic}")

    # Step 2: Write draft
    draft = await llm(f"Write based on:\n{outline}")

    # Step 3: Polish
    final = await llm(f"Edit for clarity:\n{draft}")

    return final
```

**Use cases:**
- Content generation (outline → draft → edit)
- Data processing pipelines
- Multi-stage transformations

## Pattern 2: Routing

**When:** Different types of inputs need specialized handling

**Example:** Customer support routing

```python
class SupportCategory(BaseModel):
    category: Literal["shipping", "refund", "technical", "general"]
    urgency: Literal["high", "medium", "low"]

async def routing(query: str):
    # Step 1: Classify
    classification = await client.responses.parse(
        model="gpt-4o-mini",
        input=f"Classify: {query}",
        text_format=SupportCategory
    )

    # Step 2: Route to specialist
    specialists = {
        "shipping": "You are a shipping specialist...",
        "refund": "You are a refund specialist...",
        "technical": "You are technical support...",
    }

    response = await llm(query, specialists[classification.output_parsed.category])
    return response
```

**Use cases:**
- Customer support triage
- Content moderation
- Cost optimization (route simple → cheap model, complex → expensive model)
- A/B testing different prompts

## Pattern 3: Parallelization

**When:** Processing multiple independent items

**Example:** Bulk email classification

```python
async def parallelization(emails: List[str]):
    async def classify(email: str) -> str:
        return await llm(f"Classify as spam/urgent/normal: {email}")

    # Process all concurrently
    results = await asyncio.gather(*[classify(email) for email in emails])
    return results
```

**Benefits:**
- 5x faster for 5 items (vs sequential)
- All items processed in parallel
- Results maintain input order

**Use cases:**
- Bulk document processing
- Multi-perspective analysis
- Time-sensitive applications
- Any list of independent tasks

## Pattern 4: Orchestrator-Workers

**When:** Tasks need intelligent decomposition that varies by input

**Example:** Research assistant

```python
class TaskPlan(BaseModel):
    subtasks: List[str]

async def orchestrator_workers(task: str):
    # Step 1: Orchestrator decomposes task
    plan = await client.responses.parse(
        model="gpt-4o-mini",
        input=f"Break into 3 subtasks: {task}",
        text_format=TaskPlan
    )

    # Step 2: Workers execute in parallel
    results = await asyncio.gather(
        *[llm(f"Complete: {subtask}") for subtask in plan.output_parsed.subtasks]
    )

    # Step 3: Orchestrator synthesizes
    synthesis = await llm(f"Combine for '{task}':\n" + "\n".join(results))
    return synthesis
```

**Key difference from parallelization:**
- Parallelization: Fixed set of items to process
- Orchestrator: Dynamically determines subtasks based on input

**Use cases:**
- Complex research requiring intelligent breakdown
- Multi-perspective analysis
- Tasks where subtasks depend on the input

## Pattern 5: Evaluator-Optimizer

**When:** Quality matters and outputs benefit from iterative refinement

**Example:** Code generation with quality control

```python
class Evaluation(BaseModel):
    score: int  # 1-10
    feedback: str

async def evaluator_optimizer(task: str, target_score: int = 8, max_iterations: int = 3):
    draft = await llm(task)

    for iteration in range(max_iterations):
        # Evaluate
        eval_result = await client.responses.parse(
            model="gpt-4o-mini",
            input=f"Score 1-10 and provide feedback:\n{draft}",
            text_format=Evaluation
        )

        score = eval_result.output_parsed.score

        # Check if good enough
        if score >= target_score:
            return draft

        # Refine based on feedback
        draft = await llm(f"Improve based on: {eval_result.output_parsed.feedback}\n\n{draft}")

    return draft
```

**Loop structure:**
1. Generate (or refine)
2. Evaluate with structured output (score + feedback)
3. If good enough → done, else → refine and loop

**Use cases:**
- Code generation
- Technical writing
- Content that needs quality control
- Design iterations

## Combining Patterns

Real applications often combine these patterns:

**Example 1: Intelligent support system**
```
User query
  ↓
Routing (classify)
  ↓
├─ Technical: Chaining (diagnose → solution → verify)
├─ Billing: Parallelization (check account + policy) → combine
└─ General: Single call
```

**Example 2: Research with quality control**
```
Topic
  ↓
Orchestrator-Workers (decompose → research → synthesize)
  ↓
Evaluator-Optimizer (draft → review → polish)
  ↓
Final report
```

## Cost & Performance

**Chaining:**
- API Calls: N sequential
- Latency: N × per-call latency
- When: Steps depend on each other

**Routing:**
- API Calls: 1 (classifier) + 1 (routed handler)
- Latency: Router + handler latency
- When: Different specialized domains

**Parallelization:**
- API Calls: N concurrent
- Latency: Max(individual latencies) - much faster than sequential!
- When: Independent items, need speed

**Orchestrator-Workers:**
- API Calls: 1 (decompose) + N (workers) + 1 (synthesize)
- Latency: Orchestrator + Max(workers) + Synthesis
- When: Dynamic task breakdown needed

**Evaluator-Optimizer:**
- API Calls: 2-3 per iteration (generate/refine + evaluate)
- Latency: 2-3 × per-call latency per iteration
- When: Quality critical outputs

## Running the Examples

```bash
# Run in Jupyter/IPython for best experience
cd 08-workflow-patterns
ipython

# Or run specific patterns
uv run python -c "import asyncio; from example import prompt_chaining; print(asyncio.run(prompt_chaining('AI')))"
```

**Jupyter examples:**

```python
# Pattern 1: Chaining
result = await prompt_chaining("machine learning basics")
print(result["final"])

# Pattern 2: Routing
result = await routing("I need a refund!")
print(result["response"])

# Pattern 3: Parallelization
emails = ["Meeting at 3pm", "50% off!", "Project update"]
result = await parallelization(emails)
for email, classification in zip(result["emails"], result["classifications"]):
    print(f"{email} → {classification}")

# Pattern 4: Orchestrator
result = await orchestrator_workers("Plan a product launch")
print(result["synthesis"])

# Pattern 5: Optimizer
result = await evaluator_optimizer("Write binary search in Python")
print(f"Score: {result['score']}/10 after {result['iterations']} iterations")
```

## Key Takeaways

1. **Chaining** - Sequential dependencies
2. **Routing** - Conditional branching based on classification
3. **Parallelization** - Concurrent processing for speed
4. **Orchestrator-Workers** - Dynamic decomposition
5. **Evaluator-Optimizer** - Quality control through iteration

6. **Workflows** (this lesson) = You control flow
7. **Agents** (next lesson) = LLM controls flow

8. These patterns combine in real applications
9. Choice depends on task structure and requirements
10. All patterns are async-first for performance

## Next Steps

**Lesson 09: Agent Architecture** - Learn how to build autonomous agents where the LLM decides which tools to call and when, rather than following predetermined workflows.

The key difference: Workflows execute fixed sequences, Agents make dynamic decisions.
