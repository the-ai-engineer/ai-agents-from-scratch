# Fundamentals: Understanding Large Language Models

## What You'll Learn

This lesson covers the essential concepts you need before building AI systems:

- What LLMs are and how they work at a high level
- Key capabilities and limitations
- Common types of AI systems you can build
- How to choose the right model for your use case

No code yet - just the mental models you need to make good engineering decisions.

## What Is a Large Language Model?

An LLM is a system that understands and generates human language. Think of it as a function: you give it text input, it gives you text output.

The input might be a question, a document to summarize, or a request to generate code. The output is text that responds to that input. Simple interface, powerful capabilities.

What makes LLMs remarkable is their flexibility. The same model can summarize documents, classify text, answer questions, write code, and follow complex instructions. You're not training different models for different tasks. You're using one model and changing what you ask it to do.

## How They Work (The 60-Second Version)

During training, an LLM reads massive amounts of text and learns patterns in language. It's not memorizing facts like a database. Instead, it learns statistical relationships between words and concepts.

LLMs generate text by predicting one token at a time. A token is roughly a word or part of a word. Each prediction is based on everything that came before. This process repeats until the full response is complete.

This matters because LLMs are probabilistic, not deterministic. Ask the same question twice and you might get slightly different answers. The model is making predictions based on statistical likelihood, not retrieving absolute truths.

## What LLMs Are Good At (And What They're Not)

### Strengths

LLMs excel at language tasks:
- Writing and editing text
- Summarizing documents
- Translating between languages
- Classifying and categorizing content
- Following complex instructions
- Understanding context

They're remarkably good at generating coherent, natural-sounding text that matches the style and tone you request.

### Limitations

LLMs have real constraints:

- **They hallucinate**: Make up plausible-sounding but completely wrong information
- **No real-time data**: Can't access current information unless you provide it
- **Poor at precise math**: Better at language than calculation
- **Context window limits**: Can only process a limited amount of text at once
- **Stateless by default**: Don't remember previous conversations unless you send the history

Good engineering means knowing when to use LLMs and when to reach for other tools.

## Beyond Text: Multi-Modal Models

Modern models increasingly combine text with other capabilities:

- **GPT-4o**: Analyzes images alongside text prompts
- **Gemini**: Processes documents, screenshots, and diagrams
- **Claude**: Understands images and structured data

These multi-modal models let you build richer applications. For example, you might upload a receipt image and ask "Extract all line items with prices." The model processes both the visual information and your text instruction.

## How to Think About LLMs

### LLMs Are Stateless Functions

Each API call is independent. The model doesn't remember your previous conversation unless you explicitly send that history with your next request.

This is by design, not a limitation. It gives you complete control over context and lets you optimize what you send.

### LLMs Sound Confident Regardless of Accuracy

LLMs generate responses that sound authoritative whether they're correct or not. They don't "know" when they're wrong because they're just predicting likely next tokens.

Your job as an engineer is to validate outputs, provide good context in prompts, and build systems that handle mistakes gracefully.

### You Control Behavior Through Parameters

LLMs expose parameters that affect their behavior:

- **Temperature**: Controls randomness (0 = deterministic, higher = more creative)
- **Max tokens**: Limits response length
- **Top-p**: Alternative way to control randomness
- **Stop sequences**: Define when generation should end

Understanding these controls helps you tune the model for different use cases.

## Common Types of AI Systems

Once you understand LLMs, you can apply them to build specific system types.

### 1. Conversational AI (Chatbots & Assistants)

Systems that answer questions and have multi-turn conversations. Examples:
- Customer support bots
- Internal knowledge assistants
- FAQ systems

Pattern: Maintain conversation history and generate contextual responses.

### 2. Data Intelligence & Extraction

Systems that read documents and extract structured information. Examples:
- Invoice processing
- Resume parsing
- Contract analysis

Pattern: Send document text, get structured data back (JSON, CSV, etc).

### 3. Content Generation

Systems that create new content on demand. Examples:
- Marketing copy generation
- Email drafting
- Blog post writing

Pattern: Provide context and requirements, generate content that matches your brand voice.

### 4. Automated Workflows

Systems that coordinate multiple steps and decisions. Examples:
- Customer onboarding pipelines
- Content review and approval
- Data enrichment and routing

Pattern: Chain multiple LLM calls together, each handling a specific step.

### 5. Autonomous Agents

Systems that pursue goals and figure out the steps themselves. Examples:
- Research assistants
- Code analysis and debugging
- Planning and task decomposition

Pattern: Give the LLM tools it can call, let it decide which tools to use and when.

You'll learn to build all of these throughout this course.

## Choosing the Right Model

There's no single "best" model. The right choice depends on your constraints and requirements.

### Primary Constraints

Every project has limitations. Identify yours:

- **Cost**: Tight budget or high-volume usage?
- **Speed**: Need sub-second latency?
- **Privacy**: Data too sensitive for public APIs?
- **Accuracy**: Are mistakes expensive or dangerous?

This immediately narrows your search.

### Model Types

Think of models along a spectrum:

**Standard models**: Fast, low-cost, great for everyday tasks (summarization, Q&A, classification)
- GPT-4o-mini
- Claude 3.5 Haiku
- Gemini 1.5 Flash

**Reasoning models**: Slower but more capable at multi-step logic and complex problem-solving
- OpenAI o1
- Claude 3.7 Sonnet

**Specialized models**: Optimized for specific tasks
- Embedding models for semantic search
- Code-specific models
- Fine-tuned models for your domain

Rule of thumb: Use the simplest model that works. Upgrade only if you hit limits.

### Deployment Options

Where and how you run models matters:

**Direct APIs** (OpenAI, Anthropic, Google)
- Best for prototyping
- Newest models first
- Developer-friendly docs
- Minimal setup

**Enterprise platforms** (Azure OpenAI, AWS Bedrock)
- Same models with enterprise features
- Region-specific hosting
- Stronger SLAs
- Better for production in regulated industries

**Self-hosted/open-source** (Llama, Mistral, Qwen)
- Complete control
- Best for strict privacy or high-volume cost optimization
- Requires significant infrastructure expertise

Key insight: Azure OpenAI's GPT-4 and OpenAI's GPT-4 are the same model. The difference is compliance, integration, and reliability - not capability.

## Your Job as an AI Engineer

Your job is building systems that use these models effectively.

This means:
- Choosing the right model for your task
- Writing prompts that produce consistent results
- Connecting models to external tools and databases
- Wrapping everything in reliable software that handles errors

The model is one component in your system. All the normal rules of software engineering still apply: good architecture, testing, monitoring, error handling.

You're adding powerful AI capabilities to your toolkit, but the engineering fundamentals remain the same.

## Key Takeaway

LLMs are pattern-recognition systems that predict text based on statistical training. They're powerful but bounded tools with real constraints.

Your job is using them effectively within larger systems by:
- Choosing appropriate models
- Writing clear prompts
- Providing good context
- Validating outputs
- Handling failures gracefully

Everything you build from here forward starts with this understanding.

## Assignment

Watch Andrej Karpathy's "Intro to Large Language Models" video. As you watch, identify three specific LLM limitations that would affect how you build a production system. For each one, write one sentence describing how you'd work around it.

## Resources

- [Intro to Large Language Models](https://www.youtube.com/watch?v=zjkBMFhNj_g) (Andrej Karpathy)
- [OpenAI Model Overview](https://platform.openai.com/docs/models)
- [Anthropic Model Comparison](https://docs.anthropic.com/en/docs/about-claude/models)

## Next Steps

Now that you understand what LLMs are and what they can do, the next lesson covers how to control them effectively through prompt engineering.
