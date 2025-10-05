# Lesson 13: Deploy Your AI Agent with FastAPI

## What You'll Learn

In this final lesson, you'll learn how to deploy your AI agents as production-ready web services. You'll build a REST API with FastAPI that exposes your agent to web frontends, mobile apps, or any HTTP client.

Deployment transforms your scripts into scalable services. Instead of running Python files locally, you'll create endpoints that handle requests from anywhere, manage conversation sessions across multiple users, stream responses for better UX, and handle errors gracefully.

By the end of this lesson, you'll master building REST APIs with FastAPI, managing conversation sessions and state, implementing request/response validation with Pydantic, adding streaming responses using Server-Sent Events, implementing CORS for web frontends, handling production errors, and deploying with Docker.

This is the bridge between local prototypes and production systems. Every lesson before this taught you how to build agents. This lesson teaches you how to ship them.

## The Problem

Your AI agent works perfectly on your laptop. But it's useless if others can't access it. You need to make it available as a service that frontend developers can integrate, mobile apps can call, other services can consume via API, and multiple users can access simultaneously.

Building this from scratch seems daunting. You need to handle HTTP requests, manage sessions, validate inputs, stream responses, handle CORS, add authentication, log requests, and deploy reliably.

The solution is FastAPI—a modern Python framework that makes building production APIs straightforward. It handles validation automatically, generates API docs, supports async operations, and integrates perfectly with your existing AI agent code.

## Why FastAPI?

FastAPI has become the standard for building AI services because it provides exactly what you need:

**Fast**: High performance with async support out of the box. Critical when making API calls to LLMs.

**Type-safe**: Pydantic integration is built-in. The same models you used for structured outputs work for request/response validation.

**Auto documentation**: Swagger UI generated automatically from your code. No need to write API docs separately.

**Modern Python**: Leverages Python 3.10+ features like type hints and async/await.

**Production-ready**: Companies like Microsoft, Uber, and Netflix use FastAPI in production.

You already know Pydantic from previous lessons. FastAPI is the natural next step.

## System Architecture

Here's how your deployed agent will work:

```
Frontend (React/HTML/Mobile)
        ↓ HTTP Request
   FastAPI Server
        ↓
  Session Manager
        ↓
    Agent Class
        ↓
   OpenAI API
        ↓
   Response (JSON or Stream)
        ↓
    Frontend
```

The FastAPI server handles all the HTTP complexity. Your agent code remains unchanged—you just wrap it with API endpoints.

## Key Endpoints

A production agent API needs these endpoints:

```python
POST   /chat              # Send message, get response
POST   /chat/stream       # Send message, stream response
POST   /sessions          # Create new conversation session
GET    /sessions/:id      # Get conversation history
DELETE /sessions/:id      # Delete conversation session
GET    /health            # Health check for monitoring
```

These cover the core functionality needed for conversational agents.

## Implementation: Basic Setup

First, install dependencies (or UV will do this automatically when you run the server):

Create your FastAPI app:

```python
from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel, Field
import os

app = FastAPI(
    title="AI Agent API",
    description="Production API for AI agents",
    version="1.0.0"
)

# Enable CORS for web frontends
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # In production, specify exact origins
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)
```

CORS is essential if you're building a web frontend. Without it, browsers block your API calls.

## Implementation: Request/Response Models

Define your API contracts with Pydantic:

```python
class ChatRequest(BaseModel):
    message: str = Field(
        min_length=1,
        max_length=5000,
        description="The user's message"
    )
    session_id: str | None = Field(
        default=None,
        description="Optional session ID for conversation continuity"
    )
    temperature: float = Field(
        default=0.7,
        ge=0.0,
        le=2.0,
        description="LLM temperature (0=deterministic, 2=creative)"
    )

class ChatResponse(BaseModel):
    response: str = Field(description="The agent's response")
    session_id: str = Field(description="Session ID for this conversation")
    tokens_used: int = Field(description="Total tokens consumed")
    model: str = Field(description="Model used for generation")

class ErrorResponse(BaseModel):
    error: str = Field(description="Error message")
    detail: str | None = Field(default=None, description="Additional error details")
```

These models provide type safety, automatic validation, and clear documentation.

## Implementation: Session Management

Manage conversation state across requests:

```python
from typing import Dict
from agent import Agent  # Your Agent class from previous lessons
import uuid

# In production, use Redis or a database
sessions: Dict[str, Agent] = {}

def get_or_create_session(session_id: str | None = None) -> tuple[str, Agent]:
    """Get existing session or create a new one."""
    if session_id and session_id in sessions:
        return session_id, sessions[session_id]

    # Create new session
    new_id = str(uuid.uuid4())
    sessions[new_id] = Agent(
        model="gpt-4o-mini",
        system_prompt="You are a helpful AI assistant.",
        tools=[]  # Add your tools here
    )
    return new_id, sessions[new_id]

def delete_session(session_id: str) -> bool:
    """Delete a conversation session."""
    if session_id in sessions:
        del sessions[session_id]
        return True
    return False
```

In-memory storage works for development. Production systems should use Redis or a database.

## Implementation: Chat Endpoint

Create the main chat endpoint:

```python
@app.post("/chat", response_model=ChatResponse)
async def chat(request: ChatRequest):
    """Send a message to the AI agent and get a response."""
    try:
        # Get or create session
        session_id, agent = get_or_create_session(request.session_id)

        # Run the agent
        response = agent.run(
            user_message=request.message,
            temperature=request.temperature
        )

        return ChatResponse(
            response=response,
            session_id=session_id,
            tokens_used=agent.total_tokens_used,
            model=agent.model
        )

    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))
```

FastAPI automatically validates the request against `ChatRequest` and serializes the response to JSON.

## Implementation: Streaming Endpoint

Add streaming for better UX:

```python
from fastapi.responses import StreamingResponse
from sse_starlette.sse import EventSourceResponse

@app.post("/chat/stream")
async def chat_stream(request: ChatRequest):
    """Send a message and stream the response token-by-token."""
    try:
        session_id, agent = get_or_create_session(request.session_id)

        async def generate():
            """Stream tokens as they're generated."""
            # Send session ID first
            yield {
                "event": "session",
                "data": session_id
            }

            # Stream response tokens
            for token in agent.run_stream(
                user_message=request.message,
                temperature=request.temperature
            ):
                yield {
                    "event": "token",
                    "data": token
                }

            # Send completion event
            yield {
                "event": "done",
                "data": str(agent.total_tokens_used)
            }

        return EventSourceResponse(generate())

    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))
```

Streaming provides a ChatGPT-like experience where users see text appear as it's generated.

## Implementation: Session Management Endpoints

Add endpoints for managing sessions:

```python
@app.post("/sessions", status_code=201)
async def create_session():
    """Create a new conversation session."""
    session_id, _ = get_or_create_session()
    return {"session_id": session_id}

@app.get("/sessions/{session_id}")
async def get_session(session_id: str):
    """Get conversation history for a session."""
    if session_id not in sessions:
        raise HTTPException(status_code=404, detail="Session not found")

    agent = sessions[session_id]
    return {
        "session_id": session_id,
        "message_count": len(agent.messages),
        "tokens_used": agent.total_tokens_used
    }

@app.delete("/sessions/{session_id}", status_code=204)
async def delete_session_endpoint(session_id: str):
    """Delete a conversation session."""
    if not delete_session(session_id):
        raise HTTPException(status_code=404, detail="Session not found")
```

These endpoints give frontends full control over conversation lifecycle.

## Implementation: Error Handling

Add global error handling:

```python
from fastapi import Request
from fastapi.responses import JSONResponse

@app.exception_handler(Exception)
async def global_exception_handler(request: Request, exc: Exception):
    """Handle all uncaught exceptions."""
    return JSONResponse(
        status_code=500,
        content=ErrorResponse(
            error="Internal server error",
            detail=str(exc)
        ).dict()
    )

@app.exception_handler(HTTPException)
async def http_exception_handler(request: Request, exc: HTTPException):
    """Handle HTTP exceptions with structured responses."""
    return JSONResponse(
        status_code=exc.status_code,
        content=ErrorResponse(
            error=exc.detail,
            detail=None
        ).dict()
    )
```

Structured error responses make debugging easier for frontend developers.

## Implementation: Health Check

Add a health check for monitoring:

```python
@app.get("/health")
async def health_check():
    """Health check endpoint for load balancers and monitoring."""
    return {
        "status": "healthy",
        "sessions_active": len(sessions),
        "version": "1.0.0"
    }
```

Load balancers and monitoring systems use this to verify your service is running.

## Running the Server

Start your FastAPI server:

```bash
cd 13-fastapi-deployment

# Development mode with auto-reload
uv run uvicorn server:app --reload --host 0.0.0.0 --port 8000

# Production mode (with multiple workers)
uv run uvicorn server:app --host 0.0.0.0 --port 8000 --workers 4
```

Then visit `http://localhost:8000/docs` to see the auto-generated API documentation.

## Testing the API

Test with curl:

```bash
# Create a session
curl -X POST http://localhost:8000/sessions

# Send a chat message
curl -X POST http://localhost:8000/chat \
  -H "Content-Type: application/json" \
  -d '{"message": "Hello!", "session_id": "your-session-id"}'

# Stream a response
curl -X POST http://localhost:8000/chat/stream \
  -H "Content-Type: application/json" \
  -d '{"message": "Tell me a story"}'

# Check health
curl http://localhost:8000/health
```

Or use the interactive docs at `/docs` to test all endpoints.

## Docker Deployment

Create a `Dockerfile`:

```dockerfile
FROM python:3.11-slim

WORKDIR /app

# Install dependencies
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Copy application
COPY . .

# Expose port
EXPOSE 8000

# Run the application
CMD ["uvicorn", "server:app", "--host", "0.0.0.0", "--port", "8000"]
```

Create `requirements.txt`:

```
fastapi==0.109.0
uvicorn[standard]==0.27.0
sse-starlette==1.8.2
openai==1.10.0
pydantic==2.6.0
python-dotenv==1.0.0
```

Build and run:

```bash
# Build the image
docker build -t ai-agent-api .

# Run the container
docker run -p 8000:8000 -e OPENAI_API_KEY=your-key ai-agent-api
```

Your API is now containerized and ready for deployment to any cloud platform.

## Running the Example

This lesson includes a complete FastAPI implementation:

```bash
cd 13-fastapi-deployment
uv run uvicorn server:app --reload
```

Visit `http://localhost:8000/docs` to explore the interactive API documentation.

## Key Takeaways

1. **FastAPI makes APIs easy**: Type-safe by default, auto-documentation, and Pydantic integration make development fast.

2. **Use Pydantic everywhere**: The same models work for structured outputs, request validation, and response serialization.

3. **Session management is critical**: Users expect conversations to persist. Use Redis or databases in production.

4. **CORS for web apps**: Enable CORS if you're building a web frontend. Specify exact origins in production.

5. **Error handling matters**: Return structured errors with helpful messages. Frontend developers will thank you.

6. **Streaming improves UX**: Server-Sent Events provide a ChatGPT-like experience. Use it for longer responses.

## Production Checklist

Before deploying to production, ensure you have:

- [ ] **Authentication**: API keys, OAuth, or JWT tokens to secure endpoints
- [ ] **Rate limiting**: Prevent abuse with per-user or per-IP rate limits
- [ ] **Logging**: Structured logs for debugging and monitoring
- [ ] **Monitoring**: Track request latency, error rates, and token usage
- [ ] **Database**: Replace in-memory sessions with Redis or PostgreSQL
- [ ] **Load balancing**: Distribute traffic across multiple instances
- [ ] **HTTPS/SSL**: Encrypt data in transit with proper certificates
- [ ] **Environment configs**: Use environment variables for all secrets
- [ ] **Graceful shutdown**: Clean up resources when stopping the server
- [ ] **Health checks**: Endpoint for load balancers to verify service health

## Common Pitfalls

1. **In-memory sessions in production**: Sessions are lost on restart. Use Redis or a database.

2. **Not handling errors**: Uncaught exceptions crash the server. Always add global error handlers.

3. **Missing CORS configuration**: Web frontends can't call your API without proper CORS headers.

4. **No rate limiting**: Without limits, users can abuse your API and drive up costs.

5. **Hardcoding secrets**: Never commit API keys to code. Always use environment variables.

6. **Not streaming long responses**: Users don't want to wait 30 seconds for a response. Stream it.

## Real-World Impact

FastAPI powers production AI services at companies like Microsoft, Uber, and Netflix. It handles millions of requests per day while remaining fast and reliable.

Deploying your agent as an API creates immediate business value. Frontend developers can integrate it into web apps, mobile developers can call it from iOS and Android, other services can consume it programmatically, and non-technical users can access it through friendly interfaces.

The business impact is clear: agents become accessible to entire organizations, multiple users can interact simultaneously, conversations persist across sessions, and the system scales from prototype to production without rewrites.

## Assignment

Deploy your research assistant or FAQ agent as a FastAPI service. Implement all core endpoints (chat, stream, sessions), add proper error handling, test all endpoints using the interactive docs, and containerize with Docker.

Then build a simple HTML frontend that calls your API. Use JavaScript to make POST requests to `/chat/stream` and display the streamed response.

Pay attention to:
- Does session management work correctly?
- Do errors return helpful messages?
- Is streaming smooth and responsive?
- Can you deploy the Docker container successfully?

## Congratulations!

You've completed the course! You've mastered building AI agents from scratch and deploying them as production services.

### What You've Learned

1. **OpenAI API**: Chat completions, streaming, parameters, and cost management
2. **Prompt Engineering**: System prompts, few-shot learning, and chain-of-thought
3. **Structured Output**: Type-safe responses with Pydantic validation
4. **Tool Calling**: Giving AI the ability to take actions in the world
5. **Production Tools**: Validation, error handling, and reliability patterns
6. **Agent Loop**: Multi-step reasoning and tool orchestration
7. **Agent Class**: Reusable abstractions for building agents
8. **Memory**: Token management and conversation persistence
9. **RAG**: Vector search for knowledge bases with semantic understanding
10. **Complete Examples**: Production-ready FAQ agents and research assistants
11. **ReAct**: Planning and reasoning for complex tasks
12. **Deployment**: Production-ready FastAPI servers with Docker

You're now ready to build and deploy production AI agents.

## What's Next?

Consider exploring these advanced topics:

**Testing & Evaluation**: Unit tests, integration tests, and LLM-as-judge evaluation frameworks to ensure quality.

**Monitoring & Observability**: LangSmith, OpenTelemetry, and metrics dashboards to track production performance.

**Multi-Agent Systems**: Agents coordinating together to solve complex problems collaboratively.

**Fine-tuning**: Customizing models for your specific use case when prompting isn't enough.

**Production Enhancements**: Database integration (PostgreSQL, MongoDB), Redis for session management, message queues for long-running tasks, WebSocket support for real-time updates, and Kubernetes deployment for scale.

## Resources

- [FastAPI Documentation](https://fastapi.tiangolo.com/)
- [Uvicorn Documentation](https://www.uvicorn.org/)
- [Docker Best Practices](https://docs.docker.com/develop/dev-best-practices/)
- [Production FastAPI Guide](https://fastapi.tiangolo.com/deployment/)
- [Server-Sent Events](https://developer.mozilla.org/en-US/docs/Web/API/Server-sent_events)

Thank you for completing the course. Now go build something amazing!
