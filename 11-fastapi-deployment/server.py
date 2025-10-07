"""
Lesson 11: FastAPI Deployment

Production-ready FastAPI server for your AI agent.
"""

import sys
from pathlib import Path

# Add parent directory to path to import agents module
sys.path.insert(0, str(Path(__file__).parent.parent))

import uuid
from typing import AsyncIterator
from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import StreamingResponse
from pydantic import BaseModel, Field
from dotenv import load_dotenv
from openai import OpenAI

# Import from our agents framework
from agents import Agent, tool, ConversationMemory

load_dotenv()


# === Example Tools (optional - can be added to agents) ===

@tool
def get_current_time() -> str:
    """Get the current time

    Returns:
        Current time as a string
    """
    from datetime import datetime
    return datetime.now().strftime("%Y-%m-%d %H:%M:%S")


# === Agent Wrapper for Streaming ===

class StreamingAgent:
    """Wrapper around Agent class to add streaming support for FastAPI"""

    def __init__(self, model: str = "gpt-4o-mini", system_prompt: str = None, tools: list = None):
        self.agent = Agent(
            model=model,
            system_prompt=system_prompt or "You are a helpful assistant.",
            tools=tools or []
        )
        self.client = OpenAI()
        self.model = model
        self.total_tokens = 0

    def chat(self, message: str) -> str:
        """Send a message and get a response"""
        return self.agent.chat(message)

    def chat_stream(self, message: str):
        """Stream response token by token"""
        self.agent.memory.add_user_message(message)

        stream = self.client.responses.create(
            model=self.model,
            input=self.agent.memory.get_items(),
            tools=self.agent.tool_schemas,
            stream=True
        )

        full_response = ""
        for chunk in stream:
            for output in chunk.output:
                if output.type == "message_delta" and output.content:
                    full_response += output.content
                    yield output.content

        # Add complete response to history
        self.agent.memory.add_response_output([{
            "type": "message",
            "content": [{"type": "output_text", "text": full_response}]
        }])

    @property
    def conversation_history(self):
        """Get conversation history for compatibility"""
        return self.agent.memory.get_items()


# === Pydantic Models ===

class ChatRequest(BaseModel):
    """Request model for chat endpoint"""
    message: str = Field(
        min_length=1,
        max_length=5000,
        description="User message to send to the agent"
    )
    conversation_id: str | None = Field(
        default=None,
        description="Optional conversation ID to continue existing conversation"
    )


class ChatResponse(BaseModel):
    """Response model for chat endpoint"""
    response: str = Field(description="Agent's response")
    conversation_id: str = Field(description="Conversation ID for this session")
    tokens_used: int = Field(description="Total tokens used in this conversation")


class ConversationInfo(BaseModel):
    """Information about a conversation"""
    conversation_id: str
    message_count: int
    tokens_used: int


class ErrorResponse(BaseModel):
    """Error response model"""
    error: str
    detail: str | None = None


# === FastAPI App ===

app = FastAPI(
    title="AI Agent API",
    description="Production-ready API for AI agents built from scratch",
    version="1.0.0"
)

# Enable CORS for web frontends
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # In production: specify your frontend domain
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# In-memory session storage (Production: use Redis or database)
sessions: dict[str, StreamingAgent] = {}

# System prompt for our agent
SYSTEM_PROMPT = """You are a helpful AI assistant built from scratch.

You can answer questions, help with tasks, and engage in conversation.
Be concise, accurate, and friendly."""


# === Helper Functions ===

def get_or_create_session(conversation_id: str | None) -> tuple[str, StreamingAgent]:
    """Get existing session or create new one"""
    if conversation_id and conversation_id in sessions:
        return conversation_id, sessions[conversation_id]

    # Create new session with agent framework
    new_id = str(uuid.uuid4())
    sessions[new_id] = StreamingAgent(
        system_prompt=SYSTEM_PROMPT,
        tools=[get_current_time]  # Add tools as needed
    )
    return new_id, sessions[new_id]


# === API Endpoints ===

@app.get("/")
async def root():
    """Root endpoint"""
    return {
        "message": "AI Agent API",
        "version": "1.0.0",
        "docs": "/docs",
        "active_sessions": len(sessions)
    }


@app.get("/health")
async def health_check():
    """Health check endpoint"""
    return {
        "status": "healthy",
        "active_sessions": len(sessions)
    }


@app.post("/chat", response_model=ChatResponse)
async def chat(request: ChatRequest):
    """
    Chat with the AI agent.

    Creates a new conversation if no conversation_id provided,
    or continues existing conversation.
    """
    try:
        # Get or create session
        conversation_id, agent = get_or_create_session(request.conversation_id)

        # Get response from agent
        response = agent.chat(request.message)

        return ChatResponse(
            response=response,
            conversation_id=conversation_id,
            tokens_used=agent.total_tokens
        )

    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@app.post("/chat/stream")
async def chat_stream(request: ChatRequest):
    """
    Chat with streaming response (Server-Sent Events).

    Use this for better UX - users see response as it's generated.
    """
    try:
        conversation_id, agent = get_or_create_session(request.conversation_id)

        async def event_generator() -> AsyncIterator[str]:
            """Generate SSE events"""
            # Send conversation ID first
            yield f"data: {{'conversation_id': '{conversation_id}'}}\n\n"

            # Stream response
            for token in agent.chat_stream(request.message):
                yield f"data: {{'token': '{token}'}}\n\n"

            # Send done signal
            yield f"data: {{'done': true}}\n\n"

        return StreamingResponse(
            event_generator(),
            media_type="text/event-stream"
        )

    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@app.get("/conversations/{conversation_id}", response_model=ConversationInfo)
async def get_conversation(conversation_id: str):
    """Get information about a conversation"""
    if conversation_id not in sessions:
        raise HTTPException(status_code=404, detail="Conversation not found")

    agent = sessions[conversation_id]

    return ConversationInfo(
        conversation_id=conversation_id,
        message_count=len(agent.conversation_history),
        tokens_used=agent.total_tokens
    )


@app.delete("/conversations/{conversation_id}")
async def delete_conversation(conversation_id: str):
    """Delete a conversation session"""
    if conversation_id not in sessions:
        raise HTTPException(status_code=404, detail="Conversation not found")

    del sessions[conversation_id]

    return {"message": "Conversation deleted", "conversation_id": conversation_id}


@app.get("/conversations")
async def list_conversations():
    """List all active conversations"""
    conversations = []
    for conv_id, agent in sessions.items():
        conversations.append({
            "conversation_id": conv_id,
            "message_count": len(agent.conversation_history),
            "tokens_used": agent.total_tokens
        })

    return {
        "total": len(conversations),
        "conversations": conversations
    }


# === Error Handlers ===

@app.exception_handler(HTTPException)
async def http_exception_handler(request, exc):
    """Handle HTTP exceptions"""
    return {
        "error": exc.detail,
        "status_code": exc.status_code
    }


@app.exception_handler(Exception)
async def global_exception_handler(request, exc):
    """Handle all other exceptions"""
    return {
        "error": "Internal server error",
        "detail": str(exc)
    }


# === Run Server ===

if __name__ == "__main__":
    import uvicorn

    print("=" * 60)
    print("AI Agent API Server")
    print("=" * 60)
    print("\n🚀 Starting server on http://localhost:8000")
    print("📚 API docs available at http://localhost:8000/docs")
    print("🔍 Interactive API at http://localhost:8000/redoc")
    print("\nPress Ctrl+C to stop\n")

    uvicorn.run(
        "server:app",
        host="0.0.0.0",
        port=8000,
        reload=True,  # Auto-reload on code changes
        log_level="info"
    )
