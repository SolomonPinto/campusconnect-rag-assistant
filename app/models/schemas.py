from pydantic import BaseModel, Field


class ChatRequest(BaseModel):
    sessionId: str = Field(..., min_length=1, max_length=128)
    message: str = Field(..., min_length=1, max_length=2000)


class RetrievedSource(BaseModel):
    title: str
    chunkId: str
    score: float


class ChatResponse(BaseModel):
    reply: str
    tokensUsed: int | None
    retrievedChunks: int
    sources: list[RetrievedSource] = Field(default_factory=list)


class HealthResponse(BaseModel):
    status: str
