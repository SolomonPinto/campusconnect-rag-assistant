import logging
from pathlib import Path

from fastapi import FastAPI, Request
from fastapi.exceptions import RequestValidationError
from fastapi.responses import FileResponse, JSONResponse
from fastapi.staticfiles import StaticFiles

from app.config import BASE_DIR, get_settings
from app.models.schemas import ChatRequest, ChatResponse, HealthResponse
from app.services.exceptions import ConfigurationError, ProviderError
from app.services.rag import RagAssistant

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s %(levelname)s %(name)s: %(message)s",
)

app = FastAPI(title="CampusConnect RAG Assistant", version="1.0.0")
frontend_dir = Path(BASE_DIR) / "frontend"
app.mount("/static", StaticFiles(directory=frontend_dir), name="static")
assistant: RagAssistant | None = None


def get_assistant() -> RagAssistant:
    global assistant
    if assistant is None:
        assistant = RagAssistant(get_settings())
    return assistant


@app.exception_handler(RequestValidationError)
async def validation_error_handler(
    request: Request, exc: RequestValidationError
) -> JSONResponse:
    error_types = {error["type"] for error in exc.errors()}
    locations = {tuple(error["loc"]) for error in exc.errors()}
    if "json_invalid" in error_types:
        message = "Invalid JSON payload."
    elif ("body", "message") in locations:
        message = "Message field is required."
    elif ("body", "sessionId") in locations:
        message = "SessionId field is required."
    else:
        message = "sessionId and a non-empty message are required."
    return JSONResponse(
        status_code=422,
        content={"error": message},
    )


@app.exception_handler(ConfigurationError)
async def configuration_error_handler(
    request: Request, exc: ConfigurationError
) -> JSONResponse:
    return JSONResponse(status_code=500, content={"error": str(exc)})


@app.exception_handler(ProviderError)
async def provider_error_handler(request: Request, exc: ProviderError) -> JSONResponse:
    return JSONResponse(status_code=exc.status_code, content={"error": str(exc)})


@app.get("/", include_in_schema=False)
def login_page() -> FileResponse:
    return FileResponse(frontend_dir / "login.html")


@app.get("/chat", include_in_schema=False)
def index() -> FileResponse:
    return FileResponse(frontend_dir / "index.html")


@app.get("/health", response_model=HealthResponse)
def health() -> HealthResponse:
    return HealthResponse(status="healthy")


@app.post("/api/chat", response_model=ChatResponse)
def chat(request: ChatRequest) -> ChatResponse:
    session_id = request.sessionId.strip()
    message = request.message.strip()
    if not session_id or not message:
        return JSONResponse(
            status_code=400,
            content={"error": "sessionId and message cannot be blank."},
        )
    return get_assistant().chat(session_id, message)


@app.delete("/api/chat/{session_id}", status_code=204)
def new_chat(session_id: str) -> None:
    if assistant:
        assistant.history.clear(session_id)
