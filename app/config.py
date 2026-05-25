import os
from dataclasses import dataclass
from functools import lru_cache
from pathlib import Path

from dotenv import load_dotenv


BASE_DIR = Path(__file__).resolve().parent.parent
load_dotenv(BASE_DIR / ".env")


@dataclass(frozen=True)
class Settings:
    gemini_api_key: str
    chat_model: str
    embedding_model: str
    top_k: int
    similarity_threshold: float
    history_pairs: int
    request_timeout_ms: int
    docs_path: Path


@lru_cache
def get_settings() -> Settings:
    return Settings(
        gemini_api_key=os.getenv("GEMINI_API_KEY", ""),
        chat_model=os.getenv("GEMINI_CHAT_MODEL", "gemini-2.5-flash"),
        embedding_model=os.getenv("GEMINI_EMBEDDING_MODEL", "gemini-embedding-001"),
        top_k=int(os.getenv("TOP_K", "3")),
        similarity_threshold=float(os.getenv("SIMILARITY_THRESHOLD", "0.62")),
        history_pairs=int(os.getenv("HISTORY_PAIRS", "4")),
        request_timeout_ms=int(os.getenv("REQUEST_TIMEOUT_MS", "30000")),
        docs_path=BASE_DIR / "docs.json",
    )
