from pathlib import Path

import numpy as np

from app.config import Settings
from app.services.rag import FALLBACK_REPLY, WELCOME_REPLY, RagAssistant
from app.vectorstore.memory_store import Chunk, InMemoryVectorStore, SearchResult


def make_settings() -> Settings:
    return Settings(
        gemini_api_key="test-key",
        chat_model="test-chat",
        embedding_model="test-embedding",
        top_k=3,
        similarity_threshold=0.65,
        history_pairs=4,
        request_timeout_ms=30000,
        docs_path=Path("docs.json"),
    )


class FakeGemini:
    def __init__(self) -> None:
        self.answer_calls = 0
        self.last_prompt = ""

    def answer(self, prompt: str) -> tuple[str, int]:
        self.answer_calls += 1
        self.last_prompt = prompt
        return "Use Profile > Security to change your password.", 42


class FakeVectorStore:
    def __init__(self, results: list[SearchResult]) -> None:
        self.results = results
        self.search_calls = 0

    def search(self, query: str) -> list[SearchResult]:
        self.search_calls += 1
        return self.results


def create_assistant(results: list[SearchResult]) -> tuple[RagAssistant, FakeGemini]:
    assistant = object.__new__(RagAssistant)
    assistant.settings = make_settings()
    assistant.gemini = FakeGemini()
    assistant.vector_store = FakeVectorStore(results)
    from app.services.history import ConversationStore

    assistant.history = ConversationStore(assistant.settings.history_pairs)
    return assistant, assistant.gemini


def test_fallback_does_not_call_llm_when_similarity_is_below_threshold() -> None:
    result = SearchResult(
        Chunk("Unrelated", "chunk-1", "docs.json", "No answer here."), score=0.12
    )
    assistant, gemini = create_assistant([result])

    response = assistant.chat("session-1", "What is the refund policy?")

    assert response.reply == FALLBACK_REPLY
    assert response.retrievedChunks == 0
    assert gemini.answer_calls == 0


def test_social_greeting_receives_welcome_without_document_retrieval() -> None:
    assistant, gemini = create_assistant([])

    response = assistant.chat(
        "session-new-student", "hai i am solomon i am new to this collage"
    )

    assert response.reply == WELCOME_REPLY
    assert response.retrievedChunks == 0
    assert response.sources == []
    assert assistant.vector_store.search_calls == 0
    assert gemini.answer_calls == 0


def test_accepted_chunk_is_injected_into_prompt_before_llm_answer() -> None:
    result = SearchResult(
        Chunk(
            "Password Reset",
            "chunk-1",
            "docs.json",
            "Passwords can be changed from Profile > Security.",
        ),
        score=0.82,
    )
    assistant, gemini = create_assistant([result])

    response = assistant.chat("session-2", "How do I change my password?")

    assert response.retrievedChunks == 1
    assert gemini.answer_calls == 1
    assert "Passwords can be changed from Profile > Security." in gemini.last_prompt
    assert "How do I change my password?" in gemini.last_prompt


def test_cosine_similarity_uses_vector_direction() -> None:
    score = InMemoryVectorStore._cosine_similarity(
        np.asarray([1.0, 0.0]), np.asarray([1.0, 0.0])
    )

    assert score == 1.0
