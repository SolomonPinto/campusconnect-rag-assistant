import json
import logging
from dataclasses import dataclass
from threading import Lock

import numpy as np

from app.config import Settings
from app.services.gemini import GeminiService

logger = logging.getLogger(__name__)


@dataclass
class Chunk:
    title: str
    chunk_id: str
    source_document: str
    text: str
    embedding: np.ndarray | None = None


@dataclass
class SearchResult:
    chunk: Chunk
    score: float


class InMemoryVectorStore:
    def __init__(self, settings: Settings, gemini: GeminiService) -> None:
        self.settings = settings
        self.gemini = gemini
        self.chunks: list[Chunk] = []
        self.is_indexed = False
        self._index_lock = Lock()

    @staticmethod
    def _chunk_text(text: str, max_words: int = 320) -> list[str]:
        words = text.split()
        return [
            " ".join(words[start : start + max_words])
            for start in range(0, len(words), max_words)
        ]

    def ensure_indexed(self) -> None:
        if self.is_indexed:
            return
        with self._index_lock:
            if self.is_indexed:
                return
            with self.settings.docs_path.open(encoding="utf-8") as file:
                documents = json.load(file)

            chunks: list[Chunk] = []
            for document_number, document in enumerate(documents, start=1):
                for chunk_number, text in enumerate(
                    self._chunk_text(document["content"]), start=1
                ):
                    chunks.append(
                        Chunk(
                            title=document["title"],
                            chunk_id=f"doc-{document_number}-chunk-{chunk_number}",
                            source_document=self.settings.docs_path.name,
                            text=text,
                        )
                    )

            document_texts = [
                f"Title: {chunk.title}\nContent: {chunk.text}" for chunk in chunks
            ]
            vectors = self.gemini.embed_documents(document_texts)
            for chunk, vector in zip(chunks, vectors):
                chunk.embedding = vector
            self.chunks = chunks
            self.is_indexed = True
            logger.info("Indexed %s knowledge-base chunks.", len(self.chunks))

    @staticmethod
    def _cosine_similarity(left: np.ndarray, right: np.ndarray) -> float:
        denominator = float(np.linalg.norm(left) * np.linalg.norm(right))
        if denominator == 0:
            return 0.0
        return float(np.dot(left, right) / denominator)

    def search(self, query: str) -> list[SearchResult]:
        self.ensure_indexed()
        query_vector = self.gemini.embed_question(query)
        results = [
            SearchResult(
                chunk=chunk,
                score=self._cosine_similarity(query_vector, chunk.embedding),
            )
            for chunk in self.chunks
            if chunk.embedding is not None
        ]
        results.sort(key=lambda result: result.score, reverse=True)
        top_results = results[: self.settings.top_k]
        logger.info(
            "Similarity scores for query: %s",
            [
                {"chunk_id": result.chunk.chunk_id, "score": round(result.score, 4)}
                for result in top_results
            ],
        )
        return top_results
