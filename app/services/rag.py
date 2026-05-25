from app.config import Settings
from app.models.schemas import ChatResponse, RetrievedSource
from app.prompts.rag_prompt import build_rag_prompt
from app.services.gemini import GeminiService
from app.services.history import ConversationStore
from app.vectorstore.memory_store import InMemoryVectorStore

FALLBACK_REPLY = (
    "I could not find enough information in the knowledge base to answer this question."
)


class RagAssistant:
    def __init__(self, settings: Settings) -> None:
        self.settings = settings
        self.gemini = GeminiService(settings)
        self.vector_store = InMemoryVectorStore(settings, self.gemini)
        self.history = ConversationStore(settings.history_pairs)

    def chat(self, session_id: str, message: str) -> ChatResponse:
        results = self.vector_store.search(message)
        accepted = [
            result
            for result in results
            if result.score >= self.settings.similarity_threshold
        ]
        sources = [
            RetrievedSource(
                title=result.chunk.title,
                chunkId=result.chunk.chunk_id,
                score=round(result.score, 4),
            )
            for result in accepted
        ]

        if not accepted:
            self.history.add(session_id, message, FALLBACK_REPLY)
            return ChatResponse(
                reply=FALLBACK_REPLY,
                tokensUsed=0,
                retrievedChunks=0,
                sources=[],
            )

        context = "\n\n".join(
            f"[{result.chunk.title} | {result.chunk.chunk_id}]\n{result.chunk.text}"
            for result in accepted
        )
        prompt = build_rag_prompt(
            context=context,
            history=self.history.get_formatted(session_id),
            question=message,
        )
        reply, tokens = self.gemini.answer(prompt)
        self.history.add(session_id, message, reply)
        return ChatResponse(
            reply=reply,
            tokensUsed=tokens,
            retrievedChunks=len(accepted),
            sources=sources,
        )

