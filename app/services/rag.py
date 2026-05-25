import re

from app.config import Settings
from app.models.schemas import ChatResponse, RetrievedSource
from app.prompts.rag_prompt import build_rag_prompt
from app.services.gemini import GeminiService
from app.services.history import ConversationStore
from app.vectorstore.memory_store import InMemoryVectorStore

FALLBACK_REPLY = (
    "I could not find enough information in the knowledge base to answer this question."
)
WELCOME_REPLY = (
    "Hello and welcome to CampusConnect! I can help you with password resets, "
    "course registration, tuition payments, exams, hostel leave, library "
    "policies, profile updates, and technical support. What would you like to know?"
)
ACKNOWLEDGEMENT_REPLY = (
    "Sure! Please ask me about password resets, course registration, tuition "
    "payments, exams, hostel leave, library policies, profile updates, or "
    "technical support."
)
THANKS_REPLY = "You're welcome! Is there anything else about campus services I can help with?"
GOODBYE_REPLY = "Goodbye! Come back anytime you need help with campus services."
SUPPORT_TOPIC_TERMS = {
    "password",
    "registration",
    "register",
    "course",
    "waitlist",
    "tuition",
    "payment",
    "receipt",
    "library",
    "book",
    "exam",
    "hall ticket",
    "hostel",
    "leave",
    "technical",
    "support",
    "ticket",
    "profile",
    "phone",
    "email",
    "address",
}


class RagAssistant:
    def __init__(self, settings: Settings) -> None:
        self.settings = settings
        self.gemini = GeminiService(settings)
        self.vector_store = InMemoryVectorStore(settings, self.gemini)
        self.history = ConversationStore(settings.history_pairs)

    @staticmethod
    def _is_social_greeting(message: str) -> bool:
        normalized = RagAssistant._normalized(message)
        includes_topic = any(term in normalized for term in SUPPORT_TOPIC_TERMS)
        includes_greeting = bool(re.search(r"\b(hi+|hello|hey|hai)\b", normalized))
        includes_introduction = bool(
            re.search(
                r"\b(i am|i'm|my name is|new to (this |the )?(college|collage|campus))\b",
                normalized,
            )
        )
        return (includes_greeting or includes_introduction) and not includes_topic

    @staticmethod
    def _normalized(message: str) -> str:
        return " ".join(re.sub(r"[^a-z0-9\s']", " ", message.lower()).split())

    @classmethod
    def _social_reply(cls, message: str) -> str | None:
        normalized = cls._normalized(message)
        if cls._is_social_greeting(message):
            return WELCOME_REPLY
        if normalized in {"yes", "yeah", "yep", "sure", "ok", "okay", "alright"}:
            return ACKNOWLEDGEMENT_REPLY
        if normalized in {"thanks", "thank you", "thankyou", "thanks a lot"}:
            return THANKS_REPLY
        if normalized in {"bye", "goodbye", "see you", "see you later"}:
            return GOODBYE_REPLY
        return None

    def chat(self, session_id: str, message: str) -> ChatResponse:
        social_reply = self._social_reply(message)
        if social_reply:
            self.history.add(session_id, message, social_reply)
            return ChatResponse(
                reply=social_reply,
                tokensUsed=0,
                retrievedChunks=0,
                sources=[],
            )

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
