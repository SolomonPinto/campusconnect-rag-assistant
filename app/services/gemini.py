import logging

import numpy as np
from google import genai
from google.genai import types

from app.config import Settings
from app.services.exceptions import ConfigurationError, ProviderError

logger = logging.getLogger(__name__)


def _translate_provider_error(error: Exception) -> ProviderError:
    text = str(error).lower()
    if "api key" in text or "permission" in text or "unauthenticated" in text:
        return ProviderError("Invalid Gemini API key.", 401)
    if "quota" in text or "rate" in text or "429" in text:
        return ProviderError("Gemini rate limit exceeded. Please try again shortly.", 429)
    if "timeout" in text or "deadline" in text:
        return ProviderError("Gemini request timed out. Please try again.", 504)
    logger.exception("Unexpected Gemini API failure")
    return ProviderError("The AI provider is temporarily unavailable.", 502)


class GeminiService:
    def __init__(self, settings: Settings) -> None:
        if not settings.gemini_api_key:
            raise ConfigurationError(
                "GEMINI_API_KEY is not configured. Add it to the .env file."
            )
        self.settings = settings
        self.client = genai.Client(
            api_key=settings.gemini_api_key,
            http_options=types.HttpOptions(timeout=settings.request_timeout_ms),
        )

    def embed_documents(self, texts: list[str]) -> list[np.ndarray]:
        try:
            result = self.client.models.embed_content(
                model=self.settings.embedding_model,
                contents=texts,
                config=types.EmbedContentConfig(
                    task_type="RETRIEVAL_DOCUMENT",
                    output_dimensionality=768,
                ),
            )
            return [np.asarray(item.values, dtype=np.float32) for item in result.embeddings]
        except Exception as error:
            raise _translate_provider_error(error) from error

    def embed_question(self, question: str) -> np.ndarray:
        try:
            result = self.client.models.embed_content(
                model=self.settings.embedding_model,
                contents=question,
                config=types.EmbedContentConfig(
                    task_type="RETRIEVAL_QUERY",
                    output_dimensionality=768,
                ),
            )
            return np.asarray(result.embeddings[0].values, dtype=np.float32)
        except Exception as error:
            raise _translate_provider_error(error) from error

    def answer(self, prompt: str) -> tuple[str, int | None]:
        try:
            response = self.client.models.generate_content(
                model=self.settings.chat_model,
                contents=prompt,
                config=types.GenerateContentConfig(
                    temperature=0.2,
                    max_output_tokens=500,
                ),
            )
            usage = getattr(response, "usage_metadata", None)
            tokens = getattr(usage, "total_token_count", None) if usage else None
            logger.info("Gemini response token usage: %s", tokens if tokens is not None else "unavailable")
            return response.text.strip(), tokens
        except Exception as error:
            raise _translate_provider_error(error) from error
