from collections import defaultdict, deque


class ConversationStore:
    def __init__(self, max_pairs: int) -> None:
        self._history: dict[str, deque[tuple[str, str]]] = defaultdict(
            lambda: deque(maxlen=max_pairs)
        )

    def get_formatted(self, session_id: str) -> str:
        pairs = self._history.get(session_id, [])
        if not pairs:
            return "No previous conversation."
        return "\n".join(
            f"User: {question}\nAssistant: {answer}" for question, answer in pairs
        )

    def add(self, session_id: str, question: str, answer: str) -> None:
        self._history[session_id].append((question, answer))

    def clear(self, session_id: str) -> None:
        self._history.pop(session_id, None)

