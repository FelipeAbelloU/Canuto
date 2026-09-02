"""Guarda los ultimos turnos de una conversacion."""


class ConversationHistory:
    def __init__(self, max_turns: int = 5):
        self.max_turns = max_turns
        self.turns = []

    def add(self, user: str, assistant: str) -> None:
        """Agrega un turno y descarta los mas viejos si se pasa del limite."""
        self.turns.append({"user": user, "assistant": assistant})
        self.turns = self.turns[-self.max_turns:]

    def clear(self) -> None:
        self.turns = []
