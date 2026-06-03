from dataclasses import dataclass, field


@dataclass
class ConversationHistory:
    """Almacena el historial de turnos de una conversación."""

    max_turns: int = 5
    turns: list[dict] = field(default_factory=list)

    def add(self, user: str, assistant: str) -> None:
        """Agrega un turno y descarta los más antiguos si se supera el límite."""
        self.turns.append({"user": user, "assistant": assistant})
        if len(self.turns) > self.max_turns:
            self.turns = self.turns[-self.max_turns:]

    def clear(self) -> None:
        self.turns = []
