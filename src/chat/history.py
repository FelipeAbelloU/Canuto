from dataclasses import dataclass, field


@dataclass
class ConversationHistory:
    max_turns: int = 5
    turns: list[dict] = field(default_factory=list)

    def add(self, user: str, assistant: str) -> None:
        self.turns.append({"user": user, "assistant": assistant})
        if len(self.turns) > self.max_turns:
            self.turns = self.turns[-self.max_turns:]

    def to_messages(self, system_prompt: str = "") -> list[dict]:
        """Convierte el historial al formato de mensajes OpenAI/Ollama."""
        msgs: list[dict] = []
        if system_prompt:
            msgs.append({"role": "system", "content": system_prompt})
        for turn in self.turns:
            msgs.append({"role": "user", "content": turn["user"]})
            msgs.append({"role": "assistant", "content": turn["assistant"]})
        return msgs

    def format_for_prompt(self) -> str:
        if not self.turns:
            return ""
        lines = []
        for t in self.turns:
            lines.append(f"Usuario: {t['user']}")
            lines.append(f"Asistente: {t['assistant']}")
        return "\n".join(lines)

    def clear(self) -> None:
        self.turns = []
