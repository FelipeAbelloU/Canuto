"""Recibe la pregunta del usuario y devuelve la respuesta del modelo."""
from .history import ConversationHistory

SIN_MODELO = (
    "El modelo aun no esta disponible. "
    "Configura la ruta del checkpoint en config/config.yaml (model.checkpoint_path)."
)


class ChatPipeline:
    def __init__(self, model=None, max_history: int = 5):
        self.model = model
        self.max_history = max_history
        self.history = ConversationHistory(max_turns=max_history)

    def query(self, question: str) -> dict:
        """Responde una pregunta teniendo en cuenta los turnos anteriores."""
        if self.model is None or not self.model.is_available():
            return {"answer": SIN_MODELO, "model_used": "sin modelo"}

        # Se envian los turnos anteriores y al final la pregunta nueva.
        messages = []
        for turn in self.history.turns:
            messages.append({"role": "user", "content": turn["user"]})
            messages.append({"role": "assistant", "content": turn["assistant"]})
        messages.append({"role": "user", "content": question})

        answer = self.model.chat(messages)
        self.history.add(question, answer)
        return {"answer": answer, "model_used": self.model.name}

    def reset_history(self) -> None:
        self.history.clear()

    @property
    def model_name(self) -> str:
        if self.model and self.model.is_available():
            return self.model.name
        return "sin modelo"
