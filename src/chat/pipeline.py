"""Pipeline conversacional para el asistente normativo.
Genera respuestas directamente desde el modelo fine-tuneado (sin RAG).
"""
from __future__ import annotations

from .history import ConversationHistory

_DEFAULT_SYSTEM = (
    "Eres un asistente especializado en normativa universitaria de la "
    "Universidad de los Llanos (Unillanos), Colombia. "
    "Responde preguntas sobre reglamentos, resoluciones, acuerdos y documentos normativos. "
    "Responde siempre en español, de manera clara y útil para estudiantes, docentes y administrativos. "
    "Si no tienes certeza sobre una información, indícalo claramente."
)

_NO_MODEL_MSG = (
    "El modelo aún no está disponible.\n\n"
    "El fine-tuning del modelo está pendiente. "
    "Una vez entrenado, configura la ruta del checkpoint en "
    "`config/config.yaml` bajo `model.checkpoint_path`."
)


class ChatPipeline:
    def __init__(
        self,
        model=None,
        system_prompt: str = "",
        max_history: int = 5,
    ):
        self.model = model
        self.system_prompt = system_prompt or _DEFAULT_SYSTEM
        self.history = ConversationHistory(max_turns=max_history)

    def query(self, question: str) -> dict:
        """Procesa una pregunta y retorna la respuesta del modelo."""
        if self.model is None or not self.model.is_available():
            return {
                "answer": _NO_MODEL_MSG,
                "model_used": "sin modelo",
            }

        messages = [{"role": "system", "content": self.system_prompt}]
        for turn in self.history.turns[-3:]:
            messages.append({"role": "user", "content": turn["user"]})
            messages.append({"role": "assistant", "content": turn["assistant"]})
        messages.append({"role": "user", "content": question})

        answer = self.model.chat(messages)
        self.history.add(question, answer)

        return {
            "answer": answer,
            "model_used": self.model.name,
        }

    def reset_history(self) -> None:
        self.history.clear()

    @property
    def model_name(self) -> str:
        if self.model and self.model.is_available():
            return self.model.name
        return "sin modelo (entrenamiento pendiente)"
