"""Recibe la pregunta del usuario y devuelve la respuesta del modelo."""

SIN_MODELO = (
    "El modelo aun no esta disponible. "
    "Configura la ruta del checkpoint en config/config.yaml (model.checkpoint_path)."
)


class ChatPipeline:
    def __init__(self, model=None):
        self.model = model

    def query(self, question: str) -> dict:
        """Responde una pregunta suelta, como se entreno el modelo."""
        if self.model is None or not self.model.is_available():
            return {"answer": SIN_MODELO, "model_used": "sin modelo"}

        answer = self.model.chat([{"role": "user", "content": question}])
        return {"answer": answer, "model_used": self.model.name}

    @property
    def model_name(self) -> str:
        if self.model and self.model.is_available():
            return self.model.name
        return "sin modelo"
