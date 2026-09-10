"""Arma el modelo y el pipeline de chat a partir de la configuracion."""
from .config_loader import load_config
from .chat.pipeline import ChatPipeline
from .inference.model import FineTunedModel


def create_model(config: dict, checkpoint: str = None):
    """Crea el modelo de inferencia. Devuelve None si no hay checkpoint configurado.

    Con `checkpoint` se evalua una carpeta concreta sin tocar config.yaml
    (lo usa scripts/evaluate.py para comparar varias corridas).
    """
    cfg = config.get("model", {})
    checkpoint = (checkpoint or cfg.get("checkpoint_path", "")).strip()
    if not checkpoint:
        return None

    return FineTunedModel(
        checkpoint_path=checkpoint,
        device=cfg.get("device", "cpu"),
        max_new_tokens=cfg.get("max_new_tokens", 512),
        temperature=cfg.get("temperature", 0.3),
        top_p=cfg.get("top_p", 0.85),
    )


def create_pipeline(config_path: str = "config/config.yaml") -> ChatPipeline:
    """Crea un pipeline de chat listo para responder preguntas."""
    config = load_config(config_path)
    max_history = config.get("chat", {}).get("max_history", 5)
    return ChatPipeline(create_model(config), max_history)
