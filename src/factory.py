"""Construye los componentes del sistema desde la configuración."""
from __future__ import annotations

from .config_loader import load_config
from .chat.pipeline import ChatPipeline


def _create_model(config: dict):
    """Crea el modelo de inferencia si hay un checkpoint configurado."""
    model_cfg = config.get("model", {})
    checkpoint = model_cfg.get("checkpoint_path", "").strip()

    if not checkpoint:
        return None

    from .inference.model import FineTunedModel
    return FineTunedModel(
        checkpoint_path=checkpoint,
        device=model_cfg.get("device", "cpu"),
        max_new_tokens=model_cfg.get("max_new_tokens", 512),
        temperature=model_cfg.get("temperature", 0.7),
        top_p=model_cfg.get("top_p", 0.9),
    )


def create_pipeline(config_path: str = "config/config.yaml", checkpoint_override: str = None) -> ChatPipeline:
    """Construye el pipeline de chat desde la configuración.

    checkpoint_override: si se pasa, usa ese checkpoint en vez del de config.yaml
    (util para evaluar varios adaptadores sin editar la config). El device sigue
    saliendo del perfil de hardware.
    """
    config = load_config(config_path)
    if checkpoint_override:
        config.setdefault("model", {})["checkpoint_path"] = checkpoint_override
    model = _create_model(config)
    chat_cfg = config.get("chat", {})
    return ChatPipeline(
        model=model,
        system_prompt=chat_cfg.get("system_prompt", ""),
        max_history=chat_cfg.get("max_history", 5),
    )
