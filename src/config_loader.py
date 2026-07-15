"""Carga config/config.yaml (con sustitución de variables de entorno)."""
import os
from pathlib import Path
import yaml


def _substitute_env_vars(obj):
    if isinstance(obj, str):
        if obj.startswith("${") and obj.endswith("}"):
            var_name = obj[2:-1]
            return os.environ.get(var_name, "")
        return obj
    if isinstance(obj, dict):
        return {k: _substitute_env_vars(v) for k, v in obj.items()}
    if isinstance(obj, list):
        return [_substitute_env_vars(v) for v in obj]
    return obj


def load_config(config_path: str = "config/config.yaml") -> dict:
    # Resolve path relative to caller or project root
    p = Path(config_path)
    if not p.is_absolute():
        # Try CWD first, then relative to this file's project root
        candidates = [
            Path.cwd() / p,
            Path(__file__).parent.parent / p,
        ]
        for candidate in candidates:
            if candidate.exists():
                p = candidate
                break

    if not p.exists():
        raise FileNotFoundError(f"Archivo de configuración no encontrado: {config_path}")

    with open(p, encoding="utf-8") as f:
        config = yaml.safe_load(f) or {}

    return _substitute_env_vars(config)
