"""Carga config/config.yaml y aplica el perfil de hardware como overlay."""
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


def _deep_merge(base: dict, override: dict) -> dict:
    result = base.copy()
    for k, v in override.items():
        if k in result and isinstance(result[k], dict) and isinstance(v, dict):
            result[k] = _deep_merge(result[k], v)
        else:
            result[k] = v
    return result


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

    config = _substitute_env_vars(config)

    # Apply hardware profile overlay
    profile = config.get("hardware_profile")
    if profile:
        profile_file = p.parent / f"{profile}.yaml"
        if profile_file.exists():
            with open(profile_file, encoding="utf-8") as f:
                profile_cfg = yaml.safe_load(f) or {}
            profile_cfg = _substitute_env_vars(profile_cfg)
            config = _deep_merge(config, profile_cfg)

    return config
