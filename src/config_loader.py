"""Lee config/config.yaml y lo devuelve como diccionario."""
from pathlib import Path

import yaml

RAIZ = Path(__file__).resolve().parent.parent


def load_config(config_path: str = "config/config.yaml") -> dict:
    """Carga el archivo de configuracion. Las rutas relativas se buscan
    primero desde donde se ejecuta y despues desde la raiz del proyecto."""
    ruta = Path(config_path)
    if not ruta.is_absolute() and not ruta.exists():
        ruta = RAIZ / config_path

    if not ruta.exists():
        raise FileNotFoundError(f"No se encontro la configuracion: {config_path}")

    with open(ruta, encoding="utf-8") as f:
        return yaml.safe_load(f) or {}
