"""Inicia el servidor web de CANUTO.

Uso:
    python ui/django_chatbot/run.py
    python ui/django_chatbot/run.py --port 8002
"""
import os
import sys
import argparse
from pathlib import Path

# Agrega la raiz del proyecto al path para que Django encuentre src.*
ROOT = Path(__file__).resolve().parent.parent.parent
sys.path.insert(0, str(ROOT))

os.environ.setdefault("DJANGO_SETTINGS_MODULE", "chatbot.settings")

# Cambia al directorio del proyecto Django
os.chdir(Path(__file__).parent)


def main():
    parser = argparse.ArgumentParser()
    # 8001 y no 8000: en la workstation el 8000 lo ocupa otro servicio.
    parser.add_argument("--port", type=int, default=8001)
    parser.add_argument("--host", default="127.0.0.1")
    args = parser.parse_args()

    print("CANUTO — Asistente Normativo Unillanos")
    print(f"  Inicio : http://{args.host}:{args.port}/")
    print(f"  Chat   : http://{args.host}:{args.port}/chat/")
    print("  Ctrl+C para detener")

    from django.core.management import execute_from_command_line
    execute_from_command_line([
        "manage.py",
        "runserver",
        f"{args.host}:{args.port}",
        "--noreload",
    ])


if __name__ == "__main__":
    main()
