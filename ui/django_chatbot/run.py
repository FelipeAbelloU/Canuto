"""
Inicia el servidor CANUTO.

Uso:
    python ui/django_chatbot/run.py
    python ui/django_chatbot/run.py --port 8080
"""
import os
import sys
import argparse
from pathlib import Path

# Agrega Proyecto5 al path
ROOT = Path(__file__).resolve().parent.parent.parent
sys.path.insert(0, str(ROOT))

os.environ.setdefault("DJANGO_SETTINGS_MODULE", "chatbot.settings")

# Cambia al directorio del proyecto Django para que manage.py encuentre los módulos
os.chdir(Path(__file__).parent)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--port", type=int, default=8000)
    parser.add_argument("--host", default="127.0.0.1")
    args = parser.parse_args()

    print("=" * 55)
    print("  CANUTO — Asistente Normativo Unillanos")
    print("=" * 55)
    print(f"  URL principal : http://{args.host}:{args.port}/")
    print(f"  Chat completo : http://{args.host}:{args.port}/chat/")
    print("  Ctrl+C para detener")
    print()

    from django.core.management import call_command
    call_command("runserver", f"{args.host}:{args.port}", "--noreload")


if __name__ == "__main__":
    main()
