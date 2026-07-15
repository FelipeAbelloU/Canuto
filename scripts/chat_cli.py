"""Chat por terminal con el asistente normativo.

Uso:
    python scripts/chat_cli.py

Comandos durante el chat:
    salir  — termina la sesión
    reset  — inicia nueva conversación (borra historial)
"""
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))

if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")
if hasattr(sys.stdin, "reconfigure"):
    sys.stdin.reconfigure(encoding="utf-8", errors="replace")

import argparse
from src.factory import create_pipeline


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", default="config/config.yaml")
    args = parser.parse_args()

    print("=" * 60)
    print("  Asistente Normativo Unillanos — Chat Terminal")
    print("=" * 60)
    print("  Cargando pipeline...")

    pipeline = create_pipeline(args.config)

    print(f"  Modelo: {pipeline.model_name}")
    print("  Escribe 'salir' para terminar, 'reset' para nueva conversación.\n")

    while True:
        try:
            question = input("Tu pregunta: ").strip()
        except (EOFError, KeyboardInterrupt):
            print("\nHasta luego.")
            break

        if not question:
            continue
        if question.lower() == "salir":
            print("Hasta luego.")
            break
        if question.lower() == "reset":
            pipeline.reset_history()
            print("Conversación reiniciada.\n")
            continue

        result = pipeline.query(question)
        print(f"\nAsistente: {result['answer']}\n")


if __name__ == "__main__":
    main()
