"""Chat por terminal con CANUTO.

Uso:
    python scripts/chat_cli.py

Dentro del chat: "salir" termina.
"""
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))

import argparse
from src.factory import create_pipeline


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", default="config/config.yaml")
    args = parser.parse_args()

    print("CANUTO — Asistente Normativo Unillanos")
    print("Cargando...")

    pipeline = create_pipeline(args.config)

    print(f"  Modelo: {pipeline.model_name}")
    print("  Escribe 'salir' para terminar.\n")

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

        result = pipeline.query(question)
        print(f"\nAsistente: {result['answer']}\n")


if __name__ == "__main__":
    main()
