"""
scripts/test_model.py — Pruebas automáticas del modelo entrenado

Ejecuta preguntas basadas en los documentos reales del dataset y guarda
las respuestas en data/test_results.txt para evaluación manual.

Uso:
    python scripts/test_model.py
"""
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))
if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")

from src.factory import create_pipeline

PREGUNTAS = [
    # Acuerdo Académico 003 DE 2023 — Segunda Lengua
    ("segunda_lengua_1",
     "¿Qué es el requisito de suficiencia en segunda lengua en Unillanos?"),
    ("segunda_lengua_2",
     "¿Cuál es el requisito de segunda lengua para graduarse en un programa profesional en Unillanos?"),

    # Fraccionamiento de matrícula
    ("matricula_1",
     "¿Cómo funciona el fraccionamiento de matrícula en Unillanos?"),
    ("matricula_2",
     "¿Cuántos pagos permite el fraccionamiento de matrícula según la resolución vigente?"),

    # Incapacidades
    ("incapacidad_1",
     "¿Qué debo hacer si tengo una incapacidad médica durante el semestre en Unillanos?"),

    # Opciones de grado
    ("grado_1",
     "¿Cuáles son las opciones de grado disponibles en la Facultad de Ciencias Básicas e Ingeniería?"),

    # Movilidad
    ("movilidad_1",
     "¿Qué es la convocatoria de movilidad saliente y cuáles son sus requisitos?"),

    # Pregunta fuera del dominio — para detectar alucinación
    ("fuera_dominio",
     "¿Cuánto cuesta la matrícula en Unillanos para el semestre 2026-2?"),
]


def main():
    output_file = Path("data/test_results.txt")
    output_file.parent.mkdir(parents=True, exist_ok=True)

    print("Cargando pipeline (~2 min)...")
    pipeline = create_pipeline("config/config.yaml")
    print(f"Modelo: {pipeline.model_name}\n")
    print("=" * 65)

    lines = []
    lines.append(f"RESULTADOS DE PRUEBA — Modelo: {pipeline.model_name}\n")
    lines.append("=" * 65 + "\n\n")

    for i, (tag, pregunta) in enumerate(PREGUNTAS, 1):
        print(f"[{i}/{len(PREGUNTAS)}] {pregunta[:70]}...")
        pipeline.reset_history()
        result = pipeline.query(pregunta)
        respuesta = result["answer"]

        bloque = (
            f"[{i}] TAG: {tag}\n"
            f"PREGUNTA: {pregunta}\n"
            f"RESPUESTA:\n{respuesta}\n"
            f"{'-' * 65}\n\n"
        )
        print(f"RESPUESTA: {respuesta[:200]}...\n")
        lines.append(bloque)

    output_file.write_text("".join(lines), encoding="utf-8")
    print("=" * 65)
    print(f"✓ Resultados guardados en: {output_file}")
    print("  Revisa el archivo para evaluación manual.")


if __name__ == "__main__":
    main()
