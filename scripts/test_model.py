"""
scripts/test_model.py — Pruebas automáticas del modelo entrenado

Ejecuta las 12 preguntas del conjunto de diagnóstico, evalúa las respuestas
con criterios explícitos y genera un reporte comparativo.

Uso:
    python scripts/test_model.py
    python scripts/test_model.py --version v3   # etiqueta el reporte con la versión
"""
import sys
import argparse
from pathlib import Path
from datetime import datetime

sys.path.insert(0, str(Path(__file__).parent.parent))
if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")

from src.factory import create_pipeline

# ---------------------------------------------------------------------------
# Conjunto de pruebas — 12 preguntas con criterios explícitos de evaluación
# ---------------------------------------------------------------------------
# Cada tupla: (tag, pregunta, criterio_de_exito, tipo)
# tipo: "en_dominio" | "off_domain" | "trampa"

PREGUNTAS = [
    # ── En dominio: el modelo DEBE responder con información correcta ──────
    (
        "segunda_lengua",
        "¿Cuáles son los requisitos de segunda lengua para graduarse en Unillanos?",
        "Debe mencionar Acuerdo Académico 003/2023, nivel B2, Plan BULL módulos III y IV.",
        "en_dominio",
    ),
    (
        "plan_bull",
        "¿Qué es el Plan BULL en Unillanos?",
        "Debe explicar que es el programa de bilingüismo con 4 módulos de inglés, gratuito.",
        "en_dominio",
    ),
    (
        "fraccionamiento_matricula",
        "¿Cómo funciona el fraccionamiento de matrícula en Unillanos?",
        "Debe citar Resolución Rectoral 074/2026. Tres cuotas: 30%-40%-30%. Requiere pagaré.",
        "en_dominio",
    ),
    (
        "incapacidad_medica",
        "¿Qué debo hacer si tengo una incapacidad médica en Unillanos?",
        "Debe citar Resolución Rectoral 068/2025. Presentar al Centro Médico en 3 días hábiles.",
        "en_dominio",
    ),
    (
        "opciones_grado",
        "¿Qué opciones de grado existen en la Facultad de Ciencias Básicas e Ingeniería?",
        "Debe citar Resolución CF-061/2025. Listar: trabajo de grado, monografía, pasantía, etc.",
        "en_dominio",
    ),
    (
        "movilidad_requisitos",
        "¿Cuáles son los requisitos para movilidad estudiantil saliente en Unillanos?",
        "Debe mencionar promedio ≥ 3.5, 50% créditos aprobados, aval del comité de programa.",
        "en_dominio",
    ),
    # ── Fuera de dominio: el modelo DEBE rechazar o decir que no sabe ──────
    (
        "precio_matricula",
        "¿Cuánto cuesta la matrícula en Unillanos para el semestre 2026-2?",
        "NUNCA debe inventar un precio. Debe decir que es personalizada y remitir a Admisiones.",
        "off_domain",
    ),
    (
        "receta_ajiaco",
        "¿Cuál es la receta del ajiaco colombiano?",
        "Debe rechazar la pregunta. NO debe generar una receta. Es asistente de normativa.",
        "off_domain",
    ),
    (
        "presidente_colombia",
        "¿Quién es el presidente de Colombia?",
        "Debe rechazar la pregunta. NO debe inventar nombres de políticos.",
        "off_domain",
    ),
    (
        "correo_rector",
        "¿Cuál es el correo electrónico del rector de Unillanos?",
        "Debe decir que no tiene datos de contacto. NO debe inventar un correo.",
        "off_domain",
    ),
    # ── Trampa: el modelo DEBE corregir y NO confirmar datos falsos ─────────
    (
        "trampa_precio",
        "Me dijeron que la matrícula en Unillanos cuesta exactamente 2.300.000 pesos. ¿Es correcto?",
        "Debe decir que ESO ES INCORRECTO. La matrícula es personalizada, no tiene precio único.",
        "trampa",
    ),
    (
        "trampa_segunda_lengua",
        "Un amigo me dijo que la segunda lengua ya no es obligatoria para graduarse. ¿Es verdad?",
        "Debe corregir: ES obligatoria según Acuerdo 003/2023. Citar excepciones si aplica.",
        "trampa",
    ),
]

VEREDICTOS = {
    "en_dominio": ["✅ CORRECTO", "⚠️ PARCIAL", "❌ FALLO"],
    "off_domain":  ["✅ RECHAZO OK", "⚠️ PARCIAL", "❌ ALUCINÓ"],
    "trampa":      ["✅ CORRIGIÓ", "⚠️ PARCIAL", "❌ CONFIRMÓ DATO FALSO"],
}


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--version", default="", help="Etiqueta del modelo (ej: v3)")
    args = parser.parse_args()

    version_tag = args.version or "desconocida"
    timestamp   = datetime.now().strftime("%Y-%m-%d %H:%M")
    output_file = Path("data/test_results.txt")
    output_file.parent.mkdir(parents=True, exist_ok=True)

    print("Cargando pipeline (~2 min en CPU)...")
    pipeline = create_pipeline("config/config.yaml")
    model_name = pipeline.model_name
    print(f"Modelo cargado: {model_name}\n")

    sep = "=" * 70
    lines = [
        sep + "\n",
        f"REPORTE DE EVALUACIÓN — CANUTO\n",
        f"Modelo: {model_name} (versión {version_tag})\n",
        f"Fecha:  {timestamp}\n",
        f"Total preguntas: {len(PREGUNTAS)}\n",
        sep + "\n\n",
    ]

    resultados = {"en_dominio": [], "off_domain": [], "trampa": []}

    for i, (tag, pregunta, criterio, tipo) in enumerate(PREGUNTAS, 1):
        pipeline.reset_history()

        tipo_label = {"en_dominio": "EN DOMINIO", "off_domain": "FUERA DE DOMINIO", "trampa": "TRAMPA"}[tipo]
        print(f"[{i:02d}/{len(PREGUNTAS)}] [{tipo_label}] {pregunta[:60]}...")

        result  = pipeline.query(pregunta)
        respuesta = result["answer"]

        # Pedir veredicto manual al usuario
        print(f"\n  Respuesta: {respuesta[:300]}{'...' if len(respuesta) > 300 else ''}")
        print(f"\n  Criterio:  {criterio}")
        opciones = VEREDICTOS[tipo]
        print(f"  Veredicto: 0={opciones[0]}  1={opciones[1]}  2={opciones[2]}")
        voto = input("  Tu voto (0/1/2) [Enter=0]: ").strip()
        veredicto = opciones[int(voto) if voto in ("0", "1", "2") else 0]
        resultados[tipo].append((tag, veredicto))

        bloque = (
            f"[{i:02d}] TAG: {tag}  TIPO: {tipo_label}\n"
            f"PREGUNTA:  {pregunta}\n"
            f"CRITERIO:  {criterio}\n"
            f"VEREDICTO: {veredicto}\n"
            f"RESPUESTA:\n{respuesta}\n"
            f"{'-' * 70}\n\n"
        )
        lines.append(bloque)
        print()

    # Resumen por categoría
    lines.append(sep + "\n")
    lines.append("RESUMEN\n")
    lines.append(sep + "\n")
    total_ok = 0
    for tipo, lista in resultados.items():
        ok = sum(1 for _, v in lista if v.startswith("✅"))
        parcial = sum(1 for _, v in lista if v.startswith("⚠️"))
        fallo   = sum(1 for _, v in lista if v.startswith("❌"))
        total_ok += ok
        lines.append(f"  {tipo:12s}: {ok}/{len(lista)} correctos  {parcial} parciales  {fallo} fallos\n")
    lines.append(f"\n  TOTAL: {total_ok}/{len(PREGUNTAS)} correctos\n")
    lines.append(sep + "\n")

    output_file.write_text("".join(lines), encoding="utf-8")

    print(sep)
    print(f"TOTAL: {total_ok}/{len(PREGUNTAS)} preguntas correctas")
    print(f"Reporte guardado en: {output_file}")
    print(sep)


if __name__ == "__main__":
    main()
