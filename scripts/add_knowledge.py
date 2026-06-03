"""
scripts/add_knowledge.py — Agrega conocimiento curado al dataset de entrenamiento.

Ejecutar cada vez que se quiera añadir nuevos pares QA sin regenerar todo el dataset:
    python scripts/add_knowledge.py

Los pares se agregan al final de data/dataset/dataset_alpaca.json.
Si un par ya existe (misma instruction), se omite para evitar duplicados.
"""
import json
from pathlib import Path

DATASET_PATH = Path(__file__).parent.parent / "data/dataset/dataset_alpaca.json"

# ---------------------------------------------------------------------------
# PARES CURADOS — editarlos o ampliarlos según el conocimiento del proyecto
# ---------------------------------------------------------------------------

NUEVOS_PARES = [

    # ── Valor de matrícula: conocimiento real ──────────────────────────────
    {
        "instruction": "¿Cuánto cuesta la matrícula en Unillanos?",
        "input": "",
        "output": (
            "El valor de la matrícula en Unillanos varía para cada estudiante, "
            "ya que se calcula de forma personalizada según su situación socioeconómica. "
            "Los factores que determinan el valor son:\n"
            "1. Nivel y puntaje del SISBEN del estudiante o su núcleo familiar.\n"
            "2. Ingresos económicos propios o del acudiente.\n"
            "3. Puntaje obtenido en las pruebas Saber 11 (ICFES).\n"
            "4. Estrato socioeconómico.\n"
            "5. Beneficios o descuentos especiales aplicables.\n\n"
            "Por esta razón no existe un valor único — cada estudiante recibe una "
            "liquidación personalizada. Para conocer el valor exacto que te "
            "corresponde, consulta con la Oficina de Admisiones, Registro y Control "
            "Académico de Unillanos."
        ),
    },
    {
        "instruction": "¿Por qué el valor de la matrícula es diferente para cada estudiante en Unillanos?",
        "input": "",
        "output": (
            "En Unillanos la matrícula se calcula de forma personalizada porque la "
            "universidad aplica un modelo de equidad: quienes tienen mayor capacidad "
            "económica pagan más, y quienes tienen menor capacidad pagan menos.\n\n"
            "Los factores que generan esa diferencia son:\n"
            "1. SISBEN: nivel y puntaje del sisben del estudiante o su familia.\n"
            "2. Ingresos económicos: propios o del acudiente.\n"
            "3. Puntaje ICFES (Saber 11).\n"
            "4. Estrato socioeconómico.\n"
            "5. Posibles descuentos o beneficios especiales.\n\n"
            "Para saber el valor exacto de tu matrícula, comunícate con la Oficina "
            "de Admisiones y Registro de Unillanos."
        ),
    },
    {
        "instruction": "¿Qué factores determinan el costo de la matrícula en Unillanos?",
        "input": "",
        "output": (
            "El costo de la matrícula en Unillanos se determina a partir de:\n"
            "1. SISBEN: nivel y puntaje del sisben del estudiante o su núcleo familiar.\n"
            "2. Ingresos económicos: propios o del acudiente.\n"
            "3. Puntaje Saber 11 (ICFES).\n"
            "4. Estrato socioeconómico.\n"
            "5. Beneficios o descuentos especiales vigentes.\n\n"
            "Con esa información, la Oficina de Admisiones, Registro y Control Académico "
            "realiza la liquidación personalizada de cada estudiante."
        ),
    },
    {
        "instruction": "¿El SISBEN afecta el valor de la matrícula en Unillanos?",
        "input": "",
        "output": (
            "Sí. El nivel y puntaje del SISBEN es uno de los factores que Unillanos "
            "tiene en cuenta para calcular el valor de la matrícula. "
            "Los estudiantes con puntajes de SISBEN más bajos (mayor vulnerabilidad "
            "económica) tienden a pagar un valor de matrícula menor.\n\n"
            "Junto con el SISBEN, también influyen los ingresos económicos, el puntaje "
            "ICFES y el estrato socioeconómico. Para conocer cómo aplica en tu caso "
            "específico, consulta con la Oficina de Admisiones y Registro de Unillanos."
        ),
    },
    {
        "instruction": "¿Cómo puedo saber cuánto voy a pagar de matrícula en Unillanos?",
        "input": "",
        "output": (
            "El valor de tu matrícula se calcula de forma personalizada y depende de "
            "tu situación socioeconómica. No existe un valor fijo único.\n\n"
            "Los factores que lo determinan son: tu nivel de SISBEN, tus ingresos o "
            "los de tu acudiente, tu puntaje en Saber 11 (ICFES), tu estrato "
            "socioeconómico, y posibles descuentos o beneficios especiales.\n\n"
            "Para conocer el valor exacto que te corresponde, acércate o comunícate "
            "con la Oficina de Admisiones, Registro y Control Académico de Unillanos."
        ),
    },

    # ── Preguntas fuera del dominio — el modelo debe admitir que no sabe ──
    {
        "instruction": "¿Cuándo inician las clases en Unillanos para el próximo semestre?",
        "input": "",
        "output": (
            "No tengo información sobre las fechas de inicio de clases en los "
            "documentos que conozco. Te recomiendo consultar el calendario académico "
            "oficial en el sitio web de Unillanos o comunicarte con la Oficina de "
            "Admisiones y Registro."
        ),
    },
    {
        "instruction": "¿Cuál es el horario de atención de la biblioteca de Unillanos?",
        "input": "",
        "output": (
            "No tengo información sobre ese tema en los documentos que conozco. "
            "Te recomiendo consultar directamente con la Biblioteca de Unillanos "
            "o revisar su sitio web oficial."
        ),
    },
    {
        "instruction": "¿Cuántos créditos tiene el programa de Ingeniería de Sistemas en Unillanos?",
        "input": "",
        "output": (
            "No tengo información sobre los planes de estudio ni el número de créditos "
            "de los programas académicos en los documentos que conozco. Para esa "
            "información, consulta el plan de estudios oficial en la página web de "
            "Unillanos o comunícate con la facultad correspondiente."
        ),
    },
    {
        "instruction": "¿Cuál es el correo electrónico de la Oficina de Admisiones de Unillanos?",
        "input": "",
        "output": (
            "No tengo información sobre datos de contacto en los documentos que "
            "conozco. Te recomiendo visitar el sitio web oficial de Unillanos "
            "(www.unillanos.edu.co) o acudir directamente a la Oficina de Admisiones."
        ),
    },
    {
        "instruction": "¿Cómo me inscribo a las materias en Unillanos?",
        "input": "",
        "output": (
            "No tengo información sobre el proceso de inscripción de materias en los "
            "documentos que conozco. Para esa información, consulta con la Oficina de "
            "Registro Académico o ingresa al sistema de información de Unillanos (SIRIUS)."
        ),
    },
    {
        "instruction": "¿Qué promedio académico necesito para obtener una beca en Unillanos?",
        "input": "",
        "output": (
            "No tengo información específica sobre los requisitos de promedio para "
            "becas en los documentos que conozco. Te recomiendo consultar con la "
            "Oficina de Bienestar Universitario de Unillanos."
        ),
    },
    {
        "instruction": "¿Cuándo vence el plazo para pagar la matrícula este semestre?",
        "input": "",
        "output": (
            "No tengo información sobre las fechas límite de pago de matrícula del "
            "semestre actual en los documentos que conozco. Consulta el calendario "
            "académico oficial o comunícate con la Oficina de Tesorería de Unillanos."
        ),
    },
    {
        "instruction": "¿Cómo solicito una certificación de notas en Unillanos?",
        "input": "",
        "output": (
            "No tengo información sobre ese proceso en los documentos que conozco. "
            "Te recomiendo consultar directamente con la Oficina de Registro y Control "
            "Académico de Unillanos o el sistema SIRIUS."
        ),
    },
    {
        "instruction": "¿Cuál es el número de teléfono de la Universidad de los Llanos?",
        "input": "",
        "output": (
            "No tengo información sobre datos de contacto en los documentos que "
            "conozco. Puedes buscar esa información en el sitio web oficial de "
            "Unillanos: www.unillanos.edu.co"
        ),
    },
    {
        "instruction": "¿Qué materias debo ver en tercer semestre de Ingeniería de Sistemas?",
        "input": "",
        "output": (
            "No tengo información sobre los planes de estudio específicos de los "
            "programas académicos en los documentos que conozco. Consulta el plan "
            "de estudios oficial en la página web de Unillanos o comunícate con la "
            "facultad correspondiente."
        ),
    },
    {
        "instruction": "¿En qué salón se dictan las clases de Cálculo I?",
        "input": "",
        "output": (
            "No tengo información sobre horarios ni asignación de salones en los "
            "documentos que conozco. Consulta tu horario en el sistema de información "
            "de Unillanos (SIRIUS)."
        ),
    },
    {
        "instruction": "¿Cuánto vale la inscripción para nuevos estudiantes en Unillanos?",
        "input": "",
        "output": (
            "No tengo información sobre el valor de la inscripción en los documentos "
            "que conozco. Te recomiendo consultar directamente con la Oficina de "
            "Admisiones de Unillanos o revisar la convocatoria oficial de admisiones."
        ),
    },
    # ── Variantes específicas de precio de matrícula (fuera de dominio) ──
    {
        "instruction": "¿Cuánto cuesta la matrícula en Unillanos para el semestre 2026-2?",
        "input": "",
        "output": (
            "El valor de la matrícula en Unillanos no tiene un precio fijo para ningún semestre. "
            "Se calcula de forma personalizada para cada estudiante según: nivel de SISBEN, "
            "ingresos económicos propios o del acudiente, puntaje ICFES y estrato socioeconómico.\n\n"
            "No tengo información sobre valores específicos de matrícula en los documentos que conozco. "
            "Para conocer el valor exacto que te corresponde para 2026-2, consulta con la "
            "Oficina de Admisiones, Registro y Control Académico de Unillanos."
        ),
    },
    {
        "instruction": "¿Cuánto cuesta la matrícula en Unillanos para el semestre 2025-1?",
        "input": "",
        "output": (
            "El valor de la matrícula en Unillanos varía para cada estudiante según su situación "
            "socioeconómica. No existe un valor único por semestre.\n\n"
            "No tengo información sobre valores específicos de matrícula en los documentos que conozco. "
            "Consulta con la Oficina de Admisiones y Registro de Unillanos para conocer "
            "el valor que te corresponde."
        ),
    },
    {
        "instruction": "¿Cuál es el valor de la matrícula para ingeniería en Unillanos?",
        "input": "",
        "output": (
            "El valor de la matrícula en Unillanos no depende del programa académico sino de "
            "la situación socioeconómica del estudiante: nivel de SISBEN, ingresos económicos, "
            "puntaje ICFES y estrato.\n\n"
            "No tengo información sobre valores específicos en los documentos que conozco. "
            "Consulta directamente con la Oficina de Admisiones y Registro de Unillanos."
        ),
    },
    {
        "instruction": "¿Unillanos tiene convenios con otras universidades?",
        "input": "",
        "output": (
            "No tengo información detallada sobre convenios interinstitucionales en "
            "los documentos que conozco. Para esa información, consulta con la Oficina "
            "de Relaciones Interinstitucionales de Unillanos."
        ),
    },
]


def main():
    # Cargar dataset existente
    with open(DATASET_PATH, encoding="utf-8") as f:
        dataset = json.load(f)

    instrucciones_existentes = {item["instruction"].strip().lower() for item in dataset}
    agregados = 0
    omitidos = 0

    for par in NUEVOS_PARES:
        clave = par["instruction"].strip().lower()
        if clave in instrucciones_existentes:
            print(f"  [omitido — ya existe] {par['instruction'][:60]}")
            omitidos += 1
        else:
            dataset.append(par)
            instrucciones_existentes.add(clave)
            print(f"  [agregado] {par['instruction'][:60]}")
            agregados += 1

    with open(DATASET_PATH, "w", encoding="utf-8") as f:
        json.dump(dataset, f, ensure_ascii=False, indent=2)

    print(f"\nDataset actualizado: {len(dataset)} pares totales")
    print(f"  Agregados: {agregados}  |  Omitidos (duplicados): {omitidos}")
    print(f"  Archivo: {DATASET_PATH}")


if __name__ == "__main__":
    main()
