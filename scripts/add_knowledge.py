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

    # ── Fraccionamiento de matrícula (Res. Rectoral 074/2026) ──────────────
    {
        "instruction": "¿Cómo funciona el fraccionamiento de matrícula en Unillanos?",
        "input": "",
        "output": (
            "El fraccionamiento de matrícula está reglamentado por la Resolución Rectoral "
            "No. 074 de 2026 (23 de enero de 2026), autorizada por la Resolución Superior "
            "No. 002 de 2026.\n\n"
            "El valor total de los derechos de matrícula puede pagarse en tres (3) cuotas:\n"
            "1. Primer pago: 30% del valor total de derechos de matrícula + pago completo "
            "de derechos complementarios y servicios.\n"
            "2. Segundo pago: 40% del valor total.\n"
            "3. Tercer pago: 30% del valor total.\n\n"
            "Criterios para acceder: el estudiante NO debe tener exoneración del 100%, "
            "no debe ser beneficiario de la política de gratuidad, y no debe tener descuento "
            "socioeconómico del 50%.\n\n"
            "Se requiere firmar un PAGARÉ con un AVALISTA mayor de edad con ingresos "
            "superiores a 1.5 SMMLV.\n\n"
            "En caso de mora se cobran intereses moratorios según el artículo 884 del "
            "Código de Comercio. Esta medida fue transitoria para el periodo académico 2026-1."
        ),
    },
    {
        "instruction": "¿En cuántas cuotas puedo pagar la matrícula en Unillanos?",
        "input": "",
        "output": (
            "Según la Resolución Rectoral No. 074 de 2026, la matrícula puede pagarse "
            "en tres (3) cuotas bajo la modalidad de fraccionamiento:\n"
            "1. Primera cuota: 30% del valor total + pago completo de derechos complementarios.\n"
            "2. Segunda cuota: 40% del valor total.\n"
            "3. Tercera cuota: 30% del valor total.\n\n"
            "Para acceder a este mecanismo el estudiante no debe tener gratuidad, exoneración "
            "total ni descuento socioeconómico del 50%. Además debe firmar un PAGARÉ con "
            "un avalista con ingresos superiores a 1.5 SMMLV.\n\n"
            "Esta medida fue transitoria para el periodo académico 2026-1."
        ),
    },
    {
        "instruction": "¿Cuál es la resolución que reglamenta el fraccionamiento de matrícula en Unillanos?",
        "input": "",
        "output": (
            "El fraccionamiento de matrícula en Unillanos está reglamentado por la "
            "Resolución Rectoral No. 074 de 2026 (23 de enero de 2026), \"Por medio del cual "
            "se reglamenta el fraccionamiento del pago de los derechos de matrícula estudiantes "
            "de pregrado de la Universidad de los Llanos\".\n\n"
            "Esta resolución fue expedida en el marco de la autorización dada por la "
            "Resolución Superior No. 002 de 2026 del Consejo Superior Universitario, "
            "y aplica de forma transitoria para el periodo académico 2026-1."
        ),
    },
    {
        "instruction": "¿Quién puede acceder al fraccionamiento de matrícula en Unillanos?",
        "input": "",
        "output": (
            "Según la Resolución Rectoral No. 074 de 2026, pueden acceder al fraccionamiento "
            "los estudiantes de pregrado que cumplan los tres criterios:\n"
            "1. No tener exoneración del 100% en los derechos de matrícula.\n"
            "2. No ser beneficiario de la política de gratuidad.\n"
            "3. No haber accedido al descuento socioeconómico del 50% (Acuerdo Académico 001 de 2024).\n\n"
            "Adicionalmente se debe firmar un PAGARÉ y contar con un AVALISTA mayor de edad "
            "con ingresos superiores a 1.5 Salarios Mínimos Mensuales Legales Vigentes (SMMLV)."
        ),
    },

    # ── Incapacidad médica (Res. Rectoral 068/2025) ────────────────────────
    {
        "instruction": "¿Qué debo hacer si tengo una incapacidad médica en Unillanos?",
        "input": "",
        "output": (
            "El procedimiento está reglamentado por la Resolución Rectoral No. 0068 de 2025 "
            "(27 de enero de 2025). Los pasos son:\n\n"
            "1. Llevar la incapacidad al Centro Médico de Unillanos para que sea transcrita. "
            "Solo se transcriben incapacidades de médicos registrados en el ReThus (Registro "
            "Único Nacional del Talento Humano en Salud). No se transcriben incapacidades "
            "abiertas sin tiempo de duración.\n\n"
            "2. Puedes presentarla de dos maneras:\n"
            "   - Presencialmente: tú mismo o un tercero la lleva al Centro Médico.\n"
            "   - Virtualmente: desde tu cuenta institucional al correo "
            "incapacidad@unillanos.edu.co\n\n"
            "3. Plazos para presentarla al Centro Médico:\n"
            "   - Sin hospitalización: hasta 3 días hábiles después de expedida la incapacidad.\n"
            "   - Con hospitalización: hasta 3 días hábiles después de recibir el certificado "
            "de la EPS o IPS.\n\n"
            "4. Una vez transcrita, preséntala al programa académico por correo electrónico "
            "hasta 3 días hábiles después de emitida por el Centro Médico.\n\n"
            "5. Para citas, exámenes, procedimientos o terapias: informa PREVIAMENTE a la "
            "dirección del programa y presenta el soporte hasta 3 días hábiles después del "
            "procedimiento, indicando fecha, hora y duración."
        ),
    },
    {
        "instruction": "¿Cómo presento una incapacidad médica en Unillanos?",
        "input": "",
        "output": (
            "Según la Resolución Rectoral No. 0068 de 2025, para presentar una incapacidad "
            "médica en Unillanos debes:\n\n"
            "Paso 1: Llevarla al Centro Médico de Unillanos para transcripción. Solo se aceptan "
            "incapacidades de médicos registrados en el ReThus, en letra legible y con tiempo "
            "de duración definido.\n\n"
            "Paso 2: Puedes hacerlo presencialmente o de forma virtual desde tu cuenta "
            "institucional al correo incapacidad@unillanos.edu.co\n\n"
            "Paso 3: Presentarla al Centro Médico dentro de los 3 días hábiles siguientes "
            "a la fecha de expedición (o finalización de hospitalización).\n\n"
            "Paso 4: Una vez transcrita por el Centro Médico, enviarla al programa académico "
            "por correo electrónico dentro de los 3 días hábiles siguientes a su emisión."
        ),
    },
    {
        "instruction": "¿Cuál es la resolución sobre incapacidades médicas en Unillanos?",
        "input": "",
        "output": (
            "Las incapacidades y constancias médicas de los estudiantes de Unillanos están "
            "reglamentadas por la Resolución Rectoral No. 0068 de 2025 (27 de enero de 2025), "
            "\"Por el cual se reglamentan las incapacidades y constancias médicas para los "
            "estudiantes de pregrado y posgrado de la Universidad de los Llanos\".\n\n"
            "Esta resolución fue expedida en uso de las atribuciones del Rector, con base en "
            "la Ley 30 de 1992 y el Acuerdo Superior 003 de 2021."
        ),
    },
    {
        "instruction": "¿Cuánto tiempo tengo para presentar una incapacidad médica en Unillanos?",
        "input": "",
        "output": (
            "Según la Resolución Rectoral No. 0068 de 2025:\n\n"
            "- Sin hospitalización: tienes hasta 3 días hábiles después de la fecha de "
            "expedición de la incapacidad para presentarla al Centro Médico de Unillanos.\n\n"
            "- Con hospitalización: una vez termine el periodo de hospitalización, debes "
            "solicitar el certificado de incapacidad y presentarlo al Centro Médico dentro "
            "de los 3 días hábiles siguientes.\n\n"
            "- Citas o procedimientos médicos: debes informar PREVIAMENTE al programa académico "
            "y presentar el soporte hasta 3 días hábiles después del procedimiento.\n\n"
            "Después de que el Centro Médico emita la transcripción, tienes 3 días hábiles "
            "adicionales para enviarla al programa académico por correo electrónico."
        ),
    },

    # ── Segunda lengua — Plan BULL (Acuerdo Académico 003/2023) ───────────
    {
        "instruction": "¿Cuáles son los requisitos de segunda lengua para graduarse en Unillanos?",
        "input": "",
        "output": (
            "El requisito de segunda lengua está establecido en el Acuerdo Académico No. 003 "
            "de 2023 (Plan BULL — Bilingüismo Universidad de los Llanos).\n\n"
            "Para programas profesionales de pregrado (admitidos desde 2015):\n"
            "- Acreditar suficiencia en inglés en nivel B2, o\n"
            "- Cursar y aprobar los Módulos III y IV del Plan BULL.\n\n"
            "Para continuidad de estudios (al aprobar el 60% de créditos): nivel A2 o "
            "Módulos I y II del Plan BULL.\n\n"
            "Los módulos del Plan BULL son cuatro (I, II, III, IV), cada uno con 64 horas "
            "presenciales + 64 horas de trabajo independiente. Son gratuitos para estudiantes "
            "de Unillanos.\n\n"
            "Opciones de suficiencia:\n"
            "- Prueba de clasificación del Plan BULL.\n"
            "- Certificados internacionales: IELTS (mín. 6.5 para B2), TOEFL IBT (87+), "
            "TOEIC, Cambridge, entre otros.\n"
            "- Resultados de Saber Pro o Saber 11.\n\n"
            "Exentos: admitidos antes del segundo semestre de 2010; minorías lingüísticas "
            "(indígenas, ROM, raizales, extranjeros no hispano-parlantes, sordos); personas "
            "con diversidad funcional visual o cognitiva."
        ),
    },
    {
        "instruction": "¿Es obligatorio el requisito de segunda lengua para graduarse en Unillanos?",
        "input": "",
        "output": (
            "Sí, el requisito de segunda lengua ES OBLIGATORIO para graduarse en Unillanos, "
            "según el Acuerdo Académico No. 003 de 2023.\n\n"
            "Para programas profesionales (admitidos desde 2015): debes acreditar nivel B2 "
            "en inglés o aprobar los Módulos III y IV del Plan BULL (gratuitos).\n\n"
            "Solo están exentos: estudiantes admitidos antes del segundo semestre de 2010; "
            "minorías lingüísticas (indígenas, ROM, raizales, extranjeros no hispano-parlantes, "
            "estudiantes sordos); y personas con diversidad funcional visual o cognitiva "
            "demostrable."
        ),
    },
    {
        "instruction": "¿Qué es el Plan BULL en Unillanos?",
        "input": "",
        "output": (
            "El Plan BULL (Bilingüismo Universidad de los Llanos) es el programa de segunda "
            "lengua de Unillanos, reglamentado por el Acuerdo Académico No. 003 de 2023.\n\n"
            "Consiste en cuatro módulos de inglés (I, II, III y IV), cada uno con 64 horas "
            "presenciales y 64 horas de trabajo independiente. Son gratuitos para los "
            "estudiantes de Unillanos.\n\n"
            "Para graduarse en programas profesionales (admitidos desde 2015) se requiere "
            "aprobar los Módulos III y IV o acreditar nivel B2 por otra vía.\n\n"
            "También se puede demostrar suficiencia con: certificados internacionales "
            "(IELTS, TOEFL, TOEIC, Cambridge, etc.), resultados de Saber Pro o Saber 11."
        ),
    },
    {
        "instruction": "¿Qué pasa si pierdo un módulo del Plan BULL?",
        "input": "",
        "output": (
            "Según el Acuerdo Académico No. 003 de 2023, un módulo del Plan BULL se pierde "
            "si el estudiante:\n"
            "- Tiene inasistencia superior al 20% del total de horas del módulo, o\n"
            "- Obtiene una calificación inferior a 3.5.\n\n"
            "El estudiante puede cancelar el módulo dentro de las fechas establecidas en el "
            "calendario académico del Plan BULL del Centro de Idiomas, sin que esto cuente "
            "como pérdida."
        ),
    },
    {
        "instruction": "¿Quiénes están exentos del requisito de segunda lengua en Unillanos?",
        "input": "",
        "output": (
            "Según el Acuerdo Académico No. 003 de 2023, están exentos del requisito de "
            "segunda lengua en Unillanos:\n\n"
            "1. Estudiantes admitidos antes del segundo semestre de 2010 (quedan exentos "
            "tanto para continuidad como para grado).\n\n"
            "2. Estudiantes admitidos entre 2010 y el segundo semestre de 2014: solo deben "
            "aprobar el Módulo II o acreditar nivel A2 como mínimo para grado.\n\n"
            "3. Minorías lingüísticas: indígenas, ROM, raizales, extranjeros no "
            "hispano-parlantes y estudiantes con pérdida total de la audición o sordera "
            "demostrable. Quedan exentos de inglés pero deben cursar 4 módulos de español.\n\n"
            "4. Diversidad funcional: estudiantes con discapacidad visual o cognitiva "
            "demostrable."
        ),
    },

    # ── Programas Facultad de Ciencias Básicas e Ingeniería ───────────────
    {
        "instruction": "¿Cuáles son los programas de pregrado de la Facultad de Ciencias Básicas e Ingeniería de Unillanos?",
        "input": "",
        "output": (
            "Según la Resolución No. 061 de 2025 del Consejo de Facultad de Ciencias Básicas "
            "e Ingeniería (Acta No. 049 del 06 de noviembre de 2025), los programas de "
            "pregrado adscritos a esta facultad son:\n\n"
            "1. Biología\n"
            "2. Ingeniería Ambiental\n"
            "3. Ingeniería de Procesos\n"
            "4. Ingeniería de Sistemas\n"
            "5. Ingeniería Electrónica"
        ),
    },
    {
        "instruction": "¿Ingeniería de Sistemas es un programa de la Facultad de Ciencias Básicas e Ingeniería?",
        "input": "",
        "output": (
            "Sí. Ingeniería de Sistemas es uno de los programas de pregrado de la Facultad "
            "de Ciencias Básicas e Ingeniería de la Universidad de los Llanos, según la "
            "Resolución No. 061 de 2025 del Consejo de Facultad.\n\n"
            "Los programas de esa facultad son: Biología, Ingeniería Ambiental, Ingeniería "
            "de Procesos, Ingeniería de Sistemas e Ingeniería Electrónica."
        ),
    },

    # ── Opciones de grado FCBI (Res. CF-061/2025) ─────────────────────────
    {
        "instruction": "¿Qué opciones de grado existen en la Facultad de Ciencias Básicas e Ingeniería?",
        "input": "",
        "output": (
            "Según la Resolución No. 061 de 2025 del Consejo de Facultad de Ciencias Básicas "
            "e Ingeniería (noviembre 6 de 2025), las opciones de grado disponibles son:\n\n"
            "1. Trabajo de grado\n"
            "2. Monografía (revisión de literatura sistemática o meta-análisis)\n"
            "3. Desarrollo Tecnológico\n"
            "4. EPI (Emprendimiento e Innovación)\n"
            "5. EPPS (Empresa de Base)\n"
            "6. Pasantía (individual; Ingeniería de Sistemas requiere promedio ≥ 3.5)\n"
            "7. Plan de negocio (solo para el programa de Biología)\n"
            "8. Aprobación de créditos académicos a nivel de posgrado (requiere promedio "
            "≥ 3.5 y al menos el 80% de créditos aprobados; mínimo 10 créditos de "
            "especialización u 8 de maestría/doctorado)\n"
            "9. Artículo científico aprobado en revista indexada u homologada por MinCiencias\n\n"
            "Requisito para inscribirse: tener matrícula vigente y haber aprobado al menos "
            "el 70% de los créditos del plan de estudios.\n\n"
            "No se contemplan como opciones: procedimientos técnicos, promedio destacado "
            "ni cursos de educación continuada."
        ),
    },
    {
        "instruction": "¿Cuáles son los requisitos para inscribirse en una opción de grado en la Facultad de Ciencias Básicas e Ingeniería?",
        "input": "",
        "output": (
            "Según la Resolución No. 061 de 2025 del Consejo de Facultad, para inscribirse "
            "en una opción de grado en la Facultad de Ciencias Básicas e Ingeniería se requiere:\n\n"
            "1. Tener matrícula vigente.\n"
            "2. Haber cursado y aprobado al menos el 70% de los créditos académicos del "
            "plan de estudios vigente.\n"
            "3. La temática debe ser afín a las líneas de profundización o investigación "
            "del programa.\n"
            "4. No se puede presentar más de una propuesta de opción de grado simultáneamente.\n"
            "5. La opción puede ser desarrollada por máximo dos estudiantes (excepto Pasantía, "
            "que es individual).\n\n"
            "Requisitos adicionales según la opción:\n"
            "- Pasantía en Ing. Sistemas: promedio ≥ 3.5.\n"
            "- Créditos a nivel posgrado: promedio ≥ 3.5 y 80% de créditos aprobados.\n"
            "- Artículo científico: requiere un docente coautor vinculado al Sistema Nacional "
            "de Ciencia y Tecnología."
        ),
    },

    # ── Movilidad estudiantil saliente ─────────────────────────────────────
    {
        "instruction": "¿Cuáles son los requisitos para aplicar a movilidad estudiantil saliente en Unillanos?",
        "input": "",
        "output": (
            "Según el Acuerdo Académico No. 014 de 2021 y la Convocatoria de Movilidad "
            "Saliente 2026-2 de la Oficina de Internacionalización (OIRI), los requisitos "
            "para movilidad presencial de pregrado son:\n\n"
            "A. Estar matriculado en un programa académico de Unillanos.\n"
            "B. Tener promedio ponderado de carrera igual o superior a 3.5.\n"
            "C. Haber aprobado el 50% del plan de estudios.\n"
            "D. Tener aval de movilidad del Comité de Programa.\n"
            "E. Aplicar dentro de los términos de la convocatoria.\n\n"
            "Para acceder al apoyo económico: promedio ≥ 3.7.\n\n"
            "Montos de apoyo (Resolución Superior 029 de 2025):\n"
            "- Nacionales: hasta 4 SMMLV.\n"
            "- Centroamérica, Caribe y Suramérica: hasta 6 SMMLV.\n"
            "- Estados Unidos y Canadá: hasta 7 SMMLV.\n"
            "- Europa, Asia, África y Oceanía: hasta 8 SMMLV.\n\n"
            "Contacto OIRI: movilidad.academica@unillanos.edu.co | Tel: (608) 6611623 Ext 159"
        ),
    },
    {
        "instruction": "¿Qué promedio necesito para aplicar a movilidad estudiantil en Unillanos?",
        "input": "",
        "output": (
            "Según el Acuerdo Académico No. 014 de 2021:\n\n"
            "- Para aplicar a movilidad presencial de pregrado: promedio ponderado ≥ 3.5.\n"
            "- Para aplicar a movilidad en línea (virtual): promedio ponderado ≥ 3.4.\n"
            "- Para acceder al apoyo económico: promedio ponderado ≥ 3.7.\n\n"
            "Además de cumplir el promedio, se requiere haber aprobado el 50% del plan de "
            "estudios (30% para modalidad virtual) y tener aval del Comité de Programa."
        ),
    },

    # ── Corrección de información falsa (preguntas con trampa) ────────────
    {
        "instruction": "Un amigo me dijo que la segunda lengua no es obligatoria para graduarse en Unillanos, ¿es verdad?",
        "input": "",
        "output": (
            "Esa información es incorrecta. Según el Acuerdo Académico No. 003 de 2023, "
            "el requisito de segunda lengua SÍ es obligatorio para graduarse en Unillanos "
            "para los estudiantes admitidos desde el año 2015.\n\n"
            "Para programas profesionales de pregrado debes acreditar nivel B2 en inglés "
            "o aprobar los Módulos III y IV del Plan BULL (gratuitos).\n\n"
            "Las únicas excepciones son: admitidos antes del segundo semestre de 2010, "
            "minorías lingüísticas (indígenas, ROM, raizales, extranjeros no hispano-parlantes, "
            "sordos) y personas con diversidad funcional visual o cognitiva."
        ),
    },
    {
        "instruction": "¿Es cierto que la matrícula en Unillanos cuesta 5 millones de pesos?",
        "input": "",
        "output": (
            "Esa información no es correcta. El valor de la matrícula en Unillanos no tiene "
            "un precio fijo y no existe un valor único para todos los estudiantes.\n\n"
            "Se calcula de forma personalizada según la situación socioeconómica: nivel de "
            "SISBEN, ingresos económicos, puntaje ICFES y estrato socioeconómico.\n\n"
            "No tengo información sobre valores específicos de matrícula en los documentos "
            "que conozco. Para conocer el valor exacto que te corresponde, consulta con la "
            "Oficina de Admisiones, Registro y Control Académico de Unillanos."
        ),
    },
    {
        "instruction": "Me dijeron que Unillanos tiene sede en Bogotá, ¿es verdad?",
        "input": "",
        "output": (
            "No tengo información sobre sedes adicionales en Bogotá en los documentos que "
            "conozco. Según la información disponible, la Universidad de los Llanos tiene su "
            "campus principal en Villavicencio, Meta. Para información sobre sedes oficiales, "
            "consulta el sitio web de Unillanos: www.unillanos.edu.co"
        ),
    },

    # ── Preguntas totalmente fuera del dominio universitario ──────────────
    {
        "instruction": "¿Cuál es la receta del ajiaco colombiano?",
        "input": "",
        "output": (
            "No puedo ayudarte con eso. Soy CANUTO, asistente especializado en normativa "
            "universitaria de la Universidad de los Llanos, y solo respondo preguntas "
            "relacionadas con reglamentos, resoluciones y documentos normativos de Unillanos.\n\n"
            "Para recetas de cocina, te recomiendo consultar sitios especializados en gastronomía."
        ),
    },
    {
        "instruction": "¿Quién es el presidente de Colombia?",
        "input": "",
        "output": (
            "Soy CANUTO, asistente de normativa universitaria de Unillanos, y no tengo "
            "información sobre política nacional en los documentos que conozco.\n\n"
            "Para información sobre el gobierno colombiano, consulta fuentes oficiales "
            "como www.presidencia.gov.co"
        ),
    },
    {
        "instruction": "¿Cuál es el equipo de fútbol más popular de Colombia?",
        "input": "",
        "output": (
            "No puedo ayudarte con eso. Soy CANUTO, asistente especializado exclusivamente "
            "en normativa universitaria de la Universidad de los Llanos.\n\n"
            "Para temas de fútbol y deportes, consulta medios deportivos especializados."
        ),
    },
    {
        "instruction": "¿Cómo se prepara un café colombiano?",
        "input": "",
        "output": (
            "Esa pregunta está fuera de mi dominio. Soy CANUTO, asistente de normativa "
            "universitaria de Unillanos, y solo respondo preguntas sobre reglamentos, "
            "resoluciones y documentos normativos de la universidad.\n\n"
            "Para temas de gastronomía y bebidas, consulta fuentes especializadas."
        ),
    },
    {
        "instruction": "¿Cuánto vale un pasaje de Villavicencio a Bogotá?",
        "input": "",
        "output": (
            "No tengo información sobre transporte ni precios de pasajes en los documentos "
            "que conozco. Soy CANUTO, un asistente especializado en normativa universitaria "
            "de Unillanos. Para esa información, consulta empresas de transporte o "
            "plataformas de viajes."
        ),
    },
    {
        "instruction": "¿Cuál es el clima en Villavicencio hoy?",
        "input": "",
        "output": (
            "No tengo acceso a información del clima ni datos en tiempo real. Soy CANUTO, "
            "asistente de normativa universitaria de Unillanos, y mi conocimiento proviene "
            "exclusivamente de documentos normativos de la universidad.\n\n"
            "Para el clima, consulta el IDEAM o aplicaciones meteorológicas."
        ),
    },
    {
        "instruction": "¿Cómo solicito una visa para estudiar en el exterior?",
        "input": "",
        "output": (
            "No tengo información sobre trámites de visas en los documentos que conozco. "
            "Para información sobre visas de estudio, consulta la Cancillería de Colombia "
            "(www.cancilleria.gov.co) o el consulado del país de destino.\n\n"
            "Si tu pregunta es sobre movilidad estudiantil de Unillanos, puedo ayudarte "
            "con eso — pregúntame sobre los requisitos para movilidad saliente."
        ),
    },
    {
        "instruction": "¿Cuánto cuesta un apartamento en Villavicencio?",
        "input": "",
        "output": (
            "Esa información está fuera de mi dominio. Soy CANUTO, asistente de normativa "
            "universitaria de Unillanos, y solo tengo información sobre resoluciones, "
            "acuerdos y documentos normativos de la universidad."
        ),
    },
    {
        "instruction": "¿Qué debo hacer para renovar mi cédula de ciudadanía?",
        "input": "",
        "output": (
            "No tengo información sobre trámites de documentos de identidad en los "
            "documentos que conozco. Para renovar tu cédula, consulta la Registraduría "
            "Nacional del Estado Civil (www.registraduria.gov.co).\n\n"
            "Para trámites académicos de Unillanos, con gusto puedo ayudarte."
        ),
    },
    {
        "instruction": "¿Cómo funciona el sistema de salud en Colombia?",
        "input": "",
        "output": (
            "No tengo información sobre el sistema de salud colombiano en los documentos "
            "que conozco. Soy CANUTO, asistente de normativa universitaria de Unillanos, "
            "y solo respondo preguntas sobre reglamentos y documentos normativos de la "
            "universidad."
        ),
    },

    # ── Datos inventados — el modelo debe reconocer su límite ─────────────
    {
        "instruction": "¿Cuál es el correo del decano de la Facultad de Ciencias Básicas e Ingeniería?",
        "input": "",
        "output": (
            "No tengo información sobre datos de contacto del personal universitario en los "
            "documentos que conozco, ya que estos pueden cambiar con frecuencia.\n\n"
            "Puedes consultar el directorio oficial de Unillanos en www.unillanos.edu.co "
            "o comunicarte directamente con la facultad."
        ),
    },
    {
        "instruction": "¿Cuántos estudiantes tiene Unillanos actualmente?",
        "input": "",
        "output": (
            "No tengo información sobre el número actual de estudiantes matriculados en los "
            "documentos que conozco. Para datos estadísticos actualizados puedes consultar "
            "los informes de gestión en el sitio web oficial de Unillanos: www.unillanos.edu.co"
        ),
    },
    {
        "instruction": "¿Cuál es el correo del rector de Unillanos?",
        "input": "",
        "output": (
            "No tengo información sobre datos de contacto del personal universitario en los "
            "documentos que conozco. Para contactar a las directivas de la universidad, "
            "consulta el directorio oficial en www.unillanos.edu.co"
        ),
    },
    {
        "instruction": "¿Cuál es el correo electrónico de la oficina de bienestar universitario de Unillanos?",
        "input": "",
        "output": (
            "No tengo información sobre datos de contacto institucional en los documentos que "
            "conozco. Puedes buscar esa información en el sitio web oficial de Unillanos "
            "(www.unillanos.edu.co) o acudir directamente a la Oficina de Bienestar Universitario."
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
