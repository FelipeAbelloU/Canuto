"""Revision del dataset antes de entrenar: conteos y problemas que deben dar 0.

Lee dataset_alpaca.json, train.json y test.json. No cambia nada.

Uso:
    python scripts/revisar_dataset.py
"""
import json
import re
import sys
from collections import Counter, defaultdict
from pathlib import Path

RAIZ = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(RAIZ))

from src.dataset.qa_builder import enmascarar

CARPETA = RAIZ / "data/dataset"

# El tipo de negativo se reconoce por su respuesta (ver generate_negatives y
# generate_generales en qa_builder.py).
TIPOS_NEGATIVO = [
    ("articulo que no existe", "ese artículo no exista"),
    ("otra universidad", "Solo conozco la normativa de la Universidad de los Llanos."),
    ("dato fuera de la normativa", "consultar directamente con la dependencia"),
    ("datos privados", "datos privados o de seguridad"),
    ("cultura general", "solo respondo sobre la normativa"),
]

# "No tengo informacion sobre la X No. 5 de 2018; ... Sí conozco la X No. 5 de 2016, por si..."
SUGERENCIA_RE = re.compile(r"sobre (?:el|la) (.+?) No\. \S+ de (\d{4});.* conozco (?:el|la) (.+?) No\. \S+ de (\d{4})")
A1_RE = re.compile(r"^¿Qué establece el artículo (\d+) (?:del|de la) (.+?)(?: de Unillanos)?(?: \(parte \d+ de \d+\))?\?$")
PALABRA_RE = re.compile(r"\w+")
VOCAL_RE = re.compile(r"[aeiouáéíóúüy]", re.IGNORECASE)


def cargar(nombre):
    with open(CARPETA / nombre, encoding="utf-8") as f:
        return json.load(f)


def tipo_negativo(par):
    for nombre, frase in TIPOS_NEGATIVO:
        if frase in par["output"]:
            return nombre
    # Con sugerencia: mismo tipo y otro año, o el mismo numero y año con otro tipo
    m = SUGERENCIA_RE.search(par["output"])
    if m:
        return "año equivocado" if m.group(2) != m.group(4) else "tipo equivocado"
    # Sin sugerencia: otro numero o un documento imposible
    if par["instruction"].startswith("¿Qué dispone"):
        return "documento imposible"
    return "numero equivocado"


def palabras_raras(texto):
    """Palabras que el OCR suele dañar: sin vocales o con letras y numeros mezclados."""
    raras = []
    for p in PALABRA_RE.findall(texto):
        if re.fullmatch(r"[ABC][12]", p):   # niveles de ingles
            continue
        letras = re.sub(r"[\d_]", "", p)
        if len(p) >= 4 and len(letras) >= 3 and not VOCAL_RE.search(letras):
            raras.append(p)
        elif re.search(r"[a-zA-Z]\d|\d[a-zA-Z]{2,}", p) and not re.fullmatch(r"\d+(?:st|nd|rd|th|o|a)", p):
            raras.append(p)
    return raras


def main():
    todo = cargar("dataset_alpaca.json")
    train = cargar("train.json")
    test = cargar("test.json")

    print("1. Pares por familia")
    print(f"   {'familia':<10}{'total':>8}{'train':>8}{'test':>8}")
    familias = [Counter(p["familia"] for p in d) for d in (todo, train, test)]
    for fam in ["A", "B", "E", "negativo"]:
        print(f"   {fam:<10}{familias[0][fam]:>8}{familias[1][fam]:>8}{familias[2][fam]:>8}")
    print(f"   {'total':<10}{len(todo):>8}{len(train):>8}{len(test):>8}")
    positivos = sum(1 for p in todo if p["familia"] != "negativo")
    print(f"   negativos por cada positivo: {familias[0]['negativo'] / positivos:.2f}")
    print(f"   prueba: {len(test) / len(todo):.1%} del total")

    print("\n2. Negativos por tipo")
    tipos = Counter(tipo_negativo(p) for p in todo if p["familia"] == "negativo")
    for nombre, n in tipos.most_common():
        print(f"   {nombre:<28}{n:>7}")

    print("\n3. Largo de las respuestas (caracteres)")
    for fam in ["A", "B", "E", "negativo"]:
        largos = [len(p["output"]) for p in todo if p["familia"] == fam]
        print(f"   {fam:<10} min {min(largos):>5}  promedio {sum(largos) / len(largos):>6.0f}  max {max(largos):>5}")
    partes = sum(1 for p in todo if "(parte " in p["instruction"])
    tablas = sum(1 for p in todo if "incluye una tabla" in p["output"])
    reservados = sum(1 for p in todo if "(dato reservado)" in p["output"])
    print(f"   preguntas de un articulo partido '(parte i de n)': {partes}")
    print(f"   respuestas con aviso de tabla quitada: {tablas}")
    print(f"   respuestas con '(dato reservado)': {reservados}")

    print("\n4. Problemas (todos deben dar 0)")
    preguntas = Counter(p["instruction"] for p in todo)
    respuestas = defaultdict(set)
    for p in todo:
        respuestas[p["instruction"]].add(p["output"])
    en_train = {p["instruction"] for p in train}
    print(f"   preguntas repetidas: {sum(1 for n in preguntas.values() if n > 1)}")
    print(f"   misma pregunta con dos respuestas: {sum(1 for r in respuestas.values() if len(r) > 1)}")
    print(f"   preguntas de prueba que estan en train: {sum(1 for p in test if p['instruction'] in en_train)}")
    print(f"   respuestas vacias: {sum(1 for p in todo if not p['output'].strip())}")
    print(f"   cedulas/pasaportes/codigos sin tapar: {sum(1 for p in todo if enmascarar(p['output']) != p['output'])}")

    print("\n5. Sospechas (para revisar a mano, no son errores seguros)")
    # Texto del OCR dañado: 3 o mas palabras raras y al menos 2 % de las palabras.
    danados = []
    for p in todo:
        if p["familia"] == "negativo":
            continue
        raras = palabras_raras(p["output"])
        total = len(PALABRA_RE.findall(p["output"]))
        if len(raras) >= 3 and len(raras) / total >= 0.02:
            danados.append((p, raras))
    print(f"   respuestas con posible texto dañado por el OCR: {len(danados)} "
          f"de {positivos} positivos ({len(danados) / positivos:.1%})")
    for p, raras in danados[:5]:
        print(f"      {p['instruction'][:80]}  ->  {' '.join(raras[:6])}")

    # Articulos citados de otra norma tomados como propios: el numero salta (1, 2, 30)
    # en un documento que modifica otro.
    articulos = defaultdict(set)
    modifica = set()
    for p in todo:
        m = A1_RE.match(p["instruction"])
        if p["familia"] == "A" and m:
            articulos[m.group(2)].add(int(m.group(1)))
            if re.search(r"modif[ií]c", p["output"], re.IGNORECASE):
                modifica.add(m.group(2))
    saltos = []
    for cita in modifica:
        nums = sorted(articulos[cita])
        for anterior, n in zip(nums, nums[1:]):
            if n - anterior > 5:
                saltos.append((cita, anterior, n))
    print(f"   documentos que modifican otro con un salto grande en los articulos: "
          f"{len({s[0] for s in saltos})} ({len(saltos)} saltos)")
    for cita, anterior, n in sorted(saltos)[:5]:
        print(f"      {cita}: articulo {anterior} -> {n}")


if __name__ == "__main__":
    main()
