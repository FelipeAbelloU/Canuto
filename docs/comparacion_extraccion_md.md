# Comparación de librerías PDF → Markdown (CANUTO)

**Fecha:** 2026-06-30
**Objetivo:** Elegir la librería de conversión PDF→Markdown para el corpus normativo de SIRIUS.
**Candidatas:** `pymupdf4llm`, `pdf2markdown4llm`, `docling`.

## Metodología

- Muestra: **1 PDF por carpeta hoja** del corpus (80 carpetas: 33 digitales, 47 escaneados).
- `pymupdf4llm` y `pdf2markdown4llm` se corrieron sobre **los 33 digitales** (rápidas).
- `docling` se corrió sobre un **subconjunto diverso** (6 digitales + 2 escaneados),
  por ser ~30× más lenta y requerir descarga de modelos de layout/OCR.
- Hardware de la prueba: laptop i5-7200U, 16 GB RAM, **sin GPU** (CPU only).
- Salidas guardadas en `data/_md_benchmark/<lib>/`.

### Rúbrica de evaluación
1. Estructura legal (CONSIDERANDO/RESUELVE/ARTÍCULO/PARÁGRAFO navegable)
2. Fidelidad de tablas
3. Encoding / mojibake
4. Capacidad y calidad de OCR (escaneados = 73 % del corpus)
5. Velocidad y peso de dependencias (importa para correr los 2417 PDFs en la PC nueva)
6. Post-procesamiento que aún hace falta

## Resultados

| Criterio | pymupdf4llm | pdf2markdown4llm | docling |
|---|---|---|---|
| Éxito en digitales | 33/33 | 33/33 | OK (tras arreglar deps) |
| Velocidad (avg/doc digital) | **6.8 s** | **4.3 s** | ver nota ▼ (lento) |
| Tablas *presentes* (de 33)¹ | **18/33** | **0/33** | sí (detecta y reconstruye) |
| Reflujo de párrafos | **Sí** (1 línea por considerando) | No (conserva saltos del PDF) | Sí |
| Encabezados de sección | `#`/`##` razonables | Sobre-fragmenta el título en varios `###` | `##` razonables |
| Mojibake (U+FFFD) | 0 | 0 | 0 |
| **OCR de escaneados** | ❌ no tiene | ❌ no tiene | ✅ **sí, buena calidad** |
| Tablas en escaneados (OCR) | — | — | ✅ reconstruye columnas + filas |
| Peso de dependencias | ligera (PyMuPDF) | ligera (pdfplumber) | **pesada** (torch+transformers+OCR) |
| Conflicto de versiones | ninguno | ninguno | **requiere transformers ≥4.49** → rompe `trl`/stack de entrenamiento; aislar en venv propio |

> ¹ "Tablas presentes" cuenta la *aparición* de tablas Markdown (`|---`), no su fidelidad celda
> a celda. La fidelidad solo se inspeccionó manualmente en docling (escaneado, reconstrucción
> excelente). El veredicto relativo (18 vs 0) es válido de todas formas.
>
> ▼ **Velocidad de docling:** la primera conversión midió 180 s pero incluye la **carga única**
> de modelos de layout/OCR; la segunda conversión (un escaneado, más pesado) tardó 33 s. El costo
> por documento en caliente es mucho menor que 180 s, pero aun así muy superior a las librerías
> ligeras y dependiente de GPU para correr a escala.

### Observaciones por librería

**pymupdf4llm** — Reflowa cada considerando a una sola línea (ideal para el `qa_builder`,
que es line-based), detecta tablas en más de la mitad de los documentos, encoding limpio,
muy rápida y sin dependencias nuevas. No promueve `ARTÍCULO N` a encabezado por sí sola
(emite el texto en negrita) y **repite el encabezado de página** en cada hoja.

**pdf2markdown4llm** — Velocidad similar pero **no detectó ninguna tabla** (a pesar de tener
parámetros para ello) y **conserva los saltos de línea duros** del PDF (parte frases a la mitad,
hyphenación "académico-\nadministrativas"), lo que ensucia el dataset. Sobre-fragmenta el título.
Sin ventaja frente a pymupdf4llm.

**docling** — La más potente: modelo de layout + reconocimiento de tablas + **OCR**. Sobre un
escaneado (Resolución Académica 010/2026) extrajo los considerandos en español coherente y
**reconstruyó la tabla** (apellidos, proyecto, horas, fechas, rol) — justo lo que el viejo
pipeline EasyOCR no lograba. Costos: ~30× más lenta en CPU, descarga modelos (~capa HF),
y exige `transformers` moderno que entra en conflicto con el stack actual del proyecto
(hay que aislarla en su propio entorno). El `<!-- image -->` y la repetición de encabezado de
página son ruido menor.

## Recomendación

**Estrategia híbrida según tipo de PDF:**

1. **Digitales (~27 % del corpus, alcance inmediato de Fase 8): usar `pymupdf4llm`.**
   Es la mejor relación calidad/velocidad/peso, detecta tablas y reflowa bien. Reemplaza a
   `pdfplumber` en `src/extraction/pdf_digital.py`.

2. **Escaneados (~73 %, diferido a la PC nueva con GPU): usar `docling`.**
   Es la única de las tres con OCR utilizable y buen manejo de tablas. Correrla en la
   workstation (RTX 4090) en un **entorno virtual aparte** para no chocar con `transformers`/`trl`
   del entorno de inferencia/entrenamiento. En CPU es inviable a escala.
   *Salvedad:* la conclusión se apoya en **n=1 escaneado** (muy convincente: tabla reconstruida,
   español coherente). Antes de comprometer el 73 % del corpus, validar docling en ~5–10
   escaneados variados en la PC nueva, ya que la calidad de OCR varía más que la extracción digital.

3. **Descartar `pdf2markdown4llm`**: no aporta nada sobre pymupdf4llm y maneja peor tablas y saltos de línea.

### ⚠️ Post-procesamiento: la capa regex actual NO sirve tal cual

`src/extraction/to_markdown.py` fue escrita para el texto plano de `pdfplumber`. Sus regex están
ancladas (`^CONSIDERANDO$`, etc.), pero pymupdf4llm ya emite las líneas **decoradas en Markdown**
(`## **CONSIDERANDO**`, `### **ACUERDA**`, `**Artículo 1º.**`). **Comprobado empíricamente:** al
pasar una salida de pymupdf4llm por `_promote_structure()` se obtienen **0 promociones**
(`##`=0, `###`=0, `####`=0). Es decir, conectar pymupdf4llm a la capa regex actual produce `.md`
plano que `qa_builder` no puede navegar por encabezados.

**Hechos sobre la estructura nativa de pymupdf4llm:**
- Emite *algo* de estructura por su cuenta (`## CONSIDERANDO`, a veces `### ACUERDA`), pero
  **inconsistente** (mete `ACUERDA` como `###` en vez de `##`).
- No normaliza `ARTÍCULO`/`PARÁGRAFO` a un nivel de encabezado fijo.
- **Repite el encabezado de página** en cada hoja (ruido a deduplicar).

## Estado de implementación (2026-07-02) — HECHO

El rewire de la extracción digital está implementado y validado:

- **`src/extraction/pdf_digital.py`**: extractor = pymupdf4llm (fallback pdfplumber).
  Nuevo detector `looks_corrupt()` de capa de texto corrupta (fuente rota, sistemática).
- **`src/extraction/to_markdown.py`**: normalizador nuevo `_normalize_markdown()` que
  - preserva tablas verbatim,
  - **descarta** los encabezados-ruido de pymupdf4llm, banners de página, pies (dirección/
    correo del campus), fechas sueltas y reglas,
  - **deduplica** el subtítulo repetido por hoja,
  - re-promueve CONSIDERANDO/RESUELVE/ACUERDA/CAPÍTULO→`##`, ARTÍCULO→`###`, PARÁGRAFO→`####`.
- **`src/extraction/pdf_scanned.py`**: OCR con `docling` como único motor (import diferido).
  Los motores viejos easyocr/tesseract fueron eliminados del código y de `requirements.txt` (2026-07-03).
- **`scripts/extract_text.py`**: los digitales corruptos y los escaneados se desvían a una
  **cola de OCR** (`data/_ocr_queue.txt`) para procesar en la workstation con docling.

**Validación (muestra de 33 digitales + año 2023 completo):**
- Promoción de estructura: de **0** (regex vieja) a valores reales (ej. RS 5/2023 → 14 artículos,
  CONSIDERANDO+RESUELVE, PARÁGRAFO).
- 0 banners/pies residuales en el cuerpo (fuera de tablas); tablas preservadas.
- Bug corregido en el camino: el regex de banner con `.search()` borraba cualquier línea que
  mencionara "Universidad de los Llanos" (incluía artículos). Ahora está anclado al inicio de línea.
- El único digital corrupto de la muestra (Acuerdo Superior 10/2020) se detecta y desvía a OCR.

### Implicación de dependencias
- pymupdf4llm en la laptop: sin riesgo. **Añadido** a `requirements.txt`; PyMuPDF subido a 1.28.0.
- docling: NO se instala en el venv del proyecto (rompió `transformers` 4.46.3 → 5.x, y `trl`).
  Va en entorno separado: **`requirements-docling.txt`**, para la etapa de OCR en la PC nueva.
