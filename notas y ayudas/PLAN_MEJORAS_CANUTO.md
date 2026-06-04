# Plan de Mejoras CANUTO — Rutas de Trabajo
**Elaborado:** 2026-06-03  
**Estado del modelo:** v2 (Qwen2.5-1.5B, 387 pares, LoRA rank=8)  
**Dataset actualizado:** v3-ready (413 pares verificados)

---

## Qué se hizo en esta sesión (sin reentrenar)

| Cambio | Efecto esperado |
|--------|----------------|
| System prompt reforzado (6 reglas explícitas) | Reduce confirmación de info falsa, fuerza admitir ignorancia |
| Temperature 0.7 → 0.3 | Respuestas más deterministas, menos "creativas" / inventadas |
| 26 pares QA curados añadidos al dataset | Base para v3: fraccionamiento, incapacidades, segunda lengua, movilidad, programas |
| Dataset: 387 → 413 pares | Listo para subirlo a Colab y reentrenar |

---

## Diagnóstico raíz (investigación 2026-06-03)

La literatura (2025) es clara sobre la limitación fundamental:

> **LoRA fine-tuning es mejor para adaptación de comportamiento (tono, formato, estilo), no para inyectar grandes volúmenes de conocimiento factual.**  
> — DigitalOcean, Amir Teymoori, FinLoRA (2025)

Esto explica por qué el modelo v1/v2 aprendió el formato de las respuestas (citar resoluciones, hacer listas) pero no internalizó los hechos concretos. Para conocimiento factual verificable, la literatura recomienda RAG.

---

## Rutas de mejora — 4 niveles

### Ruta A — Inmediata (sin reentrenar) ✅ YA APLICADO
**Costo:** 0 | **Tiempo:** hecho

- System prompt con 6 reglas explícitas (prohibición de inventar, prohibición de confirmar sin certeza)
- Temperatura 0.3 (más conservador)
- **Limitación:** No añade conocimiento nuevo, solo restringe comportamiento malo

---

### Ruta B — Reentrenar v3 en Colab (próximo paso)
**Costo:** ~20 min en Colab T4 | **Tiempo:** 1 sesión

**Qué hacer:**
1. Subir `data/dataset/dataset_alpaca.json` (413 pares) a Google Drive
2. Ejecutar `scripts/colab_train.py` (mismo proceso que v1 y v2)
3. Descargar checkpoint → `data/checkpoints/unillanos-v3/`
4. Actualizar `config/config.yaml`: `checkpoint_path: "data/checkpoints/unillanos-v3"`

**Mejoras esperadas respecto a v2:**
- Mejor conocimiento de fraccionamiento de matrícula (Res. 074/2026)
- Correcto procedimiento de incapacidades (Res. 068/2025)
- Correcta lista de programas de la facultad
- Corrección de preguntas trampa (segunda lengua, costos)

**Limitación:** El modelo aún puede alucinar en temas no cubiertos en el dataset

---

### Ruta C — Hybrid RAG (recomendación clave de la investigación)
**Costo:** 2-3 días de desarrollo | **Tiempo:** mes 1-2

**Por qué:** La investigación más reciente (2025) demuestra que para chatbots universitarios,  
**Fine-tuning (estilo + comportamiento) + RAG (conocimiento factual)** supera a cualquiera de los dos solos.  
Ver: [URAG — Hybrid RAG for University Chatbots (HCMUT, 2025)](https://arxiv.org/abs/2501.16276)

**Cómo implementarlo en CANUTO:**

```
data/extracted/*.txt (35 archivos ya extraídos)
      ↓
ChromaDB o FAISS (indexar chunks de texto)
      ↓
Al recibir pregunta → buscar chunks relevantes
      ↓
Pasar contexto + pregunta al modelo fine-tuneado
      ↓
El modelo responde CON el documento frente a él
```

**Stack técnico sugerido:**
- `sentence-transformers` con modelo `intfloat/multilingual-e5-base` (soporta español)
- `chromadb` para la base vectorial (simple, sin servidor externo)
- Chunk size: ~400 tokens, overlap 50 tokens
- Top-K retrieval: 3-5 chunks más relevantes

**Ventaja clave para la tesis:** Cuando SIRIUS proporcione nuevos documentos, solo hay que re-indexar, NO reentrenar el modelo.

**Código de referencia disponible en:** `src/api/app.py` ya tiene la estructura para modificarse

---

### Ruta D — Modelo 7B en workstation (producción)
**Costo:** 1 sesión de entrenamiento | **Tiempo:** mes 2-3

Cuando el workstation con RTX 4090i (24GB VRAM) esté disponible:

1. Cambiar modelo base: `Qwen/Qwen2.5-7B-Instruct`
2. Entrenar con QLoRA 4-bit (cabe en 24GB VRAM)
3. Mismo dataset y proceso, misma interfaz Django

**Por qué mejora:** 7B tiene ~5x más parámetros que 1.5B. Mejor comprensión del español, mejor razonamiento, menor tasa de alucinación base.

**Impacto esperado:** Reducción de alucinaciones estimada 40-60% respecto a 1.5B, incluso sin RAG.

---

### Ruta E — DPO Alignment (avanzado, para tesis)
**Costo:** 1-2 semanas de datos + entrenamiento | **Tiempo:** mes 3-4

**DPO (Direct Preference Optimization):** técnica de alineación post fine-tuning que enseña al modelo a preferir respuestas honestas sobre alucinaciones.

**Cómo crear el dataset de preferencias:**
```json
{
  "prompt": "¿Cuánto cuesta la matrícula?",
  "chosen": "No tengo información sobre costos de matrícula. Consulta con Admisiones.",
  "rejected": "La matrícula cuesta $2.300.000 pesos."
}
```

Tomar las respuestas incorrectas de los tests como `rejected` y las respuestas correctas como `chosen`.

**Referencia:** [Winning with Small Models: Knowledge Distillation vs Self-Training (2025)](https://arxiv.org/abs/2502.19545)

---

## Prioridad recomendada

```
[HOY]          Ruta A (prompt + temp) — YA HECHO
[ESTA SEMANA]  Ruta B — Reentrenar v3 en Colab (20 min)
[MES 1]        Ruta C — Implementar RAG básico sobre textos ya extraídos
[MES 2-3]      Ruta D — Migrar a 7B en workstation
[MES 3-4]      Fase 6 (evaluación formal) con modelo mejorado
[OPCIONAL]     Ruta E — DPO si se quiere publicar artículo
```

---

## Para la evaluación formal (Fase 6 — tesis)

**Métricas planificadas (ya en la propuesta):**
- BLEU-4 sobre respuestas generadas vs respuestas de referencia
- Similitud coseno (sentence-transformers) para relevancia semántica
- Precisión de cita documental (% de respuestas que citan el documento correcto)
- Pruebas cualitativas con usuarios de Unillanos

**Test set sugerido:** ~30 preguntas con respuestas verificadas en los PDFs originales, cubriendo todos los patrones de falla identificados en esta sesión.

---

## Recursos de referencia clave

- [7 Practical Techniques to Reduce LLM Hallucinations — Analytics Vidhya 2025](https://www.analyticsvidhya.com/blog/2025/09/reducing-hallucinations-in-llms/)
- [RAG vs Fine-Tuning — IBM 2025](https://www.ibm.com/think/topics/rag-vs-fine-tuning)
- [URAG: Hybrid RAG for University Admission Chatbots — arXiv 2025](https://arxiv.org/abs/2501.16276)
- [LoRA Fine-Tuning Guide 2025 — Amir Teymoori](https://amirteymoori.com/fine-tuning-llms-with-lora-a-practical-guide-for-2025/)
- [HybridRAG: Pre-Generated QA over PDFs — arXiv 2026](https://arxiv.org/abs/2602.11156)
