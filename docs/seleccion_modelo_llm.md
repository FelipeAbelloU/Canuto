# Justificación Técnica para la Selección del Modelo de Lenguaje en el Asistente Normativo de Unillanos

**Proyecto:** Implementación de un asistente inteligente orientado a la gestión del conocimiento normativo basado en modelos de lenguaje generativo para la Universidad de los Llanos

**Autor:** Cristian Felipe Abello Gamba  
**Programa:** Ingeniería de Sistemas — Universidad de los Llanos  
**Fecha:** Mayo 2026

---

## Resumen

Este documento argumenta la selección del modelo `Qwen/Qwen2.5` (variante 3B para desarrollo local y 7B para producción) como modelo base para el fine-tuning del asistente de normativa universitaria. Se analiza y descarta el modelo GPT-2 (Radford et al., 2019) por sus limitaciones técnicas documentadas, se aclara que el modelo denominado "GPT-2.5" no existe en ninguna fuente verificable, y se justifica la elección de Qwen2.5 mediante criterios de soporte multilingüe, capacidad de representación, ventana de contexto y viabilidad de ajuste fino en el hardware disponible.

---

## 1. Contexto y Requerimientos del Sistema

El asistente normativo de la Universidad de los Llanos debe:

1. Procesar documentos normativos en español (resoluciones rectorales, acuerdos académicos, convocatorias) con extensiones típicas de 1.000 a 8.000 palabras por documento.
2. Responder preguntas en español de manera precisa, citando artículos y numerales específicos.
3. Ser entrenado mediante **fine-tuning supervisado** (SFT) con LoRA sobre un corpus cerrado de 20 documentos institucionales.
4. Poder ejecutarse en un computador portátil con CPU Intel i5-7200U y 16 GB de RAM (para desarrollo), y en una estación de trabajo con GPU NVIDIA RTX 4090i de 24 GB de VRAM (para producción).

Estos requerimientos determinan directamente la elección del modelo base.

---

## 2. Sobre el Modelo GPT-2 (OpenAI, 2019)

GPT-2 (*Generative Pre-trained Transformer 2*) fue introducido por OpenAI en febrero de 2019 (Radford et al., 2019). Es un modelo de lenguaje autorregresivo basado en la arquitectura Transformer (Vaswani et al., 2017) con las siguientes características técnicas:

| Característica | Valor |
|---|---|
| Parámetros (versión base) | 117 millones |
| Parámetros (versión XL) | 1.500 millones |
| Ventana de contexto | **1.024 tokens (~750 palabras)** |
| Idioma principal de entrenamiento | **Inglés** |
| Año de publicación | **2019** |
| Datos de entrenamiento | ~40 GB de texto web en inglés (WebText) |

### 2.1 Limitaciones críticas para este proyecto

**Limitación 1 — Ventana de contexto insuficiente:**  
GPT-2 tiene una ventana de contexto de 1.024 tokens, equivalente a aproximadamente 750 palabras. Una resolución rectoral promedio de Unillanos contiene entre 1.500 y 5.000 palabras. Esto significa que durante el fine-tuning, el modelo nunca podría procesar un documento completo en una sola pasada; procesaría fragmentos desconectados y perdería relaciones semánticas entre artículos. Los modelos modernos como Qwen2.5 tienen ventanas de 32.768 a 131.072 tokens, suficientes para un documento completo.

**Limitación 2 — Entrenamiento predominantemente en inglés:**  
GPT-2 fue entrenado en el corpus WebText, compuesto casi exclusivamente de texto en inglés recolectado desde Reddit. No existen datos de entrenamiento verificados para español. Estudios recientes muestran que modelos pequeños con entrenamiento no multilingüe degradan su rendimiento en español hasta en un 37% respecto a su rendimiento en inglés (Sáez et al., 2024). Para un sistema cuya totalidad de documentos y consultas son en español colombiano institucional, esta es una limitación de fondo.

**Limitación 3 — Capacidad de representación reducida:**  
Con 117 millones de parámetros (versión base), GPT-2 tiene capacidad limitada para memorizar relaciones normativas complejas durante el fine-tuning. Por comparación, `Qwen2.5-3B` tiene 3 mil millones de parámetros (25 veces más), lo que permite codificar estructuras semánticas mucho más ricas con el mismo número de ejemplos de entrenamiento.

**Limitación 4 — Obsolescencia técnica:**  
GPT-2 fue publicado en 2019. En los últimos seis años, la investigación en modelos de lenguaje ha avanzado radicalmente. Usar GPT-2 en un proyecto de 2026 sería metodológicamente equivalente a implementar una solución de reconocimiento de imágenes con AlexNet (2012) ignorando ResNet, VGG o Vision Transformers. Aunque técnicamente posible, no refleja el estado del arte y debilitaría la contribución académica del trabajo.

---

## 3. Sobre "GPT-2.5": Este Modelo No Existe

Tras una búsqueda exhaustiva en las fuentes primarias de referencia en el campo —arXiv, Hugging Face Model Hub, OpenAI Blog, Semantic Scholar y Google Scholar— se confirma que **no existe ningún modelo de lenguaje denominado "GPT-2.5"**.

La nomenclatura de OpenAI sigue la secuencia: GPT-1 (2018) → GPT-2 (2019) → GPT-3 (2020, Brown et al.) → GPT-3.5 (2022) → GPT-4 (2023). No existe ninguna versión intermedia con ese nombre.

Los modelos que podrían confundirse con ese nombre son:

| Nombre posible | Modelo real | Creador | Notas |
|---|---|---|---|
| "GPT-2.5" | **No existe** | — | No hay ningún modelo con ese identificador |
| GPT-3.5 | `gpt-3.5-turbo` | OpenAI | API cerrada, sin fine-tuning local gratuito |
| GPT-J (6B) | `EleutherAI/gpt-j-6b` | EleutherAI | Abierto, pero inferior a Qwen2.5 en español |
| GPT-Neo | `EleutherAI/gpt-neo-*` | EleutherAI | Hasta 2.7B, entrenado en inglés |

Se recomienda consultar con la directora de proyecto qué fuente específica menciona "GPT-2.5", ya que posiblemente se trate de una referencia a GPT-2 (versión XL de 1.5B), o a modelos de la familia GPT-3 accesibles por API.

---

## 4. Modelos Modernos Relevantes para el Proyecto (2023–2025)

La siguiente tabla compara los modelos abiertos más relevantes para el contexto de este proyecto:

| Modelo | Creador | Año | Params | Contexto | Español | Licencia | CPU 16GB |
|---|---|---|---|---|---|---|---|
| GPT-2 XL | OpenAI | 2019 | 1.5B | 1K tokens | ⚠️ Bajo | MIT | ✅ |
| Mistral-7B-Instruct | Mistral AI | 2023 | 7B | 32K tokens | ✅ Bueno | Apache 2.0 | ❌ |
| Llama 3.2-3B-Instruct | Meta | 2024 | 3B | 128K tokens | ✅ Bueno | Llama† | ✅ |
| **Qwen2.5-3B-Instruct** | Alibaba Cloud | 2024 | 3B | 32K tokens | ✅ Excelente | Apache 2.0 | ✅ |
| **Qwen2.5-7B-Instruct** | Alibaba Cloud | 2024 | 7.6B | 128K tokens | ✅ Excelente | Apache 2.0 | ❌ |

†La licencia Meta Llama permite uso académico y comercial con restricciones menores; no es Apache 2.0 puro.

---

## 5. Justificación de Qwen2.5 como Modelo Seleccionado

### 5.1 Descripción técnica

`Qwen2.5` es una serie de modelos de lenguaje grandes desarrollada por el equipo Qwen de Alibaba Cloud, presentada en diciembre de 2024 (Qwen Team, 2024). Está basada en la arquitectura Transformer decoder con las siguientes características relevantes:

- Preentrenamiento sobre **18 billones de tokens** en 29+ idiomas, con el español como idioma explícito de primera clase.
- Post-entrenamiento mediante SFT (Supervised Fine-Tuning), DPO (Direct Preference Optimization) y GRPO sobre 1 millón de ejemplos de alta calidad.
- Vocabulario de **152.064 tokens**, que incluye suficiente cobertura morfológica del español para tokenizar correctamente palabras compuestas, tildes y términos jurídicos sin segmentación deficiente.
- Variantes disponibles desde 0.5B hasta 72B parámetros, todas con la misma arquitectura base.

### 5.2 Ventajas concretas sobre GPT-2 para este proyecto

| Criterio | GPT-2 XL | Qwen2.5-3B | Ventaja |
|---|---|---|---|
| Ventana de contexto | 1.024 tokens | 32.768 tokens | **×32 mayor** — cubre documentos completos |
| Parámetros | 1.500 M | 3.000 M | **×2 más capacidad** con mismo hardware |
| Español nativo | No | Sí (29 idiomas) | Tokenización y comprensión directas |
| Año de entrenamiento | 2019 | 2024 | 5 años de avance en la disciplina |
| Fine-tuning con LoRA | Sí | Sí | Equivalente |
| RAM en CPU (inferencia) | ~3 GB | ~6 GB | Ambos caben en 16 GB |

### 5.3 Viabilidad en el hardware disponible

El método QLoRA (Dettmers et al., 2023) permite realizar fine-tuning de modelos cuantizados en 4 bits, reduciendo los requerimientos de VRAM aproximadamente un 75%. Para `Qwen2.5-7B`:

- **Sin cuantizar (BF16):** ~15 GB de VRAM — cabe en RTX 4090i (24 GB).
- **Con QLoRA 4-bit:** ~4 GB de VRAM para fine-tuning — cabe con margen en RTX 4090i.

Para el desarrollo local en laptop:  
- `Qwen2.5-3B` requiere ~6 GB de RAM para inferencia en CPU, dentro del límite de 16 GB disponibles.
- La velocidad estimada es de 2–8 tokens por segundo en Intel i5-7200U, suficiente para pruebas funcionales.

### 5.4 Soporte para fine-tuning de dominio específico

La literatura reciente confirma que el fine-tuning supervisado con LoRA (Hu et al., 2022) sobre modelos de la escala 3B–7B produce mejoras sustanciales en tareas de comprensión de documentos legales y normativos en idiomas distintos al inglés. El trabajo de Al-Khatib et al. (2024) demuestra que modelos de la familia LLaMA (arquitectura comparable a Qwen2.5) fine-tuneados con LoRA sobre corpus legales en árabe logran comprensión precisa de textos normativos complejos, con un enfoque directamente extrapolable al español institucional colombiano.

### 5.5 Licencia

`Qwen2.5` está publicado bajo licencia **Apache 2.0**, que permite:
- Uso académico y de investigación sin restricciones.
- Modificación y redistribución del modelo y sus derivados.
- Uso comercial del modelo ajustado.

Esto es relevante para un proyecto de tesis que puede derivar en publicaciones, ponencias o transferencia tecnológica a la Universidad de los Llanos.

---

## 6. Estrategia de Implementación en Dos Fases

Dado que el acceso a la estación de trabajo universitaria no es permanente, se propone la siguiente estrategia:

**Fase de desarrollo (laptop):**  
Usar `Qwen/Qwen2.5-3B-Instruct` para validar el pipeline completo: extracción de texto, generación del dataset, fine-tuning ligero y pruebas de inferencia. La diferencia arquitectónica con el modelo 7B es mínima; ambos comparten tokenizador, plantilla de chat y formato de checkpoint.

**Fase de producción (workstation con RTX 4090i):**  
Usar `Qwen/Qwen2.5-7B-Instruct` con QLoRA 4-bit para el fine-tuning final sobre el corpus completo. El código del sistema no requiere ningún cambio; solo se actualiza el parámetro `checkpoint_path` en el archivo de configuración.

Esta estrategia desacopla el desarrollo del hardware de alta capacidad, permite iteración rápida y produce un modelo final de mayor calidad para la evaluación académica.

---

## 7. Conclusión

La selección de `Qwen/Qwen2.5` como modelo base para el asistente normativo de Unillanos está sustentada en cuatro criterios objetivos:

1. **Soporte de español verificado** como idioma de primera clase en el preentrenamiento (29+ idiomas, 18 billones de tokens).
2. **Ventana de contexto 32× mayor** que GPT-2, suficiente para procesar resoluciones completas sin truncamiento.
3. **Compatibilidad con técnicas de ajuste fino eficiente** (LoRA, QLoRA) que permiten el entrenamiento dentro de los recursos de hardware disponibles.
4. **Licencia Apache 2.0** sin restricciones para proyectos académicos ni para publicaciones derivadas.

GPT-2 (2019), si bien funcional, presenta limitaciones estructurales que lo hacen inadecuado para el procesamiento de documentos normativos en español en 2026. El modelo denominado "GPT-2.5" no existe en ningún repositorio ni publicación verificable.

---

## Referencias

Dettmers, T., Pagnoni, A., Holtzman, A., & Zettlemoyer, L. (2023). QLoRA: Efficient finetuning of quantized LLMs. *arXiv preprint arXiv:2305.14314*. https://arxiv.org/abs/2305.14314

Hu, E. J., Shen, Y., Wallis, P., Allen-Zhu, Z., Li, Y., Wang, S., Wang, L., & Chen, W. (2022). LoRA: Low-rank adaptation of large language models. *Proceedings of the International Conference on Learning Representations (ICLR 2022)*. https://arxiv.org/abs/2106.09685

Al-Khatib, A., et al. (2024). ALKAFI-LLAMA3: Fine-tuning LLMs for precise legal understanding in Palestine. *arXiv preprint arXiv:2412.14771*. https://arxiv.org/pdf/2412.14771

Qwen Team. (2024). Qwen2.5 technical report. *arXiv preprint arXiv:2412.15115*. https://arxiv.org/abs/2412.15115

Radford, A., Wu, J., Child, R., Luan, D., Amodei, D., & Sutskever, I. (2019). Language models are unsupervised multitask learners. *OpenAI Blog*, 1(8), 9. https://cdn.openai.com/better-language-models/language_models_are_unsupervised_multitask_learners.pdf

Sáez, J. B., et al. (2024). Bilingual evaluation of language models on general knowledge in university entrance exams with minimal contamination. *arXiv preprint arXiv:2409.12746*. https://arxiv.org/pdf/2409.12746

Vaswani, A., Shazeer, N., Parmar, N., Uszkoreit, J., Jones, L., Gomez, A. N., Kaiser, Ł., & Polosukhin, I. (2017). Attention is all you need. *Advances in Neural Information Processing Systems (NeurIPS 2017)*, 30. https://arxiv.org/abs/1706.03762

Zhang, T., et al. (2025). BenchMAX: A comprehensive multilingual evaluation suite for large language models. *arXiv preprint arXiv:2502.07346*. https://arxiv.org/html/2502.07346v1

---

*Documento generado como soporte técnico para la selección del modelo base del Proyecto5 — Asistente Normativo Unillanos.*
