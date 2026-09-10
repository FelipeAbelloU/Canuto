# CANUTO

Chatbot Asistente Normativo de la Universidad de los Llanos.

CANUTO responde preguntas sobre la normativa de Unillanos (resoluciones, acuerdos y
demas documentos institucionales). El modelo aprende la normativa durante el
entrenamiento: se hace fine-tuning con QLoRA sobre Qwen2.5-7B-Instruct, sin usar
busqueda de documentos en tiempo de consulta.

Proyecto de grado — Ingenieria de Sistemas, Universidad de los Llanos.

## Como funciona

El pipeline va del PDF original hasta el modelo que responde en la pagina web:

```
PDF/  ->  data/extracted/*.md  ->  dataset_alpaca.json  ->  train/test  ->  adaptador LoRA
```

1. **Extraccion**: los PDFs digitales se convierten a Markdown con pymupdf4llm y los
   escaneados con OCR (docling). El resultado conserva la estructura legal del
   documento (CONSIDERANDO, RESUELVE, ARTICULO, PARAGRAFO).
2. **Dataset**: cada documento se recorre por articulos y se arman pares
   pregunta-respuesta en formato alpaca.
3. **Entrenamiento**: fine-tuning con QLoRA (el modelo base se carga en 4 bits y solo
   se entrena el adaptador).
4. **Evaluacion**: el adaptador se mide sobre el 30% apartado y sobre un conjunto de
   preguntas fuera de dominio, para comparar configuraciones de hiperparametros.
5. **Uso**: la interfaz web en Django carga el modelo base mas el adaptador entrenado.

## Instalacion

En la workstation (Ubuntu, Python 3.12, GPU NVIDIA):

```bash
bash install.sh           # entorno principal
bash install-docling.sh   # entorno aparte para el OCR
```

El OCR va en su propio entorno virtual porque sus dependencias chocan con las del
entrenamiento.

## Uso

```bash
source venv/bin/activate

python scripts/extract_text.py            # PDFs digitales a Markdown
python scripts/extract_text.py --ocr      # escaneados (usar el venv de docling)
python scripts/build_dataset.py           # pares pregunta-respuesta
python scripts/split_dataset.py           # division 70/30
python scripts/train_gpu.py               # entrenamiento QLoRA

python scripts/chat_cli.py                # probar el modelo por terminal
python ui/django_chatbot/run.py           # interfaz web
```

Despues de entrenar hay que apuntar `model.checkpoint_path` en `config/config.yaml`
a la carpeta del adaptador.

### Entrenamiento y evaluacion

`train_gpu.py` no recibe argumentos: los hiperparametros se editan en el bloque de
constantes del principio del archivo, que es la unica fuente de verdad de cada corrida.
El nombre de la carpeta de salida se arma solo con esos valores
(`data/checkpoints/t1_7b-e3-lr2e4-r16-wd0.0`), de modo que dos corridas nunca se pisan.

Las curvas de perdida van a Weights & Biases (hay que correr `wandb login` una vez en la
maquina; con `WANDB_MODE=offline` se guardan en local y se suben despues).

```bash
python scripts/evaluate.py --checkpoint data/checkpoints/t1_7b-e3-lr2e4-r16-wd0.0
```

La evaluacion se hace en dos fases: primero genera las respuestas del modelo y las guarda
en `data/eval/<nombre>.json`, y despues calcula las metricas leyendo ese archivo. Si el
archivo ya existe se reutiliza, asi que recalcular metricas cuesta segundos en vez de
volver a generar; `--regenerar` fuerza a generarlas de nuevo. Al terminar imprime los
hiperparametros y las metricas de la corrida en una linea separada por tabulaciones,
lista para pegar en la bitacora.

## Estructura

```
config/     configuracion del proyecto
src/        extraccion, dataset, inferencia y chat
scripts/    los pasos del pipeline
ui/         interfaz web en Django
data/       textos extraidos, dataset, checkpoints y evaluaciones (no se versionan)
PDF/        documentos fuente (no se versionan)
```

## Hardware

El entrenamiento y la inferencia corren en una workstation con RTX 5080 (16 GB).
Como la tarjeta es Blackwell hace falta PyTorch compilado para CUDA 13 (wheels cu130);
los scripts de instalacion ya lo instalan asi.

## Autor

Cristian Felipe Abello Gamba
Directora: Diana Marcela Cardona Roman, M.Sc, Ph.D
Co-director: Cesar Augusto Diaz, M.Sc
