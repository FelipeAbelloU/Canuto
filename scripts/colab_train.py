"""
colab_train.py — Guia de fine-tuning en Google Colab (GPU T4, gratis)

INSTRUCCIONES:
  1. Ve a https://colab.research.google.com
  2. Nuevo notebook → Entorno de ejecucion → Cambiar tipo → T4 GPU → Guardar
  3. Crea una celda por cada seccion marcada con  # == CELDA N ==
  4. Ejecuta las celdas en orden

Tiempo total estimado: ~20-30 minutos
"""

# == CELDA 1 == Instalar dependencias
# Pega esto en la primera celda y ejecutala (~1 min, sin output visible)
#
# !pip install -q "peft>=0.13.0,<0.14.0" "trl>=0.12,<0.13" datasets
#
# IMPORTANTE: peft debe ser < 0.14 para evitar conflicto con torchao de Colab


# == CELDA 2 == Subir el dataset desde tu laptop
# Aparecera un boton "Choose Files" → sube data/dataset/dataset_alpaca.json
#
# from google.colab import files
# uploaded = files.upload()
# print("Archivo recibido:", list(uploaded.keys()))


# == CELDA 3 == Entrenamiento completo + merge
# Pega TODO el bloque de abajo en la tercera celda y ejecutala (~20-30 min)

import json
from pathlib import Path
import torch
from datasets import Dataset
from peft import LoraConfig
from transformers import AutoModelForCausalLM, AutoTokenizer
from trl import SFTConfig, SFTTrainer

MODEL_NAME   = "Qwen/Qwen2.5-1.5B-Instruct"
DATASET_PATH = "dataset_alpaca.json"
OUTPUT_DIR   = "/content/unillanos-v2"

EPOCHS   = 3
RANK     = 8
SEQ_LEN  = 750

# Prompt del sistema — debe coincidir con config/config.yaml
SYSTEM_PROMPT = (
    "Eres un asistente de normativa universitaria de la "
    "Universidad de los Llanos (Unillanos), Colombia.\n"
    "Tu conocimiento proviene exclusivamente de resoluciones, acuerdos y documentos normativos de Unillanos.\n\n"
    "REGLAS OBLIGATORIAS:\n"
    "1. Responde siempre en espanol, con lenguaje claro y amigable.\n"
    "2. Si conoces con certeza el nombre y numero de un documento, citalo. "
    "Si no estas seguro del numero exacto, NO lo menciones.\n"
    "3. Si sobre un mismo tema existen documentos de diferentes anos, da preferencia al mas reciente.\n"
    "4. Si la pregunta esta fuera de tu conocimiento, responde: "
    "'No tengo informacion sobre eso en los documentos que conozco. "
    "Te recomiendo consultar directamente con la oficina correspondiente de Unillanos.'\n"
    "5. Nunca inventes cifras, fechas, valores de matricula ni numeros de resoluciones.\n"
    "6. Si la respuesta tiene varios puntos, usa una lista numerada.\n"
    "7. Se conciso: responde lo que se pregunta sin anadir informacion innecesaria."
)

LORA_TARGETS = [
    "q_proj", "k_proj", "v_proj", "o_proj",
    "gate_proj", "up_proj", "down_proj",
]

# Tokenizador
print("[1/4] Cargando tokenizador...")
tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)
if tokenizer.pad_token is None:
    tokenizer.pad_token = tokenizer.eos_token

# Dataset: convierte pares Alpaca al formato chat de Qwen2.5
print("[2/4] Preparando dataset...")
with open(DATASET_PATH, encoding="utf-8") as f:
    records = json.load(f)

texts = []
for r in records:
    messages = [
        {"role": "system",    "content": SYSTEM_PROMPT},
        {"role": "user",      "content": r["instruction"]},
        {"role": "assistant", "content": r["output"]},
    ]
    texts.append({
        "text": tokenizer.apply_chat_template(
            messages, tokenize=False, add_generation_prompt=False
        )
    })

dataset = Dataset.from_list(texts)
print(f"    {len(dataset)} ejemplos listos")

# Modelo base — descarga ~3 GB desde HuggingFace
print("[3/4] Cargando modelo base...")
model = AutoModelForCausalLM.from_pretrained(
    MODEL_NAME,
    torch_dtype=torch.bfloat16,   # BF16 en GPU: mitad de memoria, misma calidad
    device_map="auto",
    trust_remote_code=True,
)
total = sum(p.numel() for p in model.parameters())
print(f"    {total/1e6:.0f}M parametros | BF16 | {model.device}")

# Entrenamiento con LoRA — entrena solo una fraccion pequena de los parametros
print("[4/4] Iniciando fine-tuning con LoRA...")

lora_config = LoraConfig(
    r=RANK,
    lora_alpha=RANK * 2,
    target_modules=LORA_TARGETS,
    lora_dropout=0.05,
    bias="none",
    task_type="CAUSAL_LM",
)

training_args = SFTConfig(
    output_dir=str(Path(OUTPUT_DIR) / "_checkpoints"),
    num_train_epochs=EPOCHS,
    per_device_train_batch_size=4,
    gradient_accumulation_steps=2,
    learning_rate=2e-4,
    warmup_ratio=0.1,
    lr_scheduler_type="cosine",
    logging_steps=10,
    save_strategy="epoch",
    save_total_limit=1,
    gradient_checkpointing=True,
    gradient_checkpointing_kwargs={"use_reentrant": False},
    max_seq_length=SEQ_LEN,
    dataset_text_field="text",
    fp16=False,
    bf16=True,
    report_to="none",
)

trainer = SFTTrainer(
    model=model,
    train_dataset=dataset,
    peft_config=lora_config,
    args=training_args,
)

trainer.train()

# Fusionar adaptadores LoRA con el modelo base → checkpoint estandar HuggingFace
print("\nFusionando LoRA con modelo base...")
merged = trainer.model.merge_and_unload()
Path(OUTPUT_DIR).mkdir(parents=True, exist_ok=True)

# Dividir en shards de 900 MB para descarga confiable desde el navegador
merged.save_pretrained(OUTPUT_DIR, safe_serialization=True, max_shard_size="900MB")
tokenizer.save_pretrained(OUTPUT_DIR)

archivos = [f for f in Path(OUTPUT_DIR).iterdir() if f.is_file() and not f.name.startswith("_")]
total_gb = sum(f.stat().st_size for f in archivos) / 1e9
print(f"\nCheckpoint guardado: {len(archivos)} archivos, {total_gb:.1f} GB")
for f in sorted(archivos):
    print(f"  {f.name}: {f.stat().st_size/1e6:.0f} MB")


# == CELDA 4 == Descargar checkpoint al laptop (sin Google Drive)
#
# from google.colab import files
# from pathlib import Path
#
# OUTPUT_DIR = "/content/unillanos-v2"
# archivos = sorted([
#     f for f in Path(OUTPUT_DIR).iterdir()
#     if f.is_file() and not f.name.startswith("_")
# ])
# print(f"{len(archivos)} archivos para descargar.")
# for f in archivos:
#     print(f"Descargando: {f.name}  ({f.stat().st_size/1e6:.0f} MB)")
#     files.download(str(f))
#     input("Presiona Enter cuando haya terminado de descargar...")


# == DESPUES DE DESCARGAR ==
# 1. Coloca los archivos en:
#    CANUTO\data\checkpoints\unillanos-v2\
#
# 2. Edita config/config.yaml:
#    model:
#      checkpoint_path: "data/checkpoints/unillanos-v2"
#
# 3. Prueba el chat:
#    python scripts\chat_cli.py
