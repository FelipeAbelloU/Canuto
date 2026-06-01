"""
colab_train.py — Fine-tuning de Qwen2.5-1.5B-Instruct en Google Colab (T4 GPU)

INSTRUCCIONES:
  1. Ve a https://colab.research.google.com
  2. Nuevo notebook → Entorno de ejecución → Cambiar tipo → T4 GPU
  3. Crea una celda por cada sección marcada con  # ── CELDA N ──
  4. Ejecuta las celdas en orden

Tiempo total estimado: ~20-30 minutos
"""

# ══════════════════════════════════════════════════════════════════
# ── CELDA 1 ── Instalar dependencias
# ══════════════════════════════════════════════════════════════════
# Pega esto en la primera celda y ejecútala

# !pip install -q peft>=0.13 "trl>=0.12,<0.13" datasets


# ══════════════════════════════════════════════════════════════════
# ── CELDA 2 ── Subir el dataset desde tu laptop
# ══════════════════════════════════════════════════════════════════
# Pega esto en la segunda celda y ejecútala
# Aparecerá un botón "Choose Files" → sube data/dataset/dataset_alpaca.json

# from google.colab import files
# uploaded = files.upload()
# print("Archivo recibido:", list(uploaded.keys()))


# ══════════════════════════════════════════════════════════════════
# ── CELDA 3 ── Entrenamiento completo
# ══════════════════════════════════════════════════════════════════
# Pega TODO el bloque de abajo en la tercera celda y ejecútala

COLAB_TRAIN_CODE = '''
import json
from pathlib import Path

import torch
from datasets import Dataset
from peft import LoraConfig
from transformers import AutoModelForCausalLM, AutoTokenizer
from trl import SFTConfig, SFTTrainer

# ── Configuración ──────────────────────────────────────────────
MODEL_NAME   = "Qwen/Qwen2.5-1.5B-Instruct"
DATASET_PATH = "dataset_alpaca.json"   # archivo subido en celda 2
OUTPUT_DIR   = "/content/unillanos-v1" # donde se guarda el checkpoint

EPOCHS   = 3
RANK     = 8
SEQ_LEN  = 750

# System prompt exacto del config.yaml de Proyecto5
SYSTEM_PROMPT = (
    "Eres un asistente especializado en normativa universitaria de la "
    "Universidad de los Llanos (Unillanos), Colombia.\\n"
    "Responde preguntas sobre reglamentos, resoluciones, acuerdos y documentos "
    "normativos emitidos desde 2015.\\n"
    "Responde siempre en español, de manera clara y útil para estudiantes, "
    "docentes y personal administrativo.\\n"
    "Si no tienes certeza sobre una información, indícalo claramente.\\n"
    "No inventes información que no conozcas con certeza."
)

LORA_TARGET_MODULES = [
    "q_proj", "k_proj", "v_proj", "o_proj",
    "gate_proj", "up_proj", "down_proj",
]

# ── Dataset ───────────────────────────────────────────────────
print("[1/4] Cargando tokenizador...")
tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)
if tokenizer.pad_token is None:
    tokenizer.pad_token = tokenizer.eos_token

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

# ── Modelo ────────────────────────────────────────────────────
print("[3/4] Cargando modelo base (descarga ~3 GB)...")
model = AutoModelForCausalLM.from_pretrained(
    MODEL_NAME,
    torch_dtype=torch.bfloat16,  # BF16 en GPU: mitad de memoria, misma calidad
    device_map="auto",
    trust_remote_code=True,
)
total = sum(p.numel() for p in model.parameters())
print(f"    {total/1e6:.0f}M parámetros  |  dtype: BF16  |  device: {model.device}")

# ── LoRA + Entrenamiento ──────────────────────────────────────
print("[4/4] Iniciando fine-tuning con LoRA...")

lora_config = LoraConfig(
    r=RANK,
    lora_alpha=RANK * 2,
    target_modules=LORA_TARGET_MODULES,
    lora_dropout=0.05,
    bias="none",
    task_type="CAUSAL_LM",
)

adapters_dir = Path(OUTPUT_DIR) / "_adapters"

training_args = SFTConfig(
    output_dir=str(Path(OUTPUT_DIR) / "_checkpoints"),
    num_train_epochs=EPOCHS,
    per_device_train_batch_size=4,      # T4 puede con batch 4 en BF16
    gradient_accumulation_steps=2,      # batch efectivo = 8
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
    bf16=True,                          # BF16 training en GPU
    report_to="none",
)

trainer = SFTTrainer(
    model=model,
    train_dataset=dataset,
    peft_config=lora_config,
    args=training_args,
)

trainer.train()

# ── Guardar adaptadores LoRA (backup ligero) ──────────────────
print("\\nGuardando adaptadores LoRA como backup...")
adapters_dir.mkdir(parents=True, exist_ok=True)
trainer.model.save_pretrained(str(adapters_dir))
tokenizer.save_pretrained(str(adapters_dir))
print(f"  → {adapters_dir}  (backup ligero, por si falla el merge)")

# ── Fusionar y guardar checkpoint final ───────────────────────
print("\\nFusionando LoRA con modelo base...")
merged = trainer.model.merge_and_unload()

Path(OUTPUT_DIR).mkdir(parents=True, exist_ok=True)
merged.save_pretrained(OUTPUT_DIR, safe_serialization=True)
tokenizer.save_pretrained(OUTPUT_DIR)

print(f"\\n✓ Checkpoint guardado en: {OUTPUT_DIR}")
print(f"  Tamaño aprox: {sum(f.stat().st_size for f in Path(OUTPUT_DIR).rglob("*") if f.is_file()) / 1e9:.1f} GB")
'''

# Para ejecutar en Colab, pega COLAB_TRAIN_CODE en la celda 3 sin las comillas triples


# ══════════════════════════════════════════════════════════════════
# ── CELDA 4 ── Guardar en Google Drive y descargar
# ══════════════════════════════════════════════════════════════════
# Opción A: Guardar en Google Drive (recomendado — no se pierde si Colab cierra)
#
# from google.colab import drive
# import shutil
# drive.mount('/content/drive')
# dest = '/content/drive/MyDrive/Tesis/unillanos-v1'
# shutil.copytree('/content/unillanos-v1', dest, dirs_exist_ok=True)
# print(f"✓ Guardado en Google Drive: {dest}")
# print("  Descarga desde drive.google.com → carpeta Tesis/unillanos-v1")
#
# ──────────────────────────────────────────────────────────────────
# Opción B: Comprimir y descargar directamente desde Colab
#
# import shutil
# from google.colab import files
# shutil.make_archive('/content/unillanos-v1', 'zip', '/content/unillanos-v1')
# files.download('/content/unillanos-v1.zip')
# print("Descargando unillanos-v1.zip (~3 GB)...")


# ══════════════════════════════════════════════════════════════════
# ── DESPUÉS DE DESCARGAR ── Configurar en tu laptop
# ══════════════════════════════════════════════════════════════════
#
# 1. Descomprime el zip en:
#    C:\Users\Equipo\OneDrive\Documentos\Tesis\Proyecto5\data\checkpoints\unillanos-v1\
#
# 2. Edita config/config.yaml:
#    model:
#      checkpoint_path: "data/checkpoints/unillanos-v1"
#
# 3. Prueba el chat:
#    python scripts\chat_cli.py
#
# 4. Prueba la UI web:
#    python ui\app.py
