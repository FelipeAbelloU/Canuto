"""Entrena el modelo con QLoRA sobre la normativa de Unillanos.

Corre en la workstation (RTX 5080, 16 GB). El modelo base se carga en 4-bit
para que quepa en la GPU y el resultado es un adaptador LoRA.

Este script NO recibe flags. Los hiperparametros se editan a mano en el bloque
de constantes de abajo: es la unica fuente de verdad de cada corrida. El nombre
de la carpeta de salida se arma solo con esos valores, asi que es imposible
sobrescribir un checkpoint por error.

Antes de correr:
    export HF_HOME=/bodega/hf-cache   # cache de modelos en el disco grande
    wandb login                       # una sola vez por maquina
    export WANDB_MODE=offline         # solo si la red del laboratorio falla

Uso:
    python scripts/train_gpu.py
"""
import json
import os
import random
import sys
from pathlib import Path

RAIZ = Path(__file__).resolve().parent.parent

# ===========================================================================
# HIPERPARAMETROS DE LA CORRIDA  --  editar a mano antes de entrenar
# ===========================================================================
NUM_PRUEBA = 1            # numero de prueba: prefijo de la carpeta (t1, t2, ...)
EPOCAS = 3                # 2, 3, 4
LEARNING_RATE = 2e-4      # 1e-4, 2e-4, 3e-4
LORA_RANK = 16            # 8, 16, 32  (alpha = rank * 2, se deriva solo)
WEIGHT_DECAY = 0.0        # 0.0, 0.01, 0.1

# Regla del barrido: mover UNA sola perilla por corrida. Si se mueven dos, la
# corrida no ensena nada porque no se sabe cual de las dos hizo el efecto.
# ===========================================================================

# --- Fijos: NO cambiar entre corridas o dejan de ser comparables -----------
MODELO_BASE = "Qwen/Qwen2.5-7B-Instruct"
DATASET_TRAIN = RAIZ / "data/dataset/train.json"
SEQ_LEN = 1024            # cambiarlo cambia que se trunca
BATCH_SIZE = 1            # batch efectivo = 1 x 8 = 8 (limite de los 16 GB)
ACUMULACION = 8
LORA_DROPOUT = 0.05
WARMUP_RATIO = 0.1
SCHEDULER = "cosine"      # decaimiento del learning rate (distinto del weight decay)
LORA_TARGET_MODULES = ["q_proj", "k_proj", "v_proj", "o_proj",
                       "gate_proj", "up_proj", "down_proj"]

# Validacion: se aparta un 5% del TRAIN (nunca del test) para ver la perdida de
# validacion epoca por epoca. Sin esto el entrenamiento es a ciegas: la perdida
# de entrenamiento siempre baja y el sobreajuste es justo el momento en que la
# de validacion se voltea hacia arriba. La semilla es fija a proposito: si
# cambia, dos corridas entrenan con datos distintos y dejan de ser comparables.
FRACCION_VALIDACION = 0.05
SEMILLA_VALIDACION = 42

PROYECTO_WANDB = "canuto"

# Prueba rapida de 3 pasos para ver que todo carga. No guarda adaptador.
PRUEBA_RAPIDA = False

try:
    import torch
    from datasets import Dataset
    from peft import LoraConfig, prepare_model_for_kbit_training
    from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig
    from trl import SFTConfig, SFTTrainer
except ImportError as exc:
    print(f"\nFalta una dependencia: {exc}")
    print("Instala el entorno con: bash install.sh")
    sys.exit(1)


def nombre_corrida():
    """Arma el nombre de la carpeta desde los hiperparametros.

    Ejemplo: t1_7b-e3-lr2e4-r16-wd0.0
    """
    lr = f"{LEARNING_RATE:.0e}".replace("e-0", "e").replace("e-", "e")
    return f"t{NUM_PRUEBA}_7b-e{EPOCAS}-lr{lr}-r{LORA_RANK}-wd{WEIGHT_DECAY}"


def cargar_textos(path, tokenizer):
    """Pasa los pares {instruction, output} al formato de chat de Qwen2.5.

    Sin system prompt: el modelo se entrena y responde solo user/assistant.
    """
    with open(path, encoding="utf-8") as f:
        records = json.load(f)
    textos = []
    for r in records:
        messages = [
            {"role": "user", "content": r["instruction"]},
            {"role": "assistant", "content": r["output"]},
        ]
        text = tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=False)
        textos.append({"text": text})
    return textos


def main():
    nombre = nombre_corrida()
    salida = RAIZ / "data/checkpoints" / nombre

    print(f"Corrida: {nombre}")
    print(f"  epocas={EPOCAS}  lr={LEARNING_RATE}  rank={LORA_RANK} "
          f"(alpha={LORA_RANK * 2})  weight_decay={WEIGHT_DECAY}")

    # Se mira el adaptador, no la carpeta: si una corrida se cayo a mitad, la
    # carpeta ya existe (el Trainer guarda dentro) pero no hay nada que perder,
    # y repetir esa misma configuracion tiene que poder hacerse.
    if (salida / "adapter_config.json").exists() and not PRUEBA_RAPIDA:
        print(f"\nYa hay un adaptador entrenado en {salida}")
        print("Sube NUM_PRUEBA (o borra esa carpeta) para no perder el checkpoint anterior.")
        sys.exit(1)

    if not torch.cuda.is_available():
        print("No se detecta GPU. Este script corre en la workstation.")
        sys.exit(1)
    print("GPU:", torch.cuda.get_device_name(0))

    print(f"[1/4] Tokenizador ({MODELO_BASE})...")
    tokenizer = AutoTokenizer.from_pretrained(MODELO_BASE, trust_remote_code=True)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    print("[2/4] Dataset...")
    textos = cargar_textos(DATASET_TRAIN, tokenizer)
    random.Random(SEMILLA_VALIDACION).shuffle(textos)
    n_val = max(1, int(len(textos) * FRACCION_VALIDACION))
    ds_val = Dataset.from_list(textos[:n_val])
    ds_train = Dataset.from_list(textos[n_val:])
    print(f"      {len(ds_train)} de entrenamiento + {n_val} de validacion")
    if n_val < 20:
        print(f"      OJO: con {n_val} ejemplos la curva de validacion sale irregular.")
        print("      Si cuesta leerla, sube FRACCION_VALIDACION a 0.10 y no la muevas mas.")

    print("[3/4] Modelo base en 4-bit (QLoRA)...")
    bnb_config = BitsAndBytesConfig(
        load_in_4bit=True,
        bnb_4bit_quant_type="nf4",
        bnb_4bit_use_double_quant=True,
        bnb_4bit_compute_dtype=torch.bfloat16,
    )
    model = AutoModelForCausalLM.from_pretrained(
        MODELO_BASE,
        quantization_config=bnb_config,
        device_map="auto",
        trust_remote_code=True,
    )
    model = prepare_model_for_kbit_training(model)

    lora_config = LoraConfig(
        r=LORA_RANK,
        lora_alpha=LORA_RANK * 2,
        target_modules=LORA_TARGET_MODULES,
        lora_dropout=LORA_DROPOUT,
        bias="none",
        task_type="CAUSAL_LM",
    )

    # W&B registra el ENTRENAMIENTO (las curvas y los hiperparametros). Las
    # metricas de evaluacion NO van aqui: salen de scripts/evaluate.py a la
    # pantalla y de ahi al Excel, para no tener dos fuentes de verdad.
    os.environ.setdefault("WANDB_PROJECT", PROYECTO_WANDB)

    print("[4/4] Entrenando...")
    training_args = SFTConfig(
        output_dir=str(salida / "_checkpoints"),
        num_train_epochs=EPOCAS,
        max_steps=3 if PRUEBA_RAPIDA else -1,
        per_device_train_batch_size=BATCH_SIZE,
        per_device_eval_batch_size=1,   # el default es 8 y no cabe en los 16 GB
        gradient_accumulation_steps=ACUMULACION,
        learning_rate=LEARNING_RATE,
        weight_decay=WEIGHT_DECAY,
        optim="paged_adamw_8bit",   # optimizador paginado: usa menos VRAM
        warmup_ratio=WARMUP_RATIO,
        lr_scheduler_type=SCHEDULER,
        eval_strategy="no" if PRUEBA_RAPIDA else "epoch",
        logging_steps=1 if PRUEBA_RAPIDA else 10,
        save_strategy="no" if PRUEBA_RAPIDA else "epoch",
        save_total_limit=1,
        gradient_checkpointing=True,
        gradient_checkpointing_kwargs={"use_reentrant": False},
        max_seq_length=SEQ_LEN,
        dataset_text_field="text",
        bf16=True,
        report_to="none" if PRUEBA_RAPIDA else "wandb",
        run_name=nombre,
    )

    trainer = SFTTrainer(
        model=model,
        train_dataset=ds_train,
        eval_dataset=ds_val,
        peft_config=lora_config,
        args=training_args,
    )
    trainer.train()

    if PRUEBA_RAPIDA:
        print("\nPrueba rapida OK. Pon PRUEBA_RAPIDA = False para entrenar completo.")
        return

    # Se guarda solo el adaptador: al servir se carga el base en 4-bit + el adaptador.
    salida.mkdir(parents=True, exist_ok=True)
    trainer.model.save_pretrained(str(salida))
    tokenizer.save_pretrained(str(salida))

    # Registro de TODOS los hiperparametros de esta corrida, los que se barren y
    # los fijos. El adapter_config.json que escribe PEFT solo guarda los de LoRA
    # (rank, alpha, dropout): las epocas, el learning rate y el weight decay no
    # quedan en ningun lado si no es aqui. evaluate.py lo lee para armar la fila
    # del Excel, y sirve para responder "que seq_len uso la corrida 4" meses
    # despues.
    hiperparametros = {
        "num_prueba": NUM_PRUEBA,
        "nombre": nombre,
        "epocas": EPOCAS,
        "learning_rate": LEARNING_RATE,
        "rank": LORA_RANK,
        "lora_alpha": LORA_RANK * 2,
        "lora_dropout": LORA_DROPOUT,
        "weight_decay": WEIGHT_DECAY,
        "batch_size": BATCH_SIZE,
        "acumulacion": ACUMULACION,
        "batch_efectivo": BATCH_SIZE * ACUMULACION,
        "seq_len": SEQ_LEN,
        "warmup_ratio": WARMUP_RATIO,
        "scheduler": SCHEDULER,
        "target_modules": LORA_TARGET_MODULES,
        "modelo_base": MODELO_BASE,
        "n_train": len(ds_train),
        "n_validacion": n_val,
        "fraccion_validacion": FRACCION_VALIDACION,
        "semilla_validacion": SEMILLA_VALIDACION,
    }
    with open(salida / "hiperparametros.json", "w", encoding="utf-8") as f:
        json.dump(hiperparametros, f, ensure_ascii=False, indent=2)

    print(f"\nAdaptador guardado en: {salida}")
    print("\nSiguiente paso:")
    print(f"  python scripts/evaluate.py --checkpoint data/checkpoints/{nombre}")
    print("Y en W&B: mirar en que epoca se voltea eval/loss (esa es la respuesta")
    print("a cuantas epocas usar). Perdida de entrenamiento < 0.2 = memorizacion.")


if __name__ == "__main__":
    main()
