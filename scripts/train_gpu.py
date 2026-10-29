"""Entrena el adaptador LoRA (QLoRA, base en 4 bits) en la workstation.

Sin flags: los hiperparametros se editan abajo. La carpeta de salida se arma
con ellos, asi que no se puede sobrescribir un checkpoint por error.

Antes de correr:
    export HF_HOME=/bodega/hf-cache   # cache de modelos en el disco grande
    wandb login                       # una sola vez al inicio de todo el proceso
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
sys.path.insert(0, str(RAIZ))

from src.config_loader import load_config

# Hiperparametros de la corrida: editar a mano antes de entrenar.
NUM_PRUEBA = 1            # numero de prueba del dataset v2: prefijo de la carpeta (v2t1, v2t2, ...)
EPOCAS = 4                # 4, 6, 8
LEARNING_RATE = 2e-4      # 1e-4, 2e-4, 3e-4
LORA_RANK = 16            # 8, 16, 32  (alpha = rank * 2, se deriva solo)
WEIGHT_DECAY = 0.01       # 0.01, 0.05, 0.1   NUNCA 0 = sin regularizacion
LORA_DROPOUT = 0.05       # 0.0, 0.05, 0.1    regularizacion propia de LoRA
WARMUP_RATIO = 0.1        # 0.03, 0.1, 0.2    (5-10% de los pasos es lo usual)

# Barrido: mover una sola perilla por corrida, en este orden de prioridad:
#   epocas > learning rate > weight decay > lora_dropout > rank > warmup_ratio
# Mas de 3 epocas: en la corrida 1 la validacion seguia bajando en la tercera y
# no se vio donde empieza el sobreajuste. Weight decay 0 seria sin regularizacion.

# Fijos: cambiarlos rompe la comparacion entre corridas.
MODELO_BASE = "Qwen/Qwen2.5-7B-Instruct"
DATASET_TRAIN = RAIZ / "data/dataset/train.json"
SEQ_LEN = 2048            # con 1024 se cortaba el 27 % del texto
BATCH_SIZE = 1            # batch efectivo = 1 x 8 = 8 (limite de los 16 GB)
ACUMULACION = 8
SCHEDULER = "cosine"
LORA_TARGET_MODULES = ["q_proj", "k_proj", "v_proj", "o_proj",
                       "gate_proj", "up_proj", "down_proj"]

# Cada cuantos pasos se mide la validacion (mas chico = mas lento). Con el
# dataset v2 son ~5.000 pasos por epoca: 1000 da ~5 puntos por epoca.
EVAL_STEPS = 1000

# Validacion: 5 % del train (nunca del test). Semilla fija para que todas las
# corridas validen con los mismos ejemplos.
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
    from transformers.trainer_utils import get_last_checkpoint
    from trl import SFTConfig, SFTTrainer
except ImportError as exc:
    print(f"\nFalta una dependencia: {exc}")
    print("Instala el entorno con: bash install.sh")
    sys.exit(1)


def nombre_corrida():
    """Arma el nombre de la carpeta desde los hiperparametros.

    Ejemplo: v2t1_7b-e4-lr2e4-r16-wd0.01-do0.05-wu0.1

    Van los SEIS que se barren. Si faltaran el dropout y el warmup, dos corridas
    que solo difieren en uno de esos dos caerian en la misma carpeta.
    """
    lr = f"{LEARNING_RATE:.0e}".replace("e-0", "e").replace("e-", "e")
    return (f"v2t{NUM_PRUEBA}_7b-e{EPOCAS}-lr{lr}-r{LORA_RANK}"
            f"-wd{WEIGHT_DECAY}-do{LORA_DROPOUT}-wu{WARMUP_RATIO}")


def quedarse_con_el_token(logits, labels):
    """Se queda solo con el token mas probable de cada posicion.

    Hace falta para poder calcular el accuracy sin quedarse sin RAM: el modelo
    saca un numero por cada una de las 152.000 palabras del vocabulario y por
    cada token de los 474 ejemplos de validacion. Guardar todo eso son decenas
    de GB. Con quedarse con el ganador (argmax) basta para contar aciertos.
    """
    return logits.argmax(dim=-1)


def accuracy_tokens(evaluacion):
    """Que fraccion de los tokens de validacion acierta el modelo.

    Es la curva de "accuracy" que hacia falta para acompanar a la de perdida.
    Se saca del paso de validacion que ya se hace, sin generar texto.

    OJO, y hay que decirlo asi en el documento: este numero esta INFLADO. Los
    pares de preguntas y respuestas salen de plantillas, asi que buena parte de
    los tokens son formulas repetidas y faciles de acertar; ademas aqui cuentan
    tambien los tokens de la pregunta, no solo los de la respuesta. Sirve como
    curva de diagnostico (si se dispara, esta memorizando), NO para elegir cual
    corrida gana. Para eso estan la exactitud de citas y la tasa de abstencion.
    """
    predicciones, etiquetas = evaluacion
    # Lo predicho en t se compara con el token real de t+1.
    predicciones = predicciones[:, :-1]
    etiquetas = etiquetas[:, 1:]
    # -100 es el relleno de los ejemplos cortos: no cuenta.
    reales = etiquetas != -100
    aciertos = (predicciones == etiquetas) & reales
    return {"accuracy_tokens": float(aciertos.sum() / reales.sum())}


def cargar_textos(path, tokenizer, system_prompt):
    """Pasa los pares {instruction, output} al formato de chat de Qwen2.5.

    Con el system prompt de config.yaml, el mismo que usan el chat y evaluate.py.
    """
    with open(path, encoding="utf-8") as f:
        records = json.load(f)
    textos = []
    for r in records:
        messages = [
            {"role": "system", "content": system_prompt},
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
    print(f"  lora_dropout={LORA_DROPOUT}  warmup_ratio={WARMUP_RATIO}  "
          f"eval_steps={EVAL_STEPS}")

    # Se mira el adaptador y no la carpeta: una corrida caida se puede repetir.
    if (salida / "adapter_config.json").exists() and not PRUEBA_RAPIDA:
        print(f"\nYa hay un adaptador entrenado en {salida}")
        print("Sube NUM_PRUEBA (o borrar esa carpeta) para no perder el checkpoint anterior.")
        sys.exit(1)

    if not torch.cuda.is_available():
        print("No se detecta GPU. Este script corre en la workstation.")
        sys.exit(1)
    total_vram = torch.cuda.get_device_properties(0).total_memory / 1024**3
    print(f"GPU: {torch.cuda.get_device_name(0)}  ({total_vram:.1f} GB, "
          f"CUDA {torch.version.cuda})")
    torch.cuda.reset_peak_memory_stats()

    print(f"[1/4] Tokenizador ({MODELO_BASE})...")
    tokenizer = AutoTokenizer.from_pretrained(MODELO_BASE, trust_remote_code=True)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    print("[2/4] Dataset...")
    system_prompt = load_config(str(RAIZ / "config/config.yaml"))["model"]["system_prompt"]
    textos = cargar_textos(DATASET_TRAIN, tokenizer, system_prompt)
    print(f"      system prompt: {system_prompt[:60]}...")
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
        # GPU 0 explicita: con "auto", si falta VRAM baja capas a CPU en silencio.
        device_map={"": 0},
        trust_remote_code=True,
    )
    model = prepare_model_for_kbit_training(model)

    # Deja en el log en que dispositivos quedaron los pesos.
    dispositivos = {str(p.device) for p in model.parameters()}
    print(f"      parametros en: {sorted(dispositivos)}")
    print(f"      VRAM ocupada tras cargar: "
          f"{torch.cuda.memory_allocated() / 1024**3:.2f} GB")
    if any(d == "cpu" for d in dispositivos):
        print("      OJO: hay capas en CPU. La corrida va a ir lentisima.")
        sys.exit(1)

    lora_config = LoraConfig(
        r=LORA_RANK,
        lora_alpha=LORA_RANK * 2,
        target_modules=LORA_TARGET_MODULES,
        lora_dropout=LORA_DROPOUT,
        bias="none",
        task_type="CAUSAL_LM",
    )

    # W&B solo guarda el entrenamiento; las metricas salen de evaluate.py.
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
        # Por pasos y no por epoca: con 3 o 4 puntos no se ve donde se voltea.
        eval_strategy="steps",
        eval_steps=2 if PRUEBA_RAPIDA else EVAL_STEPS,
        logging_steps=1 if PRUEBA_RAPIDA else 10,
        save_strategy="no" if PRUEBA_RAPIDA else "epoch",
        # Un adaptador por epoca. Con coseno, la epoca 3 de una corrida de 4 no
        # equivale a una corrida de 3 epocas.
        save_total_limit=EPOCAS,
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
        compute_metrics=accuracy_tokens,
        preprocess_logits_for_metrics=quedarse_con_el_token,
    )
    # Si la maquina se reinicio a mitad, sigue desde el ultimo checkpoint (uno por
    # epoca) en vez de empezar de cero. No cambiar train.json antes de retomar.
    carpeta_checkpoints = salida / "_checkpoints"
    ultimo = None
    if carpeta_checkpoints.exists() and not PRUEBA_RAPIDA:
        ultimo = get_last_checkpoint(str(carpeta_checkpoints))
    if ultimo:
        print(f"      retomando desde {ultimo}")
        # torch 2.6+ no deja leer rng_state.pth (trae arreglos de numpy) con la
        # version de transformers que usamos; el archivo lo guardamos nosotros.
        import numpy as np
        torch.serialization.add_safe_globals([
            np._core.multiarray._reconstruct, np.ndarray, np.dtype,
            type(np.dtype(np.uint32)),
        ])
    trainer.train(resume_from_checkpoint=ultimo)

    if PRUEBA_RAPIDA:
        print(f"\nVRAM maxima usada: {torch.cuda.max_memory_allocated() / 1024**3:.2f} GB "
              f"de {total_vram:.1f} GB")
        print("Prueba rapida OK. Pon PRUEBA_RAPIDA = False para entrenar completo.")
        return

    # Se guarda solo el adaptador: al servir se carga el base en 4-bit + el adaptador.
    salida.mkdir(parents=True, exist_ok=True)
    trainer.model.save_pretrained(str(salida))
    tokenizer.save_pretrained(str(salida))

    # PEFT solo guarda los de LoRA; epocas, learning rate y demas quedan aqui
    # (los lee evaluate.py para la fila del Excel).
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
        "eval_steps": EVAL_STEPS,
        "vram_pico_gb": round(torch.cuda.max_memory_allocated() / 1024**3, 2),
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
    print(f"VRAM maxima usada: {torch.cuda.max_memory_allocated() / 1024**3:.2f} GB "
          f"de {total_vram:.1f} GB")
    print("\nSiguiente paso:")
    print(f"  python scripts/evaluate.py --checkpoint data/checkpoints/{nombre}")
    print("Y en W&B: mirar en que epoca se voltea eval/loss (esa es la respuesta")
    print("a cuantas epocas usar). Perdida de entrenamiento < 0.2 = memorizacion.")


if __name__ == "__main__":
    main()
