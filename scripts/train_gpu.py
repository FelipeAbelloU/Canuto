"""Entrena el modelo con QLoRA sobre la normativa de Unillanos.

Corre en la workstation (RTX 5080, 16 GB). El modelo base se carga en 4-bit
para que quepa en la GPU y el resultado es un adaptador LoRA.

Antes de correr, mandar la cache de modelos al disco grande:
    export HF_HOME=/bodega/hf-cache

Uso:
    python scripts/train_gpu.py --smoke                     # prueba rapida de 3 pasos
    python scripts/train_gpu.py                             # entrenamiento completo
    python scripts/train_gpu.py --epochs 3 --lr 2e-4 --rank 16 --output data/checkpoints/prueba1
"""
import argparse
import json
import sys
from pathlib import Path

RAIZ = Path(__file__).resolve().parent.parent

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

# Capas donde se insertan los adaptadores LoRA en Qwen2.5.
LORA_TARGET_MODULES = ["q_proj", "k_proj", "v_proj", "o_proj",
                       "gate_proj", "up_proj", "down_proj"]


def parse_args():
    p = argparse.ArgumentParser(description="Entrenamiento QLoRA de CANUTO")
    p.add_argument("--dataset", default=str(RAIZ / "data/dataset/train.json"),
                   help="Split de entrenamiento (correr antes scripts/split_dataset.py)")
    p.add_argument("--model", default="Qwen/Qwen2.5-7B-Instruct", help="Modelo base")
    p.add_argument("--output", default=str(RAIZ / "data/checkpoints/canuto-7b"),
                   help="Carpeta donde se guarda el adaptador LoRA")
    p.add_argument("--epochs", type=int, default=3)
    p.add_argument("--rank", type=int, default=16, help="Rango de las matrices LoRA")
    p.add_argument("--lr", type=float, default=2e-4, help="Learning rate")
    p.add_argument("--seq-len", type=int, default=1024)
    p.add_argument("--smoke", action="store_true", help="Prueba rapida de 3 pasos")
    return p.parse_args()


def build_dataset(path, tokenizer):
    """Pasa los pares {instruction, output} al formato de chat de Qwen2.5."""
    with open(path, encoding="utf-8") as f:
        records = json.load(f)
    texts = []
    for r in records:
        messages = [
            {"role": "user", "content": r["instruction"]},
            {"role": "assistant", "content": r["output"]},
        ]
        text = tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=False)
        texts.append({"text": text})
    return Dataset.from_list(texts)


def main():
    args = parse_args()

    if not torch.cuda.is_available():
        print("No se detecta GPU. Este script corre en la workstation.")
        sys.exit(1)
    print("GPU:", torch.cuda.get_device_name(0))

    print(f"[1/4] Tokenizador ({args.model})...")
    tokenizer = AutoTokenizer.from_pretrained(args.model, trust_remote_code=True)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    print("[2/4] Dataset...")
    dataset = build_dataset(args.dataset, tokenizer)
    print(f"      {len(dataset)} ejemplos")

    print("[3/4] Modelo base en 4-bit (QLoRA)...")
    bnb_config = BitsAndBytesConfig(
        load_in_4bit=True,
        bnb_4bit_quant_type="nf4",
        bnb_4bit_use_double_quant=True,
        bnb_4bit_compute_dtype=torch.bfloat16,
    )
    model = AutoModelForCausalLM.from_pretrained(
        args.model,
        quantization_config=bnb_config,
        device_map="auto",
        trust_remote_code=True,
    )
    model = prepare_model_for_kbit_training(model)

    lora_config = LoraConfig(
        r=args.rank,
        lora_alpha=args.rank * 2,
        target_modules=LORA_TARGET_MODULES,
        lora_dropout=0.05,
        bias="none",
        task_type="CAUSAL_LM",
    )

    print("[4/4] Entrenando...")
    training_args = SFTConfig(
        output_dir=str(Path(args.output) / "_checkpoints"),
        num_train_epochs=args.epochs,
        max_steps=3 if args.smoke else -1,
        per_device_train_batch_size=1,
        gradient_accumulation_steps=8,
        learning_rate=args.lr,
        optim="paged_adamw_8bit",   # optimizador paginado: usa menos VRAM
        warmup_ratio=0.1,
        lr_scheduler_type="cosine",
        logging_steps=1 if args.smoke else 10,
        save_strategy="no" if args.smoke else "epoch",
        save_total_limit=1,
        gradient_checkpointing=True,
        gradient_checkpointing_kwargs={"use_reentrant": False},
        max_seq_length=args.seq_len,
        dataset_text_field="text",
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

    if args.smoke:
        print("\nSmoke test OK. Corre sin --smoke para entrenar completo.")
        return

    # Se guarda solo el adaptador: al servir se carga el base en 4-bit + el adaptador.
    out = Path(args.output)
    out.mkdir(parents=True, exist_ok=True)
    trainer.model.save_pretrained(str(out))
    tokenizer.save_pretrained(str(out))
    print(f"\nAdaptador guardado en: {out}")
    print("Siguiente paso: apuntar model.checkpoint_path de config/config.yaml a esa carpeta")


if __name__ == "__main__":
    main()
