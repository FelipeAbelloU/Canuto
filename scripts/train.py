"""
scripts/train.py — Fine-tuning de Qwen2.5-1.5B-Instruct sobre normativa Unillanos

TIEMPO ESTIMADO EN ESTA MÁQUINA (i5-7200U, CPU):
  Entrenamiento completo (3 épocas): ~15-30 horas segun tamanio del dataset
  Alternativa recomendada: Google Colab T4  →  ~20-30 minutos (gratis)
    Ver scripts/colab_train.py para la guia paso a paso

ANTES DE EJECUTAR EL ENTRENAMIENTO COMPLETO:
  1. pip install "peft>=0.13,<0.14" "trl>=0.12,<0.13" datasets
  2. Pausar OneDrive — el checkpoint final pesa ~3 GB y está dentro de la carpeta
  3. Cerrar Chrome, VSCode preview, apps pesadas — necesitas ~10 GB de RAM libre

Uso:
  python scripts/train.py --smoke          # prueba rápida (~10 min, 3 pasos)
  python scripts/train.py                  # entrenamiento completo
  python scripts/train.py --epochs 5       # más épocas si la calidad es baja
"""
import argparse
import json
import os
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent

# Forzar UTF-8 en consola Windows (evita errores con caracteres especiales)
try:
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")
except AttributeError:
    pass

# ---------------------------------------------------------------------------
# Imports: mensaje de error claro si faltan las dependencias de entrenamiento
# ---------------------------------------------------------------------------
try:
    import torch
    from datasets import Dataset
    from peft import LoraConfig
    from transformers import AutoModelForCausalLM, AutoTokenizer
    from trl import SFTConfig, SFTTrainer
except ImportError as exc:
    print(f"\nDependencia faltante: {exc}")
    print('Instala con:  pip install peft>=0.13 "trl>=0.12,<0.13" datasets')
    sys.exit(1)

# Módulos objetivo para LoRA en la familia Qwen2.5
_LORA_TARGET_MODULES = [
    "q_proj", "k_proj", "v_proj", "o_proj",
    "gate_proj", "up_proj", "down_proj",
]


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Fine-tuning normativa Unillanos con LoRA")
    p.add_argument(
        "--dataset",
        default=str(ROOT / "data/dataset/dataset_alpaca.json"),
        help="Dataset en formato Alpaca (JSON)",
    )
    p.add_argument(
        "--model",
        default="Qwen/Qwen2.5-1.5B-Instruct",
        help="Nombre o ruta del modelo base en HuggingFace",
    )
    p.add_argument(
        "--output",
        default=str(ROOT / "data/checkpoints/unillanos-v1"),
        help="Directorio de salida para el checkpoint final fusionado",
    )
    p.add_argument(
        "--config",
        default=str(ROOT / "config/config.yaml"),
        help="Ruta a config.yaml (se lee el system_prompt para coincidir con inferencia)",
    )
    p.add_argument("--epochs", type=int, default=3, help="Épocas de entrenamiento")
    p.add_argument("--rank", type=int, default=8, help="Rango de LoRA (r)")
    p.add_argument("--seq-len", type=int, default=750, help="Longitud máxima de secuencia en tokens")
    p.add_argument(
        "--smoke",
        action="store_true",
        help="Modo prueba: 3 pasos (~10 min) para verificar que el pipeline funciona",
    )
    return p.parse_args()


# ---------------------------------------------------------------------------
# Utilidades
# ---------------------------------------------------------------------------

def load_system_prompt(config_path: str) -> str:
    """Lee el system_prompt de config.yaml para que training e inferencia sean idénticos."""
    try:
        import yaml
        with open(config_path, encoding="utf-8") as f:
            cfg = yaml.safe_load(f)
        prompt = cfg.get("chat", {}).get("system_prompt", "").strip()
        if prompt:
            return prompt
    except Exception:
        pass
    return (
        "Eres un asistente especializado en normativa universitaria de la "
        "Universidad de los Llanos (Unillanos), Colombia."
    )


def build_dataset(path: str, tokenizer, system_prompt: str) -> Dataset:
    """Convierte pares Alpaca {instruction, output} al formato chat de Qwen2.5-Instruct."""
    with open(path, encoding="utf-8") as f:
        records = json.load(f)

    texts = []
    for r in records:
        messages = [
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": r["instruction"]},
            {"role": "assistant", "content": r["output"]},
        ]
        text = tokenizer.apply_chat_template(
            messages,
            tokenize=False,
            add_generation_prompt=False,  # False en training; True en inferencia
        )
        texts.append({"text": text})

    return Dataset.from_list(texts)


def print_dataset_stats(dataset: Dataset, tokenizer, seq_len: int) -> None:
    """Muestra estadísticas de longitud y alerta si hay ejemplos que se truncarán."""
    lengths = [len(tokenizer.encode(ex["text"])) for ex in dataset]
    exceeded = sum(1 for ln in lengths if ln > seq_len)
    avg = sum(lengths) / len(lengths)
    print(f"      Tokens promedio   : {avg:.0f}")
    print(f"      Tokens máximo     : {max(lengths)}")
    print(f"      Exceden seq_len   : {exceeded} / {len(lengths)}", end="")
    if exceeded:
        print(f"  ← serán truncados (usa --seq-len {max(lengths) + 64} para evitarlo)")
    else:
        print()


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main() -> None:
    args = parse_args()

    # Usar todos los hilos disponibles del CPU
    torch.set_num_threads(os.cpu_count() or 4)

    print("=" * 62)
    if args.smoke:
        print("  SMOKE TEST — 3 pasos para verificar que el pipeline funciona")
    else:
        print("  ENTRENAMIENTO — Asistente Normativo Unillanos")
    print("=" * 62)
    print(f"  Modelo  : {args.model}")
    print(f"  Dataset : {args.dataset}")
    print(f"  Salida  : {args.output}")

    if not args.smoke:
        print()
        print("  ⏱  TIEMPO ESTIMADO CPU (i5-7200U)  : ~15-30 horas")
        print("     Alternativa: Google Colab T4    : ~15-20 minutos (gratis)")
        print("  ⚠  Pausa OneDrive y cierra apps pesadas antes de continuar")
    print()

    # 1. System prompt — leído desde config.yaml para coincidir exactamente con inferencia
    system_prompt = load_system_prompt(args.config)
    print("[1/5] System prompt cargado desde config.yaml")

    # 2. Tokenizador
    print(f"[2/5] Cargando tokenizador ({args.model})...")
    tokenizer = AutoTokenizer.from_pretrained(args.model, trust_remote_code=True)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    # 3. Dataset
    print("[3/5] Preparando dataset...")
    dataset = build_dataset(args.dataset, tokenizer, system_prompt)
    print(f"      {len(dataset)} ejemplos")
    print_dataset_stats(dataset, tokenizer, args.seq_len)

    # 4. Modelo base
    print("[4/5] Cargando modelo base (primera vez descarga ~3 GB de HuggingFace)...")
    model = AutoModelForCausalLM.from_pretrained(
        args.model,
        torch_dtype=torch.float32,   # FP32 es la única opción estable en CPU
        low_cpu_mem_usage=True,       # reduce el pico de RAM durante la carga
        trust_remote_code=True,
    )
    total = sum(p.numel() for p in model.parameters())
    print(f"      {total / 1e6:.0f}M parámetros  |  RAM aprox.: {total * 4 / 1e9:.1f} GB (FP32)")

    # 5. LoRA + entrenamiento
    print("[5/5] Configurando LoRA y lanzando entrenamiento...")

    lora_config = LoraConfig(
        r=args.rank,
        lora_alpha=args.rank * 2,
        target_modules=_LORA_TARGET_MODULES,
        lora_dropout=0.05,
        bias="none",
        task_type="CAUSAL_LM",
    )

    output_path = Path(args.output)
    adapters_dir = output_path / "_adapters"      # backup LoRA antes del merge
    checkpoints_dir = output_path / "_checkpoints"

    training_args = SFTConfig(
        output_dir=str(checkpoints_dir),
        num_train_epochs=args.epochs,
        max_steps=3 if args.smoke else -1,         # 3 pasos en smoke, todas las épocas si no
        per_device_train_batch_size=1,
        gradient_accumulation_steps=1 if args.smoke else 8,
        learning_rate=2e-4,
        warmup_ratio=0.1,
        lr_scheduler_type="cosine",
        logging_steps=1 if args.smoke else 10,
        save_strategy="no" if args.smoke else "epoch",
        save_total_limit=1,
        gradient_checkpointing=True,               # reduce RAM de activaciones ~30-40%
        gradient_checkpointing_kwargs={"use_reentrant": False},  # compatibilidad con PEFT
        max_seq_length=args.seq_len,
        dataset_text_field="text",
        fp16=False,
        bf16=False,
        use_cpu=True,                              # fuerza CPU aunque haya CUDA disponible
        dataloader_num_workers=0,                  # evita problemas de multiprocessing en Windows
        report_to="none",
    )

    trainer = SFTTrainer(
        model=model,
        train_dataset=dataset,
        peft_config=lora_config,
        args=training_args,
    )

    trainer.train()

    # --- Smoke test: solo verificamos que el pipeline corrió sin errores ---
    if args.smoke:
        print()
        print("=" * 62)
        print("  ✓ Smoke test completado — el pipeline funciona correctamente")
        print("  Ejecuta sin --smoke para el entrenamiento completo")
        print("=" * 62)
        return

    # --- Entrenamiento completo: guardar adaptadores LoRA como backup ---
    print("\nGuardando adaptadores LoRA como backup (ligero, por si falla el merge)...")
    adapters_dir.mkdir(parents=True, exist_ok=True)
    trainer.model.save_pretrained(str(adapters_dir))
    tokenizer.save_pretrained(str(adapters_dir))
    print(f"  → {adapters_dir}")

    # --- Fusionar LoRA con modelo base y guardar checkpoint final ---
    # El merge sube temporalmente el uso de RAM: modelo base (~6 GB) + copia fusionada (~6 GB)
    # Si hay OOM aquí, los adaptadores en _adapters/ siguen intactos
    print("\nFusionando LoRA con modelo base...")
    print("  (pico de RAM ~12 GB — si hay OOM, carga los adaptadores de _adapters/)")
    merged = trainer.model.merge_and_unload()

    output_path.mkdir(parents=True, exist_ok=True)
    merged.save_pretrained(str(output_path))
    tokenizer.save_pretrained(str(output_path))

    print()
    print("=" * 62)
    print(f"  ✓ Checkpoint guardado en: {output_path}")
    print("=" * 62)
    print()
    print("Próximo paso — edita config/config.yaml:")
    print("  model:")
    print(f'    checkpoint_path: "{output_path.as_posix()}"')
    print()
    print("Luego prueba el chat:")
    print("  python scripts/chat_cli.py")


if __name__ == "__main__":
    main()
