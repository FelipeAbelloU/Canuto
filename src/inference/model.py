"""Carga un modelo fine-tuneado en formato HuggingFace y genera respuestas."""
from __future__ import annotations

from pathlib import Path


class FineTunedModel:
    """Wrapper sobre un checkpoint HuggingFace para inferencia conversacional."""

    def __init__(
        self,
        checkpoint_path: str,
        device: str = "cpu",
        max_new_tokens: int = 512,
        temperature: float = 0.7,
        top_p: float = 0.9,
    ):
        self.checkpoint_path = checkpoint_path
        self.device = device
        self.max_new_tokens = max_new_tokens
        self.temperature = temperature
        self.top_p = top_p
        self._model = None
        self._tokenizer = None
        self._loaded = False

    def _load(self) -> None:
        if self._loaded:
            return
        import torch
        from transformers import AutoModelForCausalLM, AutoTokenizer

        print(f"  Cargando modelo desde: {self.checkpoint_path}")
        ckpt = Path(self.checkpoint_path)
        # BF16 en GPU (mismo dtype del entrenamiento), FP32 en CPU
        dtype = torch.bfloat16 if self.device != "cpu" else torch.float32

        # Si el checkpoint es un adaptador LoRA (lo que guarda train_gpu.py), se carga
        # el modelo base y encima el adaptador. En GPU el base va en 4-bit (QLoRA) para
        # que el 7B/14B quepa en 16 GB. Si es un modelo completo, se carga directo.
        if (ckpt / "adapter_config.json").exists():
            import json
            from peft import PeftModel
            base_name = json.loads((ckpt / "adapter_config.json").read_text())["base_model_name_or_path"]
            print(f"  Adaptador LoRA sobre el modelo base: {base_name}")

            quant = None
            if self.device != "cpu":
                from transformers import BitsAndBytesConfig
                quant = BitsAndBytesConfig(
                    load_in_4bit=True,
                    bnb_4bit_quant_type="nf4",
                    bnb_4bit_use_double_quant=True,
                    bnb_4bit_compute_dtype=torch.bfloat16,
                )
            base = AutoModelForCausalLM.from_pretrained(
                base_name,
                torch_dtype=dtype,
                quantization_config=quant,
                device_map=self.device if self.device != "cpu" else None,
                trust_remote_code=True,
                low_cpu_mem_usage=True,
            )
            self._model = PeftModel.from_pretrained(base, str(ckpt))
            self._tokenizer = AutoTokenizer.from_pretrained(str(ckpt), trust_remote_code=True)
        else:
            self._tokenizer = AutoTokenizer.from_pretrained(str(ckpt), trust_remote_code=True)
            self._model = AutoModelForCausalLM.from_pretrained(
                str(ckpt),
                torch_dtype=dtype,
                device_map=self.device if self.device != "cpu" else None,
                trust_remote_code=True,
                low_cpu_mem_usage=True,
            )

        if self.device == "cpu":
            self._model = self._model.to("cpu")
        self._model.eval()
        self._loaded = True
        print("  Modelo listo.")

    def is_available(self) -> bool:
        """Retorna True si hay un checkpoint configurado y existe en disco."""
        return bool(
            self.checkpoint_path and Path(self.checkpoint_path).exists()
        )

    @property
    def hf_model(self):
        """Modelo HuggingFace cargado (para métricas como perplejidad en evaluate.py)."""
        self._load()
        return self._model

    @property
    def tokenizer(self):
        self._load()
        return self._tokenizer

    @property
    def name(self) -> str:
        return Path(self.checkpoint_path).name if self.checkpoint_path else "sin modelo"

    def chat(self, messages: list[dict]) -> str:
        """Genera respuesta dado un historial de mensajes [{role, content}]."""
        self._load()
        import torch

        text = self._tokenizer.apply_chat_template(
            messages,
            tokenize=False,
            add_generation_prompt=True,
        )
        target = "cpu" if self.device == "cpu" else "cuda"
        inputs = self._tokenizer([text], return_tensors="pt").to(target)

        with torch.no_grad():
            outputs = self._model.generate(
                **inputs,
                max_new_tokens=self.max_new_tokens,
                temperature=self.temperature,
                top_p=self.top_p,
                do_sample=self.temperature > 0,
                pad_token_id=self._tokenizer.eos_token_id,
            )

        generated = outputs[0][inputs["input_ids"].shape[-1]:]
        return self._tokenizer.decode(generated, skip_special_tokens=True).strip()

    def generate(self, prompt: str) -> str:
        """Genera respuesta dado un prompt de texto plano."""
        return self.chat([{"role": "user", "content": prompt}])
