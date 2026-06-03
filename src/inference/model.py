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
        self._tokenizer = AutoTokenizer.from_pretrained(
            self.checkpoint_path, trust_remote_code=True
        )
        # BF16 en GPU (mismo dtype usado en el entrenamiento), FP32 en CPU
        dtype = torch.bfloat16 if self.device != "cpu" else torch.float32
        self._model = AutoModelForCausalLM.from_pretrained(
            self.checkpoint_path,
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
        inputs = self._tokenizer([text], return_tensors="pt").to(self._model.device)

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
