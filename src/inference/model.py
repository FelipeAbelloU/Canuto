"""Carga el modelo fine-tuneado y genera las respuestas."""
from pathlib import Path


class FineTunedModel:
    def __init__(
        self,
        checkpoint_path: str,
        device: str = "cpu",
        max_new_tokens: int = 512,
        temperature: float = 0.3,
        top_p: float = 0.85,
    ):
        self.checkpoint_path = checkpoint_path
        self.device = device
        self.max_new_tokens = max_new_tokens
        self.temperature = temperature
        self.top_p = top_p
        self._model = None
        self._tokenizer = None

    def _load(self) -> None:
        """Carga el modelo la primera vez que se necesita (tarda y pesa varios GB)."""
        if self._model is not None:
            return
        import torch
        from transformers import AutoModelForCausalLM, AutoTokenizer

        print(f"  Cargando modelo desde: {self.checkpoint_path}")
        ckpt = Path(self.checkpoint_path)
        dtype = torch.bfloat16 if self.device != "cpu" else torch.float32

        if (ckpt / "adapter_config.json").exists():
            # Es un adaptador LoRA: se carga el modelo base y encima el adaptador.
            # En GPU el base va en 4-bit para que el 7B quepa en los 16 GB.
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
            # Es un modelo completo: se carga directo.
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
        print("  Modelo listo.")

    def is_available(self) -> bool:
        """Indica si el checkpoint existe en disco."""
        return bool(self.checkpoint_path and Path(self.checkpoint_path).exists())

    @property
    def hf_model(self):
        """El modelo de HuggingFace, para hacer forward pass directo (perplejidad)."""
        self._load()
        return self._model

    @property
    def tokenizer(self):
        self._load()
        return self._tokenizer

    @property
    def name(self) -> str:
        return Path(self.checkpoint_path).name if self.checkpoint_path else "sin modelo"

    def chat(self, messages: list) -> str:
        """Genera la respuesta a partir de la lista de mensajes [{role, content}]."""
        self._load()
        import torch

        text = self._tokenizer.apply_chat_template(
            messages,
            tokenize=False,
            add_generation_prompt=True,
        )
        destino = "cpu" if self.device == "cpu" else "cuda"
        inputs = self._tokenizer([text], return_tensors="pt").to(destino)

        # Con temperatura 0 la generacion es greedy: no se pasan temperature ni
        # top_p porque no aplican y transformers protesta si van en cero.
        opciones = {"do_sample": self.temperature > 0}
        if opciones["do_sample"]:
            opciones["temperature"] = self.temperature
            opciones["top_p"] = self.top_p

        with torch.no_grad():
            outputs = self._model.generate(
                **inputs,
                max_new_tokens=self.max_new_tokens,
                pad_token_id=self._tokenizer.eos_token_id,
                **opciones,
            )

        # Se recorta el prompt para quedarse solo con lo que genero el modelo.
        generado = outputs[0][inputs["input_ids"].shape[-1]:]
        return self._tokenizer.decode(generado, skip_special_tokens=True).strip()
