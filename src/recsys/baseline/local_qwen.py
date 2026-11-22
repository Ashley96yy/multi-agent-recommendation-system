"""Opt-in real local Qwen inference for the one-pass recommendation baseline.

This adapter downloads a public pinned model at initialization. It never runs in
the lightweight unit-test CI and does not add extra retries or agent calls.
"""

import re
from time import perf_counter


MODEL_ID = "Qwen/Qwen2.5-0.5B-Instruct"
MODEL_REVISION = "7ae557604adf67be50417f59c2c2f167def9a775"


class LocalQwenGenerator:
    """A real Transformers causal LM; CPU-first, deterministic greedy decoding."""

    def __init__(
        self,
        *,
        model_id: str = MODEL_ID,
        revision: str = MODEL_REVISION,
        device: str = "cpu",
        max_new_tokens: int = 240,
        max_prompt_tokens: int = 3000,
    ):
        if not re.fullmatch(r"[0-9a-f]{40}", revision):
            raise ValueError("model revision must be a 40-character commit SHA")
        if not model_id or not model_id.strip():
            raise ValueError("model_id must not be blank")
        if device not in {"cpu", "cuda", "mps"}:
            raise ValueError("unsupported device")
        if isinstance(max_new_tokens, bool) or not isinstance(max_new_tokens, int) or not 1 <= max_new_tokens <= 1024:
            raise ValueError("max_new_tokens must be in [1, 1024]")
        if isinstance(max_prompt_tokens, bool) or not isinstance(max_prompt_tokens, int) or max_prompt_tokens < 100:
            raise ValueError("max_prompt_tokens must be at least 100")
        try:
            import torch
            from transformers import AutoModelForCausalLM, AutoTokenizer
        except ImportError as exc:
            raise ImportError('Install local inference support: pip install -e ".[local-llm]"') from exc

        self.name = f"{model_id}@{revision}"
        self.model_id = model_id
        self.revision = revision
        self.device = device
        self.max_new_tokens = max_new_tokens
        self.max_prompt_tokens = max_prompt_tokens
        self._torch = torch
        self._tokenizer = AutoTokenizer.from_pretrained(
            model_id, revision=revision, trust_remote_code=False
        )
        self._model = AutoModelForCausalLM.from_pretrained(
            model_id,
            revision=revision,
            trust_remote_code=False,
            dtype=torch.float32,
        ).to(device)
        self._model.eval()
        self.call_count = 0
        self.last_response: str | None = None
        self.last_prompt_tokens = 0
        self.last_generated_tokens = 0
        self.last_generation_seconds = 0.0

    def generate(self, instructions: str, context: str) -> str:
        inputs = self._tokenizer.apply_chat_template(
            [
                {"role": "system", "content": instructions},
                {"role": "user", "content": context},
            ],
            tokenize=True,
            add_generation_prompt=True,
            return_tensors="pt",
        ).to(self.device)
        self.last_prompt_tokens = int(inputs.shape[-1])
        if self.last_prompt_tokens > self.max_prompt_tokens:
            raise ValueError("prompt exceeded configured token budget")
        self.call_count += 1
        self.last_response = None
        start = perf_counter()
        with self._torch.inference_mode():
            output = self._model.generate(
                input_ids=inputs,
                attention_mask=self._torch.ones_like(inputs),
                max_new_tokens=self.max_new_tokens,
                do_sample=False,
                pad_token_id=self._tokenizer.eos_token_id,
                eos_token_id=self._tokenizer.eos_token_id,
            )
        self.last_generation_seconds = perf_counter() - start
        tokens = output[0, inputs.shape[-1]:]
        self.last_generated_tokens = int(tokens.shape[-1])
        self.last_response = self._tokenizer.decode(tokens, skip_special_tokens=True).strip()
        return self.last_response
