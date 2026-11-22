"""True next-token constrained classification, not JSON post-processing.

Transformers' prefix_allowed_tokens_fn masks all tokens other than the
four pretokenized canonical labels and EOS. The output is ONE RAW LABEL,
not a generated JSON/rationale. Never claim this constrains free-form
rationales or automatically proves a claim has evidence.
"""

from time import perf_counter

from recsys.evaluation.atomic_claims import FOUR_LABELS


class CanonicalLabelTrie:
    """Pure token-level prefix constraint, independently unit-testable."""

    def __init__(self, tokenizer):
        self.eos_id = tokenizer.eos_token_id
        if self.eos_id is None:
            raise ValueError("tokenizer has no EOS ID")
        self.sequences: dict[str, tuple[int, ...]] = {}
        for label in FOUR_LABELS:
            seq = tuple(tokenizer.encode(label, add_special_tokens=False))
            if not seq or self.eos_id in seq:
                raise ValueError("invalid label token encoding")
            if tokenizer.decode(list(seq), skip_special_tokens=True) != label:
                raise ValueError("label tokenizer round-trip failed")
            if seq in self.sequences.values():
                raise ValueError("labels have ambiguous token encodings")
            self.sequences[label] = seq
        self.max_new_tokens = max(len(seq) for seq in self.sequences.values()) + 1

    def allowed(self, generated_suffix: tuple[int, ...]) -> list[int]:
        permitted: set[int] = set()
        for seq in self.sequences.values():
            if seq[: len(generated_suffix)] == generated_suffix:
                if len(generated_suffix) == len(seq):
                    permitted.add(self.eos_id)
                else:
                    permitted.add(seq[len(generated_suffix)])
        if not permitted:
            raise ValueError("generated token prefix deviates from fixed four-label trie")
        return sorted(permitted)

    def callback(self, prompt_tokens: int):
        if prompt_tokens <= 0:
            raise ValueError("empty prompt cannot be constrained")

        def allowed_next(batch_id, input_ids):
            if batch_id != 0:
                raise ValueError("stage 3.6 runs batch size exactly one")
            suffix = tuple(input_ids[prompt_tokens:].tolist())
            return self.allowed(suffix)

        return allowed_next


class ConstrainedLabelQwen:
    """Use exactly the model/tokenizer loaded by the unchanged LocalQwen adapter.

    No model replacement, external API, synonym mapping or second response
    generation. This adapter uses genuine Transformers logits constraints.
    """

    def __init__(self, base):
        for name in ("_model", "_tokenizer", "_torch", "device"):
            if not hasattr(base, name):
                raise TypeError("requires an initialized LocalQwenGenerator")
        self.base = base
        self.trie = CanonicalLabelTrie(base._tokenizer)
        self.name = base.name + "#prefix-constrained-canonical-labels"

    @property
    def call_count(self):
        return self.base.call_count

    @property
    def last_prompt_tokens(self):
        return self.base.last_prompt_tokens

    @property
    def last_generated_tokens(self):
        return self.base.last_generated_tokens

    @property
    def last_generation_seconds(self):
        return self.base.last_generation_seconds

    def generate(self, instructions: str, context: str) -> str:
        tokenizer = self.base._tokenizer
        torch = self.base._torch
        inputs = tokenizer.apply_chat_template(
            [{"role": "system", "content": instructions},
             {"role": "user", "content": context}],
            tokenize=True, add_generation_prompt=True, return_tensors="pt",
        ).to(self.base.device)
        prompt_tokens = int(inputs.shape[-1])
        self.base.last_prompt_tokens = prompt_tokens
        if prompt_tokens > self.base.max_prompt_tokens:
            raise ValueError("prompt exceeded configured token budget")
        self.base.call_count += 1
        self.base.last_response = None
        begin = perf_counter()
        with torch.inference_mode():
            output = self.base._model.generate(
                input_ids=inputs,
                attention_mask=torch.ones_like(inputs),
                max_new_tokens=self.trie.max_new_tokens,
                do_sample=False,
                prefix_allowed_tokens_fn=self.trie.callback(prompt_tokens),
                pad_token_id=self.trie.eos_id,
                eos_token_id=self.trie.eos_id,
            )
        self.base.last_generation_seconds = perf_counter() - begin
        token_ids = output[0, prompt_tokens:]
        self.base.last_generated_tokens = int(token_ids.shape[-1])
        decoded = tokenizer.decode(token_ids, skip_special_tokens=True,
                                   clean_up_tokenization_spaces=False)
        self.base.last_response = decoded
        if decoded not in FOUR_LABELS:
            raise ValueError("true constrained generation returned out-of-vocabulary label")
        return decoded
