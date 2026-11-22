"""No model downloads: assert constraints mask tokens BEFORE generation."""

from contextlib import nullcontext

import pytest

from recsys.evaluation.atomic_claims import FOUR_LABELS
from recsys.evaluation.label_constraints import CanonicalLabelTrie, ConstrainedLabelQwen


class TinyTokenizer:
    eos_token_id = 2

    def encode(self, text, *, add_special_tokens=False):
        assert not add_special_tokens
        return [ord(ch) + 100 for ch in text]

    def decode(self, tokens, skip_special_tokens=True, clean_up_tokenization_spaces=False):
        return "".join(chr(x - 100) for x in tokens if x != self.eos_token_id)

    def apply_chat_template(self, messages, **kwargs):
        assert len(messages) == 2
        assert kwargs["add_generation_prompt"] is True
        return FakeTensor([21, 22, 23])


class FakeTensor:
    def __init__(self, values):
        self.values = list(values)

    def to(self, device):
        return self

    @property
    def shape(self):
        return (1, len(self.values))

    def tolist(self):
        return list(self.values)

    def __getitem__(self, item):
        if isinstance(item, tuple):
            assert item[0] == 0
            item = item[1]
        if isinstance(item, slice):
            return FakeTensor(self.values[item])
        return self.values[item]


class FakeHF:
    """Uses the callback at each generation step, not a posthoc edit."""

    def __init__(self, tokenizer):
        self.tokenizer = tokenizer
        self.callbacks_checked = 0

    def generate(self, *, input_ids, prefix_allowed_tokens_fn, max_new_tokens, **kwargs):
        assert kwargs["do_sample"] is False
        assert kwargs["eos_token_id"] == self.tokenizer.eos_token_id
        target = self.tokenizer.encode("supported", add_special_tokens=False)
        suffix = []
        for token in target + [self.tokenizer.eos_token_id]:
            allowed = prefix_allowed_tokens_fn(0, FakeTensor(input_ids.values + suffix))
            assert token in allowed
            assert allowed and set(allowed) <= {
                self.tokenizer.eos_token_id, *self.tokenizer.encode("supported"),
                *self.tokenizer.encode("contradicted"),
                *self.tokenizer.encode("unsupported"),
                *self.tokenizer.encode("ambiguous"),
            }
            self.callbacks_checked += 1
            suffix.append(token)
        assert len(suffix) <= max_new_tokens
        return FakeTensor(input_ids.values + suffix)


class TinyTorch:
    @staticmethod
    def ones_like(inputs):
        return inputs

    @staticmethod
    def inference_mode():
        return nullcontext()


class FakeQwen:
    def __init__(self):
        self._tokenizer = TinyTokenizer()
        self._model = FakeHF(self._tokenizer)
        self._torch = TinyTorch()
        self.device = "cpu"
        self.max_prompt_tokens = 100
        self.name = "fake-qwen"
        self.call_count = 0
        self.last_response = None
        self.last_prompt_tokens = 0
        self.last_generated_tokens = 0
        self.last_generation_seconds = 0


def test_token_level_trie_has_exact_four_canonical_terminations():
    trie = CanonicalLabelTrie(TinyTokenizer())
    first = trie.allowed(())
    assert set(first) == {ord(c[0]) + 100 for c in FOUR_LABELS}
    for label in FOUR_LABELS:
        seq = trie.sequences[label]
        assert trie.allowed(seq) == [2]
        for i in range(len(seq)):
            assert seq[i] in trie.allowed(seq[:i])
    with pytest.raises(ValueError, match="deviates"):
        trie.allowed((9999,))
    with pytest.raises(ValueError, match="empty prompt"):
        trie.callback(0)


def test_constrained_real_adapter_uses_prefix_mask_before_emitting_label():
    base = FakeQwen()
    adapter = ConstrainedLabelQwen(base)
    got = adapter.generate("system", "fictional source context")
    assert got == "supported"
    assert base.call_count == 1
    assert adapter.call_count == 1
    assert adapter.last_generated_tokens == len("supported") + 1
    assert adapter.last_prompt_tokens == 3
    assert base._model.callbacks_checked == len("supported") + 1
    assert base.last_response == "supported"
    assert adapter.trie.max_new_tokens == len("contradicted") + 1


def test_bad_batch_and_off_trie_prefix_fail_closed():
    trie = CanonicalLabelTrie(TinyTokenizer())
    callback = trie.callback(3)
    with pytest.raises(ValueError, match="batch"):
        callback(1, FakeTensor([21, 22, 23]))
    with pytest.raises(ValueError, match="deviates"):
        callback(0, FakeTensor([21, 22, 23, 9999]))


def test_invalid_tokenizer_round_trip_and_missing_eos_rejected():
    class MissingEOS(TinyTokenizer):
        eos_token_id = None
    with pytest.raises(ValueError, match="EOS"):
        CanonicalLabelTrie(MissingEOS())

    class BadDecoder(TinyTokenizer):
        def decode(self, tokens, **kwargs):
            return "invented"
    with pytest.raises(ValueError, match="round-trip"):
        CanonicalLabelTrie(BadDecoder())
