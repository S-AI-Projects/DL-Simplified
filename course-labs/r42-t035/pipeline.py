"""Explicit preprocessing contract for R42-T035."""
from typing import Any, Dict, List

class VocabularyEncoder:
    def __init__(self, unk_token: str = "<unk>"):
        self.unk_token = unk_token
        self.vocab = {unk_token: 0}
        self.is_fit = False

    def fit(self, records: List[Dict[str, Any]]):
        self.vocab = {self.unk_token: 0}
        idx = 1
        for r in records:
            tokens = r["text"].lower().strip().split()
            for t in tokens:
                if t not in self.vocab:
                    self.vocab[t] = idx
                    idx += 1
        self.is_fit = True

    def transform(self, records: List[Dict[str, Any]]) -> List[List[int]]:
        if not self.is_fit:
            raise RuntimeError("VocabularyEncoder must be fit on training data before transform")
        encoded = []
        for r in records:
            tokens = r["text"].lower().strip().split()
            encoded.append([self.vocab.get(t, 0) for t in tokens])
        return encoded
