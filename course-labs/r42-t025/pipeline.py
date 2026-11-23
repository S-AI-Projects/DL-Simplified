"""
R42-T025: Explicit text preprocessing: whitespace tokenization and training-only vocabulary.
Unseen tokens map to an explicit unknown ID (<UNK>).
"""
from typing import Dict, List, Set

UNK_TOKEN = "<UNK>"
UNK_ID = 0

class TextTokenizer:
    def __init__(self):
        self.vocab: Dict[str, int] = {UNK_TOKEN: UNK_ID}
        self.inverse_vocab: Dict[int, str] = {UNK_ID: UNK_TOKEN}

    def fit(self, texts: List[str]) -> None:
        idx = 1
        for text in texts:
            tokens = text.strip().split()
            for t in tokens:
                t_lower = t.lower()
                if t_lower not in self.vocab:
                    self.vocab[t_lower] = idx
                    self.inverse_vocab[idx] = t_lower
                    idx += 1

    def encode(self, text: str) -> List[int]:
        tokens = text.strip().split()
        return [self.vocab.get(t.lower(), UNK_ID) for t in tokens]

    def decode(self, token_ids: List[int]) -> List[str]:
        return [self.inverse_vocab.get(tid, UNK_TOKEN) for tid in token_ids]
