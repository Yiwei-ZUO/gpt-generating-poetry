"""Data utilities for the poetry GPT project."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
import random

import torch


@dataclass
class CorpusConfig:
    path: str
    train_ratio: float = 0.9
    seed: int = 42


class CharCorpus:
    """Character-level corpus with train/validation splits."""

    def __init__(self, config: CorpusConfig):
        self.config = config
        self.path = Path(config.path)
        self.text = self.path.read_text(encoding="utf-8")
        chars = sorted(set(self.text))
        self.stoi = {ch: idx for idx, ch in enumerate(chars)}
        self.itos = {idx: ch for ch, idx in self.stoi.items()}
        encoded = torch.tensor([self.stoi[ch] for ch in self.text], dtype=torch.long)

        split_index = int(len(encoded) * config.train_ratio)
        self.train_data = encoded[:split_index]
        self.val_data = encoded[split_index:]
        self.vocab_size = len(chars)

    def encode(self, text: str) -> torch.Tensor:
        return torch.tensor([self.stoi[ch] for ch in text], dtype=torch.long)

    def decode(self, tokens: torch.Tensor | list[int]) -> str:
        if isinstance(tokens, torch.Tensor):
            values = tokens.tolist()
        else:
            values = tokens
        return "".join(self.itos[idx] for idx in values)

    def get_batch(
        self,
        split: str,
        batch_size: int,
        context_length: int,
        device: torch.device,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        data = self.train_data if split == "train" else self.val_data
        starts = torch.randint(0, len(data) - context_length - 1, (batch_size,))
        x = torch.stack([data[start : start + context_length] for start in starts])
        y = torch.stack([data[start + 1 : start + context_length + 1] for start in starts])
        return x.to(device), y.to(device)

    def sample_prompt(self, max_chars: int = 96) -> str:
        poems = [poem for poem in self.text.split("\n\n\n") if poem.strip()]
        if not poems:
            return ""

        rng = random.Random(self.config.seed)
        poem = rng.choice(poems)
        lines = [line for line in poem.splitlines() if line.strip()]
        prompt_lines = lines[:2]
        prompt = "\n".join(prompt_lines).strip()
        return prompt[:max_chars]
