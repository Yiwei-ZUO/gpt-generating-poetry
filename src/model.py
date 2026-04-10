"""Character-level GPT model."""

from __future__ import annotations

from dataclasses import dataclass
import math

import torch
from torch import nn
from torch.nn import functional as F


@dataclass
class GPTConfig:
    vocab_size: int
    context_length: int = 256
    n_layers: int = 6
    d_model: int = 256
    n_heads: int = 8
    dropout: float = 0.1
    mlp_ratio: int = 4
    attention_backend: str = "sdpa"
    position_embedding: str = "learned"


class CausalSelfAttention(nn.Module):
    def __init__(self, config: GPTConfig):
        super().__init__()
        if config.d_model % config.n_heads != 0:
            raise ValueError("d_model must be divisible by n_heads")

        self.n_heads = config.n_heads
        self.head_dim = config.d_model // config.n_heads
        self.dropout = config.dropout
        self.backend = config.attention_backend

        self.qkv = nn.Linear(config.d_model, 3 * config.d_model)
        self.proj = nn.Linear(config.d_model, config.d_model)
        self.attn_dropout = nn.Dropout(config.dropout)
        self.resid_dropout = nn.Dropout(config.dropout)
        self.register_buffer(
            "causal_mask",
            torch.tril(torch.ones(config.context_length, config.context_length, dtype=torch.bool)),
            persistent=False,
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        batch_size, seq_len, channels = x.shape
        qkv = self.qkv(x)
        q, k, v = qkv.chunk(3, dim=-1)

        q = q.view(batch_size, seq_len, self.n_heads, self.head_dim).transpose(1, 2)
        k = k.view(batch_size, seq_len, self.n_heads, self.head_dim).transpose(1, 2)
        v = v.view(batch_size, seq_len, self.n_heads, self.head_dim).transpose(1, 2)

        if self.backend == "sdpa":
            out = F.scaled_dot_product_attention(
                q,
                k,
                v,
                attn_mask=None,
                dropout_p=self.dropout if self.training else 0.0,
                is_causal=True,
            )
        elif self.backend == "masked":
            scores = (q @ k.transpose(-2, -1)) / math.sqrt(self.head_dim)
            mask = self.causal_mask[:seq_len, :seq_len]
            scores = scores.masked_fill(~mask, float("-inf"))
            attn = F.softmax(scores, dim=-1)
            attn = self.attn_dropout(attn)
            out = attn @ v
        else:
            raise ValueError(f"Unknown attention backend: {self.backend}")

        out = out.transpose(1, 2).contiguous().view(batch_size, seq_len, channels)
        out = self.proj(out)
        return self.resid_dropout(out)


class FeedForward(nn.Module):
    def __init__(self, config: GPTConfig):
        super().__init__()
        hidden = config.d_model * config.mlp_ratio
        self.net = nn.Sequential(
            nn.Linear(config.d_model, hidden),
            nn.GELU(),
            nn.Linear(hidden, config.d_model),
            nn.Dropout(config.dropout),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x)


class TransformerBlock(nn.Module):
    def __init__(self, config: GPTConfig):
        super().__init__()
        self.ln_1 = nn.LayerNorm(config.d_model)
        self.attn = CausalSelfAttention(config)
        self.ln_2 = nn.LayerNorm(config.d_model)
        self.ffn = FeedForward(config)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = x + self.attn(self.ln_1(x))
        x = x + self.ffn(self.ln_2(x))
        return x


class CharGPT(nn.Module):
    def __init__(self, config: GPTConfig):
        super().__init__()
        self.config = config
        self.token_embedding = nn.Embedding(config.vocab_size, config.d_model)
        self.dropout = nn.Dropout(config.dropout)
        self.blocks = nn.ModuleList([TransformerBlock(config) for _ in range(config.n_layers)])
        self.ln_f = nn.LayerNorm(config.d_model)
        self.lm_head = nn.Linear(config.d_model, config.vocab_size, bias=False)

        if config.position_embedding == "learned":
            self.position_embedding = nn.Embedding(config.context_length, config.d_model)
            self.register_buffer("sinusoidal_positions", torch.empty(0), persistent=False)
        elif config.position_embedding == "sinusoidal":
            self.position_embedding = None
            self.register_buffer(
                "sinusoidal_positions",
                self._build_sinusoidal_table(config.context_length, config.d_model),
                persistent=False,
            )
        else:
            raise ValueError(f"Unknown position embedding: {config.position_embedding}")

        self.apply(self._init_weights)

    @staticmethod
    def _build_sinusoidal_table(context_length: int, d_model: int) -> torch.Tensor:
        position = torch.arange(context_length, dtype=torch.float32).unsqueeze(1)
        div_term = torch.exp(
            torch.arange(0, d_model, 2, dtype=torch.float32) * (-math.log(10000.0) / d_model)
        )
        table = torch.zeros(context_length, d_model)
        table[:, 0::2] = torch.sin(position * div_term)
        table[:, 1::2] = torch.cos(position * div_term)
        return table

    @staticmethod
    def _init_weights(module: nn.Module) -> None:
        if isinstance(module, (nn.Linear, nn.Embedding)):
            nn.init.normal_(module.weight, mean=0.0, std=0.02)
            if isinstance(module, nn.Linear) and module.bias is not None:
                nn.init.zeros_(module.bias)

    def forward(
        self,
        tokens: torch.Tensor,
        targets: torch.Tensor | None = None,
    ) -> tuple[torch.Tensor, torch.Tensor | None]:
        batch_size, seq_len = tokens.shape
        if seq_len > self.config.context_length:
            raise ValueError("Sequence length exceeds context length")

        positions = torch.arange(0, seq_len, device=tokens.device)
        tok = self.token_embedding(tokens)

        if self.position_embedding is not None:
            pos = self.position_embedding(positions)[None, :, :]
        else:
            pos = self.sinusoidal_positions[:seq_len].to(tokens.device)[None, :, :]

        x = self.dropout(tok + pos)
        for block in self.blocks:
            x = block(x)
        x = self.ln_f(x)
        logits = self.lm_head(x)

        loss = None
        if targets is not None:
            loss = F.cross_entropy(logits.view(batch_size * seq_len, -1), targets.view(-1))
        return logits, loss

    @torch.no_grad()
    def generate(
        self,
        tokens: torch.Tensor,
        max_new_tokens: int,
        temperature: float = 1.0,
        top_k: int | None = 40,
    ) -> torch.Tensor:
        for _ in range(max_new_tokens):
            idx_cond = tokens[:, -self.config.context_length :]
            logits, _ = self(idx_cond)
            next_logits = logits[:, -1, :] / max(temperature, 1e-6)

            if top_k is not None:
                values, _ = torch.topk(next_logits, min(top_k, next_logits.size(-1)))
                next_logits[next_logits < values[:, [-1]]] = float("-inf")

            probs = F.softmax(next_logits, dim=-1)
            next_token = torch.multinomial(probs, num_samples=1)
            tokens = torch.cat([tokens, next_token], dim=1)
        return tokens

