"""Train a character-level GPT on the final sonnet corpus.

The default configuration follows the starting point suggested in the
project instructions: context length 256, 6 layers, hidden size 256,
8 attention heads, 4x MLP expansion, AdamW, dropout 0.1, a
learning rate of 3e-4, and 40k training steps.
"""

from __future__ import annotations

import argparse
from dataclasses import asdict, dataclass
from datetime import datetime
import json
import math
import os
from pathlib import Path
import random
import sys
from typing import Any

import numpy as np
import torch

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT / "src") not in sys.path:
    sys.path.insert(0, str(ROOT / "src"))

from character_corpus import CharCorpus, CorpusConfig
from model import CharGPT, GPTConfig


# Configuration
@dataclass
class TrainingConfig:
    corpus_path: str = "data/cleaned/final_sonnets_corpus.txt"
    output_dir: str = "outputs"
    run_name: str = "baseline"
    seed: int = 42
    batch_size: int = 32
    learning_rate: float = 3e-4
    weight_decay: float = 0.1
    training_steps: int = 40000
    warmup_steps: int = 1000
    eval_batches: int = 32
    log_interval: int = 200
    eval_interval: int = 500
    sample_interval: int = 1000
    checkpoint_interval: int = 2000
    sample_tokens: int = 512
    grad_clip: float = 1.0
    temperature: float = 0.9
    top_k: int = 40
    device: str = "auto"
    plot_curves: bool = True


# Small utilities
def set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def resolve_device(device_name: str) -> torch.device:
    if device_name == "auto":
        if torch.cuda.is_available():
            return torch.device("cuda")
        if torch.backends.mps.is_available():
            return torch.device("mps")
        return torch.device("cpu")
    return torch.device(device_name)


def bits_per_character(loss: float) -> float:
    return loss / math.log(2)


def is_logging_step(step: int, interval: int, final_step: int) -> bool:
    return step % interval == 0 or step == final_step


def learning_rate_for_step(step: int, config: TrainingConfig) -> float:
    if step < config.warmup_steps:
        return config.learning_rate * (step + 1) / max(1, config.warmup_steps)

    progress = (step - config.warmup_steps) / max(1, config.training_steps - config.warmup_steps)
    cosine = 0.5 * (1.0 + math.cos(math.pi * min(progress, 1.0)))
    return config.learning_rate * cosine


# Evaluation and outputs
@torch.no_grad()
def evaluate(
    model: CharGPT,
    corpus: CharCorpus,
    train_config: TrainingConfig,
    model_config: GPTConfig,
    device: torch.device,
) -> dict[str, float]:
    model.eval()
    losses: dict[str, list[float]] = {"train": [], "val": []}
    for split in ("train", "val"):
        for _ in range(train_config.eval_batches):
            xb, yb = corpus.get_batch(split, train_config.batch_size, model_config.context_length, device)
            _, loss = model(xb, yb)
            losses[split].append(float(loss.item()))
    model.train()
    return {split: sum(values) / len(values) for split, values in losses.items()}


def save_checkpoint(
    run_dir: Path,
    name: str,
    model: CharGPT,
    optimizer: torch.optim.Optimizer,
    corpus: CharCorpus,
    train_config: TrainingConfig,
    model_config: GPTConfig,
    step: int,
) -> None:
    payload = {
        "step": step,
        "model_state": model.state_dict(),
        "optimizer_state": optimizer.state_dict(),
        "train_config": asdict(train_config),
        "model_config": asdict(model_config),
        "stoi": corpus.stoi,
        "itos": corpus.itos,
    }
    torch.save(payload, run_dir / name)


def append_jsonl(path: Path, record: dict[str, Any]) -> None:
    with path.open("a", encoding="utf-8") as handle:
        handle.write(json.dumps(record) + "\n")


def plot_metrics(run_dir: Path, records: list[dict[str, Any]]) -> None:
    eval_records = [record for record in records if "val_loss" in record]
    if not eval_records:
        return

    cache_dir = run_dir / ".mpl_cache"
    cache_dir.mkdir(parents=True, exist_ok=True)
    os.environ.setdefault("MPLCONFIGDIR", str(cache_dir))
    os.environ.setdefault("XDG_CACHE_HOME", str(cache_dir))
    import matplotlib.pyplot as plt

    steps = [record["step"] for record in eval_records]
    train_losses = [record["train_loss"] for record in eval_records]
    val_losses = [record["val_loss"] for record in eval_records]
    train_bpc = [record["train_bpc"] for record in eval_records]
    val_bpc = [record["val_bpc"] for record in eval_records]

    fig, axes = plt.subplots(1, 2, figsize=(12, 4))
    axes[0].plot(steps, train_losses, label="train")
    axes[0].plot(steps, val_losses, label="validation")
    axes[0].set_title("Loss")
    axes[0].set_xlabel("Step")
    axes[0].set_ylabel("Cross-entropy")
    axes[0].legend()

    axes[1].plot(steps, train_bpc, label="train")
    axes[1].plot(steps, val_bpc, label="validation")
    axes[1].set_title("Bits per character")
    axes[1].set_xlabel("Step")
    axes[1].set_ylabel("BPC")
    axes[1].legend()

    fig.tight_layout()
    fig.savefig(run_dir / "training_curves.png", dpi=160)
    plt.close(fig)


def save_sample(run_dir: Path, step: int, prompt: str, sample: str) -> None:
    sample_path = run_dir / "samples.txt"
    with sample_path.open("a", encoding="utf-8") as handle:
        handle.write(f"=== STEP {step} ===\n")
        handle.write("PROMPT:\n")
        handle.write(prompt + "\n\n")
        handle.write("SAMPLE:\n")
        handle.write(sample + "\n\n")


def print_sample(step: int, prompt: str, sample: str, max_chars: int = 600) -> None:
    preview = sample[:max_chars].rstrip()
    print(f"[sample {step:05d}] prompt:")
    print(prompt)
    print(f"[sample {step:05d}] generated:")
    print(preview)
    print()


def build_run_dir(output_dir: str, run_name: str) -> Path:
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    run_dir = Path(output_dir) / f"{run_name}_{timestamp}"
    run_dir.mkdir(parents=True, exist_ok=True)
    return run_dir


def save_run_config(
    run_dir: Path,
    train_config: TrainingConfig,
    model_config: GPTConfig,
    vocab_size: int,
    device: torch.device,
) -> None:
    payload = {
        "training": asdict(train_config),
        "model": asdict(model_config),
        "vocab_size": vocab_size,
        "device": str(device),
    }
    (run_dir / "config.json").write_text(json.dumps(payload, indent=2), encoding="utf-8")


# Training loop
def train_one_run(train_config: TrainingConfig, model_config: GPTConfig) -> Path:
    set_seed(train_config.seed)
    device = resolve_device(train_config.device)
    corpus = CharCorpus(CorpusConfig(path=train_config.corpus_path, seed=train_config.seed))
    model_config.vocab_size = corpus.vocab_size
    model = CharGPT(model_config).to(device)
    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=train_config.learning_rate,
        weight_decay=train_config.weight_decay,
    )

    run_dir = build_run_dir(train_config.output_dir, train_config.run_name)
    save_run_config(run_dir, train_config, model_config, corpus.vocab_size, device)

    metrics_path = run_dir / "metrics.jsonl"
    records: list[dict[str, Any]] = []
    best_val_loss = float("inf")
    final_step = train_config.training_steps - 1

    for step in range(train_config.training_steps):
        current_lr = learning_rate_for_step(step, train_config)
        for group in optimizer.param_groups:
            group["lr"] = current_lr

        xb, yb = corpus.get_batch("train", train_config.batch_size, model_config.context_length, device)
        _, loss = model(xb, yb)

        optimizer.zero_grad(set_to_none=True)
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), train_config.grad_clip)
        optimizer.step()

        if is_logging_step(step, train_config.log_interval, final_step):
            train_loss = float(loss.item())
            record = {
                "step": step,
                "learning_rate": current_lr,
                "train_batch_loss": train_loss,
                "train_batch_bpc": bits_per_character(train_loss),
            }
            append_jsonl(metrics_path, record)
            records.append(record)
            print(
                f"[step {step:05d}] train_loss={train_loss:.4f} "
                f"train_bpc={bits_per_character(train_loss):.4f} lr={current_lr:.6f}"
            )

        if is_logging_step(step, train_config.eval_interval, final_step):
            metrics = evaluate(model, corpus, train_config, model_config, device)
            record = {
                "step": step,
                "train_loss": metrics["train"],
                "val_loss": metrics["val"],
                "train_bpc": bits_per_character(metrics["train"]),
                "val_bpc": bits_per_character(metrics["val"]),
                "learning_rate": current_lr,
            }
            append_jsonl(metrics_path, record)
            records.append(record)
            if train_config.plot_curves:
                plot_metrics(run_dir, records)
            print(
                f"[eval {step:05d}] train_loss={metrics['train']:.4f} "
                f"val_loss={metrics['val']:.4f} val_bpc={bits_per_character(metrics['val']):.4f}"
            )

            if metrics["val"] < best_val_loss:
                best_val_loss = metrics["val"]
                save_checkpoint(
                    run_dir,
                    "checkpoint_best.pt",
                    model,
                    optimizer,
                    corpus,
                    train_config,
                    model_config,
                    step,
                )

        if is_logging_step(step, train_config.sample_interval, final_step):
            prompt = corpus.sample_prompt()
            prompt_ids = corpus.encode(prompt).unsqueeze(0).to(device)
            generated = model.generate(
                prompt_ids,
                max_new_tokens=train_config.sample_tokens,
                temperature=train_config.temperature,
                top_k=train_config.top_k,
            )
            sample = corpus.decode(generated[0].detach().cpu())
            save_sample(run_dir, step, prompt, sample)
            print_sample(step, prompt, sample)

        if is_logging_step(step, train_config.checkpoint_interval, final_step):
            save_checkpoint(
                run_dir,
                f"checkpoint_step_{step:05d}.pt",
                model,
                optimizer,
                corpus,
                train_config,
                model_config,
                step,
            )

    return run_dir


# CLI
def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--corpus-path", default="data/cleaned/final_sonnets_corpus.txt")
    parser.add_argument("--output-dir", default="outputs")
    parser.add_argument("--run-name", default="baseline")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--context-length", type=int, default=256)
    parser.add_argument("--n-layers", type=int, default=6)
    parser.add_argument("--d-model", type=int, default=256)
    parser.add_argument("--n-heads", type=int, default=8)
    parser.add_argument("--mlp-ratio", type=int, default=4)
    parser.add_argument("--dropout", type=float, default=0.1)
    parser.add_argument("--position-embedding", choices=["learned", "sinusoidal"], default="learned")
    parser.add_argument("--attention-backend", choices=["sdpa", "masked"], default="sdpa")
    parser.add_argument("--batch-size", type=int, default=32)
    parser.add_argument("--learning-rate", type=float, default=3e-4)
    parser.add_argument("--weight-decay", type=float, default=0.1)
    parser.add_argument("--training-steps", type=int, default=40000)
    parser.add_argument("--warmup-steps", type=int, default=1000)
    parser.add_argument("--eval-batches", type=int, default=32)
    parser.add_argument("--log-interval", type=int, default=200)
    parser.add_argument("--eval-interval", type=int, default=500)
    parser.add_argument("--sample-interval", type=int, default=1000)
    parser.add_argument("--checkpoint-interval", type=int, default=2000)
    parser.add_argument("--sample-tokens", type=int, default=512)
    parser.add_argument("--temperature", type=float, default=0.9)
    parser.add_argument("--top-k", type=int, default=40)
    parser.add_argument("--device", default="auto")
    parser.add_argument("--no-plot-curves", action="store_true")
    return parser.parse_args()


def build_training_config(args: argparse.Namespace) -> TrainingConfig:
    return TrainingConfig(
        corpus_path=args.corpus_path,
        output_dir=args.output_dir,
        run_name=args.run_name,
        seed=args.seed,
        batch_size=args.batch_size,
        learning_rate=args.learning_rate,
        weight_decay=args.weight_decay,
        training_steps=args.training_steps,
        warmup_steps=args.warmup_steps,
        eval_batches=args.eval_batches,
        log_interval=args.log_interval,
        eval_interval=args.eval_interval,
        sample_interval=args.sample_interval,
        checkpoint_interval=args.checkpoint_interval,
        sample_tokens=args.sample_tokens,
        temperature=args.temperature,
        top_k=args.top_k,
        device=args.device,
        plot_curves=not args.no_plot_curves,
    )


def build_model_config(args: argparse.Namespace) -> GPTConfig:
    return GPTConfig(
        vocab_size=0,
        context_length=args.context_length,
        n_layers=args.n_layers,
        d_model=args.d_model,
        n_heads=args.n_heads,
        dropout=args.dropout,
        mlp_ratio=args.mlp_ratio,
        attention_backend=args.attention_backend,
        position_embedding=args.position_embedding,
    )


def main() -> None:
    args = parse_args()
    train_config = build_training_config(args)
    model_config = build_model_config(args)
    run_dir = train_one_run(train_config, model_config)
    print(f"Training finished. Outputs saved to {run_dir}")


if __name__ == "__main__":
    main()
