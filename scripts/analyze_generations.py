"""Generate poems from a trained model and analyse their structure."""

from __future__ import annotations

import argparse
from collections import Counter
import json
from pathlib import Path
import re
import sys

import torch

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT / "src") not in sys.path:
    sys.path.insert(0, str(ROOT / "src"))

from character_corpus import CharCorpus, CorpusConfig
from model import CharGPT, GPTConfig
from train_poetry_gpt import resolve_device


def load_checkpoint(path: Path, device: torch.device):
    checkpoint = torch.load(path, map_location=device)
    model_config = GPTConfig(**checkpoint["model_config"])
    model = CharGPT(model_config).to(device)
    model.load_state_dict(checkpoint["model_state"])
    model.eval()
    return checkpoint, model, model_config


def split_generated_poems(text: str) -> list[str]:
    return [poem.strip() for poem in text.split("\n\n\n") if poem.strip()]


def stanza_pattern(poem: str) -> tuple[int, ...]:
    stanzas = [stanza for stanza in poem.split("\n\n") if stanza.strip()]
    return tuple(len([line for line in stanza.splitlines() if line.strip()]) for stanza in stanzas)


def line_lengths(poem: str) -> list[int]:
    return [len(re.findall(r"\S+", line)) for line in poem.splitlines() if line.strip()]


def poem_stats(poems: list[str]) -> dict[str, object]:
    stanza_patterns = Counter(stanza_pattern(poem) for poem in poems)
    line_count_distribution = Counter(len([line for line in poem.splitlines() if line.strip()]) for poem in poems)
    all_word_counts = [count for poem in poems for count in line_lengths(poem)]
    return {
        "poem_count": len(poems),
        "line_count_distribution": dict(line_count_distribution),
        "stanza_pattern_distribution": {str(key): value for key, value in stanza_patterns.items()},
        "average_words_per_line": sum(all_word_counts) / max(1, len(all_word_counts)),
        "share_with_14_lines": line_count_distribution.get(14, 0) / max(1, len(poems)),
        "share_with_4433_pattern": stanza_patterns.get((4, 4, 3, 3), 0) / max(1, len(poems)),
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--checkpoint", required=True)
    parser.add_argument("--corpus-path", default="data/cleaned/final_sonnets_corpus.txt")
    parser.add_argument("--output-path", default="outputs/generated_poems.txt")
    parser.add_argument("--stats-path", default="outputs/generated_analysis.json")
    parser.add_argument("--num-poems", type=int, default=20)
    parser.add_argument("--max-new-tokens", type=int, default=1024)
    parser.add_argument("--temperature", type=float, default=0.9)
    parser.add_argument("--top-k", type=int, default=40)
    parser.add_argument("--device", default="auto")
    parser.add_argument("--prompt", default=None)
    parser.add_argument("--empty-prompt", action="store_true")
    parser.add_argument("--stop-at-end", action="store_true")
    args = parser.parse_args()

    device = resolve_device(args.device)
    checkpoint, model, _ = load_checkpoint(Path(args.checkpoint), device)
    corpus = CharCorpus(CorpusConfig(path=args.corpus_path))

    if args.empty_prompt:
        prompt = ""
    elif args.prompt is not None:
        prompt = args.prompt
    else:
        prompt = corpus.sample_prompt()

    if prompt:
        prompt_ids = corpus.encode(prompt).unsqueeze(0).to(device)
    else:
        prompt_ids = torch.zeros((1, 1), dtype=torch.long, device=device)

    generated_poems = []
    while len(generated_poems) < args.num_poems:
        with torch.no_grad():
            generated = model.generate(
                prompt_ids.clone(),
                max_new_tokens=args.max_new_tokens,
                temperature=args.temperature,
                top_k=args.top_k,
            )
        text = corpus.decode(generated[0].cpu())

        if args.stop_at_end:
            end_pos = text.find("<END>")
            blank_pos = text.find("\n\n\n")

            candidates = [pos for pos in (end_pos, blank_pos) if pos != -1]
            if candidates:
                cut_pos = min(candidates)

                if end_pos != -1 and cut_pos == end_pos:
                    text = text[:cut_pos] + "<END>"
                else:
                    text = text[:cut_pos] + "\n\n\n"


        poems = split_generated_poems(text)

        if poems:
            generated_poems.extend(poems)

    generated_poems = generated_poems[: args.num_poems]
    output_path = Path(args.output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text("\n\n\n".join(generated_poems) + "\n", encoding="utf-8")

    training_poems = split_generated_poems(Path(args.corpus_path).read_text(encoding="utf-8"))
    stats = {
        "checkpoint": str(args.checkpoint),
        "prompt": prompt,
        "training_corpus": poem_stats(training_poems),
        "generated_corpus": poem_stats(generated_poems),
        "generation_settings": {
            "temperature": args.temperature,
            "top_k": args.top_k,
            "max_new_tokens": args.max_new_tokens,
        },
    }
    Path(args.stats_path).write_text(json.dumps(stats, indent=2), encoding="utf-8")

    print(f"Generated poems saved to {output_path}")
    print(f"Analysis saved to {args.stats_path}")


if __name__ == "__main__":
    main()
