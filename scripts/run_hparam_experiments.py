"""Run a small set of hyperparameter experiments for the poetry GPT."""

from __future__ import annotations

from copy import deepcopy
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT / "src") not in sys.path:
    sys.path.insert(0, str(ROOT / "src"))

from model import GPTConfig
from train_poetry_gpt import TrainingConfig, train_one_run


def main() -> None:
    base_train = TrainingConfig(run_name="baseline", training_steps=40000)
    base_model = GPTConfig(vocab_size=0)

    experiments = [
        ("baseline", {}, {}),
        ("context_128", {"run_name": "context_128"}, {"context_length": 128}),
        ("layers_8", {"run_name": "layers_8"}, {"n_layers": 8}),
        ("masked_attention", {"run_name": "masked_attention"}, {"attention_backend": "masked"}),
    ]

    for _, train_overrides, model_overrides in experiments:
        train_config = deepcopy(base_train)
        model_config = deepcopy(base_model)

        for key, value in train_overrides.items():
            setattr(train_config, key, value)
        for key, value in model_overrides.items():
            setattr(model_config, key, value)

        train_one_run(train_config, model_config)


if __name__ == "__main__":
    main()
