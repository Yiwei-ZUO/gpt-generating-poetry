# gpt-generating-poetry

Character-level GPT experiments for French sonnet generation, developed for a lab project on large language models.

## Project overview

This repository contains:

- scripts for preparing a French sonnet corpus from two sources
- a character-level GPT implementation for training and generation
- experiments comparing plain and structured corpora
- saved outputs for the main completed runs

The main goal of the project is to study how well a small character-level Transformer can learn formal properties of the sonnet, especially lineation, stanza structure, and the 4/4/3/3 layout.

## Repository structure

```text
data/
  raw/         raw source files
  cleaned/     cleaned and merged corpora

scripts/
  prepare_local_corpus.py
  prepare_oupoco_corpus.py
  build_final_corpus.py
  build_structured_corpus.py
  train_poetry_gpt.py
  analyze_generations.py
  compare_poem_outputs.py
  run_hparam_experiments.py

src/
  model.py
  character_corpus.py

outputs/
  baseline/
  context_512/
  structured_256/
  structured_512/
  structured_256_l8/
  generated poems/
```

## Data pipeline

The corpus is built in four stages:

1. Clean the local raw sonnet collection.
2. Clean the Oupoco TEI sonnet collection.
3. Merge the two cleaned corpora into a final plain-text corpus.
4. Build a structured corpus with explicit markers such as `<BEGIN>`, `<Q1>`, `<Q2>`, `<T1>`, `<T2>`, and `<END>`.

## Environment

The project uses `pixi` for dependency management. The main dependencies are:

- Python 3.11
- NumPy
- Matplotlib
- PyTorch

Install the environment with:

```bash
pixi install
```

## Main commands

Prepare the corpora:

```bash
pixi run prepare_local_corpus
pixi run prepare_oupoco_corpus
pixi run build_final_corpus
pixi run build_structured_corpus
```

Train the default baseline model:

```bash
pixi run train_poetry_gpt
```

Train the default structured 512 model:

```bash
pixi run train_structured_poetry_gpt
```

Run generation analysis:

```bash
pixi run analyze_generations -- --checkpoint outputs/baseline/checkpoint_best.pt
```

Compare generated poem files with lightweight structural and rhyme-like statistics:

```bash
python scripts/compare_poem_outputs.py \
  --label baseline \
  --input-path "outputs/generated poems/generated_poems_baseline_50poems.txt" \
  --output-path "outputs/generated poems/compare_baseline_50.json"
```

## Saved experimental runs

The repository keeps the final artefacts for five main runs:

- `baseline`
- `context_512`
- `structured_256`
- `structured_512`
- `structured_256_l8`

For each run, the repository keeps:

- `checkpoint_best.pt`
- `config.json`
- `metrics.jsonl`
- `samples.txt`
- `training_curves.png`

Intermediate `checkpoint_step_*.pt` files are intentionally ignored to keep the repository manageable.

## Notes

- The codebase keeps executable entry points in `scripts/` and reusable modules in `src/`.
- Structured generation experiments use prompts that match the annotated corpus format.
- The rhyme analysis is approximate and based on orthographic similarity at line endings, not on full French phonology.
