# gpt-generating-poetry

This repository contains a small experimental study on French sonnet generation with a character-level GPT model. The project investigates how different modelling choices affect training behaviour and generated poems, with a particular focus on context length, model depth, and corpus format. The main question is not only whether the model can generate French-looking verse, but also whether it can learn formal properties of the sonnet such as lineation, stanza boundaries, the 4/4/3/3 structure, and the rhyme.

## Project overview

This repository contains:

- scripts for preparing a French sonnet corpus from two sources
- a character-level GPT implementation for training and generation
- experiments comparing different hyperparameter settings and corpus variants
- saved outputs for the main completed runs

The main goal of the project is to study how well a small character-level Transformer can learn formal properties of the sonnet, and how those results change when the training setup is modified.

## Corpus

The corpus combines two sources of French sonnets. One is a local raw collection assembled manually, mainly containing older sonnets from roughly the sixteenth to the eighteenth centuries. The other is the [Oupoco corpus](https://openhumanitiesdata.metajnl.com/articles/10.5334/johd.89) (Mélanie-Becquet, Grunspan,
Maignant, Plancq, & Poibeau, 2022) in TEI XML format, which contributes additional sonnet material from the nineteenth and early twentieth centuries.

From these sources, the project builds two corpus variants:

- a plain-text corpus, where stanza boundaries are preserved with blank lines
- a structured corpus, where poem boundaries and stanza roles are marked explicitly with tags such as `<BEGIN>`, `<Q1>`, `<Q2>`, `<T1>`, `<T2>`, and `<END>`

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

The plain-text corpus keeps stanza boundaries through blank lines. The structured corpus makes poem boundaries and stanza roles explicit, which allows direct comparison between weaker and stronger forms of structural supervision during training.

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

## References
Mélanie-Becquet, F., Grunspan, C., Maignant, M., Plancq, C., & Poibeau, T. (2022). _The Oupoco Database of French Sonnets from the 19th Century. Journal of Open Humanities Data_, 8. 

