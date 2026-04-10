"""Build a structure-supervised sonnet corpus with stanza-type labels.

This script creates an experimental corpus for a stronger sonnet-form
supervision setup. Each poem is formatted with explicit poem boundaries
and stanza-role markers: Q1, Q2, T1, and T2.
"""

from pathlib import Path


INPUT_FILE = Path("data/cleaned/final_sonnets_corpus.txt")
OUTPUT_FILE = Path("data/cleaned/final_sonnets_structured_corpus.txt")


def load_poems(path: Path) -> list[str]:
    text = path.read_text(encoding="utf-8")
    return [poem.strip() for poem in text.split("\n\n\n") if poem.strip()]


def format_structured_poem(poem: str) -> str:
    stanzas = [stanza.strip() for stanza in poem.split("\n\n") if stanza.strip()]
    if len(stanzas) != 4:
        raise ValueError("Expected each poem to contain exactly four stanzas.")

    labels = ["<Q1>", "<Q2>", "<T1>", "<T2>"]
    lines = ["<BEGIN>"]
    for label, stanza in zip(labels, stanzas, strict=True):
        lines.append(label)
        lines.extend(stanza.splitlines())
    lines.append("<END>")
    return "\n".join(lines)


def main() -> None:
    poems = load_poems(INPUT_FILE)
    structured_poems = [format_structured_poem(poem) for poem in poems]

    OUTPUT_FILE.parent.mkdir(parents=True, exist_ok=True)
    OUTPUT_FILE.write_text("\n\n\n".join(structured_poems).strip() + "\n", encoding="utf-8")

    print(f"Input poems: {len(poems)}")
    print(f"Structured poems written: {len(structured_poems)}")
    print(f"Saved structured corpus to {OUTPUT_FILE}")


if __name__ == "__main__":
    main()
