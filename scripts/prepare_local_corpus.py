"""Prepare the manually collected French sonnet corpus.

This script cleans a local corpus assembled by hand from mainly
16th-18th century sonnet sources and exports a plain-text version
formatted as 4/4/3/3 stanzas.
"""

from pathlib import Path
import re


INPUT_FILE = Path("data/raw/local_sonnets_raw.txt")
OUTPUT_FILE = Path("data/cleaned/local_sonnets_cleaned.txt")
SHORT_LINE_MAX_LENGTH = 10


def normalize_line(line):
    line = line.lstrip("\ufeff")
    line = line.replace("\u00a0", " ")
    line = re.sub(r"\s+", " ", line)
    return line.strip()


def split_poems(text):
    poems = []
    current = []
    non_empty_lines = []

    for raw_line in text.splitlines():
        line = normalize_line(raw_line)
        if line:
            non_empty_lines.append(line)

    for line in non_empty_lines:
        if len(line) <= SHORT_LINE_MAX_LENGTH:
            if current:
                poems.append(current)
                current = []
            continue

        current.append(line)

    if current:
        poems.append(current)

    return poems


def keep_sonnets(poems):
    return [poem for poem in poems if len(poem) == 14]


def format_poem(lines):
    return "\n".join(
        lines[:4]
        + [""]
        + lines[4:8]
        + [""]
        + lines[8:11]
        + [""]
        + lines[11:14]
    )


def main():
    text = INPUT_FILE.read_text(encoding="utf-8")
    poems = split_poems(text)
    sonnets = keep_sonnets(poems)
    formatted = "\n\n\n".join(format_poem(poem) for poem in sonnets)

    OUTPUT_FILE.parent.mkdir(parents=True, exist_ok=True)
    OUTPUT_FILE.write_text(formatted.strip() + "\n", encoding="utf-8")

    print(f"Source poems: {len(poems)}")
    print(f"Poems kept: {len(sonnets)}")
    print(f"Poems dropped: {len(poems) - len(sonnets)}")
    print(f"Saved to {OUTPUT_FILE}")


if __name__ == "__main__":
    main()
