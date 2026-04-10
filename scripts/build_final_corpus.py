"""Build the final training corpus by merging and deduplicating sources.

This script combines the cleaned local corpus and the cleaned Oupoco
corpus, then removes overlaps between them, even though duplicates are
very rare in practice.
"""

from pathlib import Path
import re


OLD_DATASET = Path("data/cleaned/local_sonnets_cleaned.txt")
OUPOCO_DATASET = Path("data/cleaned/oupoco_sonnets_cleaned.txt")
OUTPUT_FILE = Path("data/cleaned/final_sonnets_corpus.txt")
TAGGED_OUTPUT_FILE = Path("data/cleaned/final_sonnets_tagged_corpus.txt")


def load_poems(path):
    text = path.read_text(encoding="utf-8")
    return [poem.strip() for poem in text.split("\n\n\n") if poem.strip()]


def normalize_poem(poem):
    lines = []
    for line in poem.splitlines():
        cleaned = re.sub(r"\s+", " ", line).strip()
        if cleaned:
            lines.append(cleaned.casefold())
    return "\n".join(lines)


def format_tagged_poem(poem):
    stanzas = [stanza.strip() for stanza in poem.split("\n\n") if stanza.strip()]
    tagged_parts = ["<BEGIN>"]
    for index, stanza in enumerate(stanzas):
        tagged_parts.append(stanza)
        if index < len(stanzas) - 1:
            tagged_parts.append("<STANZA>")
    tagged_parts.append("<END>")
    return "\n".join(tagged_parts)


def main():
    old_poems = load_poems(OLD_DATASET)
    oupoco_poems = load_poems(OUPOCO_DATASET)

    seen = {}
    duplicates_within_old = 0
    duplicates_within_oupoco = 0
    duplicates_across_sets = 0
    merged = []

    for source_name, poems in [("old", old_poems), ("oupoco", oupoco_poems)]:
        for poem in poems:
            normalized = normalize_poem(poem)
            if normalized in seen:
                if seen[normalized] == source_name:
                    if source_name == "old":
                        duplicates_within_old += 1
                    else:
                        duplicates_within_oupoco += 1
                else:
                    duplicates_across_sets += 1
                continue

            seen[normalized] = source_name
            merged.append(poem)

    OUTPUT_FILE.parent.mkdir(parents=True, exist_ok=True)
    OUTPUT_FILE.write_text("\n\n\n".join(merged).strip() + "\n", encoding="utf-8")

    tagged_merged = [format_tagged_poem(poem) for poem in merged]
    TAGGED_OUTPUT_FILE.write_text("\n\n\n".join(tagged_merged).strip() + "\n", encoding="utf-8")

    print(f"Old dataset poems: {len(old_poems)}")
    print(f"Oupoco dataset poems: {len(oupoco_poems)}")
    print(f"Duplicates within old dataset: {duplicates_within_old}")
    print(f"Duplicates within Oupoco dataset: {duplicates_within_oupoco}")
    print(f"Duplicates across datasets: {duplicates_across_sets}")
    print(f"Unique poems after merge: {len(merged)}")
    print(f"Saved plain corpus to {OUTPUT_FILE}")
    print(f"Saved tagged corpus to {TAGGED_OUTPUT_FILE}")


if __name__ == "__main__":
    main()
