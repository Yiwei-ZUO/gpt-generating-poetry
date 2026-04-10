"""Compare generated poem files with simple structural and rhyme-like statistics."""

from __future__ import annotations

import argparse
import json
from collections import Counter
from pathlib import Path
import re


WORD_RE = re.compile(r"\b[\wÀ-ÿ'-]+\b", re.UNICODE)
STRUCTURE_MARKERS = {"<BEGIN>", "<END>", "<Q1>", "<Q2>", "<T1>", "<T2>"}


def split_poems(text: str) -> list[str]:
    return [poem.strip() for poem in text.split("\n\n\n") if poem.strip()]


def split_stanzas(poem: str) -> list[list[str]]:
    raw_lines = [line.strip() for line in poem.splitlines() if line.strip()]
    if any(line in {"<Q1>", "<Q2>", "<T1>", "<T2>"} for line in raw_lines):
        stanzas: list[list[str]] = []
        current: list[str] = []
        for line in raw_lines:
            if line in {"<BEGIN>", "<END>"}:
                continue
            if line in {"<Q1>", "<Q2>", "<T1>", "<T2>"}:
                if current:
                    stanzas.append(current)
                    current = []
                continue
            current.append(line)
        if current:
            stanzas.append(current)
        return stanzas

    return [
        [line.strip() for line in stanza.splitlines() if line.strip()]
        for stanza in poem.split("\n\n")
        if stanza.strip()
    ]


def clean_line(line: str) -> str:
    return line.strip()


def line_word_count(line: str) -> int:
    return len(WORD_RE.findall(line))


def final_word(line: str) -> str:
    words = WORD_RE.findall(line.casefold())
    return words[-1] if words else ""


def rhyme_suffix(word: str, n: int = 3) -> str:
    if not word:
        return ""
    return word[-n:]


def vowel_groups(line: str) -> int:
    lowered = line.casefold()
    groups = re.findall(r"[aeiouyàâäéèêëîïôöùûüÿœæ]+", lowered)
    return len(groups)


def abba_like(lines: list[str]) -> bool:
    if len(lines) != 4:
        return False
    suffixes = [rhyme_suffix(final_word(line)) for line in lines]
    return bool(suffixes[0] and suffixes[1]) and suffixes[0] == suffixes[3] and suffixes[1] == suffixes[2]


def tercet_repetition(lines: list[str]) -> bool:
    if len(lines) != 3:
        return False
    suffixes = [rhyme_suffix(final_word(line)) for line in lines]
    counts = Counter(suffixes)
    return any(suf and count >= 2 for suf, count in counts.items())


def analyze_poems(poems: list[str]) -> dict[str, object]:
    poem_count = len(poems)
    line_counts = []
    stanza_patterns = Counter()
    words_per_line = []
    vowel_groups_per_line = []
    quatrain_abba_hits = 0
    quatrain_total = 0
    tercet_repeat_hits = 0
    tercet_total = 0
    complete_end_marker = 0
    complete_structured_poems = 0
    end_word_suffix_counts = Counter()

    for poem in poems:
        stanzas = split_stanzas(poem)
        flat_lines = [
            clean_line(line)
            for stanza in stanzas
            for line in stanza
            if clean_line(line) and clean_line(line) not in STRUCTURE_MARKERS
        ]
        stanza_patterns[tuple(len(stanza) for stanza in stanzas)] += 1
        line_counts.append(len(flat_lines))

        if "<END>" in poem:
            complete_end_marker += 1
        if poem.strip().startswith("<BEGIN>") and poem.strip().endswith("<END>"):
            complete_structured_poems += 1

        for line in flat_lines:
            words_per_line.append(line_word_count(line))
            vowel_groups_per_line.append(vowel_groups(line))
            end_word_suffix_counts[rhyme_suffix(final_word(line))] += 1

        for stanza in stanzas:
            if len(stanza) == 4:
                quatrain_total += 1
                if abba_like(stanza):
                    quatrain_abba_hits += 1
            elif len(stanza) == 3:
                tercet_total += 1
                if tercet_repetition(stanza):
                    tercet_repeat_hits += 1

    line_count_distribution = Counter(line_counts)
    return {
        "poem_count": poem_count,
        "line_count_distribution": dict(line_count_distribution),
        "share_with_14_lines": line_count_distribution.get(14, 0) / max(1, poem_count),
        "stanza_pattern_distribution": {str(k): v for k, v in stanza_patterns.items()},
        "share_with_4433_pattern": stanza_patterns.get((4, 4, 3, 3), 0) / max(1, poem_count),
        "average_words_per_line": sum(words_per_line) / max(1, len(words_per_line)),
        "average_vowel_groups_per_line": sum(vowel_groups_per_line) / max(1, len(vowel_groups_per_line)),
        "share_complete_with_end_marker": complete_end_marker / max(1, poem_count),
        "share_complete_begin_end": complete_structured_poems / max(1, poem_count),
        "quatrain_abba_like_share": quatrain_abba_hits / max(1, quatrain_total),
        "tercet_repetition_share": tercet_repeat_hits / max(1, tercet_total),
        "top_line_end_suffixes": end_word_suffix_counts.most_common(10),
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--label", required=True)
    parser.add_argument("--input-path", required=True)
    parser.add_argument("--output-path", required=True)
    args = parser.parse_args()

    text = Path(args.input_path).read_text(encoding="utf-8")
    poems = split_poems(text)
    result = {
        "label": args.label,
        "input_path": args.input_path,
        "analysis": analyze_poems(poems),
    }
    Path(args.output_path).write_text(json.dumps(result, indent=2, ensure_ascii=False), encoding="utf-8")
    print(f"Saved analysis to {args.output_path}")


if __name__ == "__main__":
    main()
