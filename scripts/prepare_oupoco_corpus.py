"""Prepare the Oupoco sonnet corpus from the TEI XML source.

This script extracts sonnets from the Oupoco dataset, which mainly
covers 19th-20th century material, and exports them as plain text
with the expected 4/4/3/3 sonnet structure.
"""

from pathlib import Path
import re
import xml.etree.ElementTree as ET


INPUT_FILE = Path("data/raw/oupoco_sonnets_raw.xml")
OUTPUT_FILE = Path("data/cleaned/oupoco_sonnets_cleaned.txt")
MIN_WORDS_PER_LINE = 4


def normalize_text(text):
    text = text.replace("\u00a0", " ")
    text = re.sub(r"\s+", " ", text)
    return text.strip()


def word_count(line):
    return len(line.split())


def get_namespace(root):
    if root.tag.startswith("{"):
        return {"tei": root.tag.split("}")[0].strip("{")}
    return {}


def extract_sonnets():
    root = ET.parse(INPUT_FILE).getroot()
    ns = get_namespace(root)
    sonnets = []

    for div in root.findall('.//tei:div[@type="sonnet"]', ns):
        stanzas = []
        all_lines = []

        for lg in div.findall("./tei:lg", ns):
            stanza_lines = []
            for line in lg.findall("./tei:l", ns):
                text = normalize_text("".join(line.itertext()))
                if text:
                    stanza_lines.append(text)
                    all_lines.append(text)
            if stanza_lines:
                stanzas.append(stanza_lines)

        if (
            len(all_lines) == 14
            and [len(stanza) for stanza in stanzas] == [4, 4, 3, 3]
            and all(word_count(line) >= MIN_WORDS_PER_LINE for line in all_lines)
        ):
            sonnets.append(stanzas)

    return sonnets


def format_sonnet(stanzas):
    return "\n\n".join("\n".join(stanza) for stanza in stanzas)


def main():
    sonnets = extract_sonnets()
    output = "\n\n\n".join(format_sonnet(sonnet) for sonnet in sonnets)

    OUTPUT_FILE.parent.mkdir(parents=True, exist_ok=True)
    OUTPUT_FILE.write_text(output.strip() + "\n", encoding="utf-8")

    print(f"Sonnets kept: {len(sonnets)}")
    print(f"Minimum words per line: {MIN_WORDS_PER_LINE}")
    print(f"Saved to {OUTPUT_FILE}")


if __name__ == "__main__":
    main()
