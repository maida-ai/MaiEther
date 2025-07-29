#!/usr/bin/env python3

"""Cleans up characters that LLMs might generate, but have more common alternatives."""

import sys

PASS = 0
FAIL = 1


def get_replace_map() -> dict[str, str]:
    mapping = {
        # Arrows
        "↔": "<->",
        "→": "->",
        "←": "<-",
        # Smart quotes
        "“": '"',
        "”": '"',
        "‘": "'",
        "’": "'",
        # Smart dashes
        "‑": "-",
        "—": "--",
        "–": "--",
        "―": "--",
        # Smart and blank spaces
        " ": " ",
        " ": " ",
        " ": " ",
        # Smart punctuation
        "…": "...",
        "·": ".",
        # Latexable symbols
        "≤": r"$\leq$",
        "≥": r"$\geq$",
        "≠": r"$\neq$",
        "≡": r"$\equiv$",
        "≈": r"$\approx$",
        "∈": r"$\in$",
        "∉": r"$\notin$",
        "∪": r"$\cup$",
        "∩": r"$\cap$",
        "∅": r"$\emptyset$",
        "∀": r"$\forall$",
    }
    return mapping


def detect_chars(text: str, mapping: dict[str, str]) -> list[str]:
    return [char for char in text if char in mapping]


def replace_chars(text: str, mapping: dict[str, str]) -> str:
    for char, replacement in mapping.items():
        text = text.replace(char, replacement)
    return text


def process_file(file_path: str, replace: bool = True) -> int:
    with open(file_path, encoding="utf-8") as file:
        text = file.read()
        old_hash = hash(text)
    if replace:
        new_text = replace_chars(text, get_replace_map())
        new_hash = hash(new_text)
        if new_hash != old_hash:
            with open(file_path, "w", encoding="utf-8") as file:
                file.write(new_text)
            return FAIL
    else:
        chars = detect_chars(text, get_replace_map())
        if chars:
            print(f"Found chars: '{chars}' in {file_path}")
            return FAIL
    return PASS


def main() -> int:
    args = sys.argv[1:]
    modified_files = []
    for file_path in args:
        if process_file(file_path, replace=True) == FAIL:
            modified_files.append(file_path)
    if modified_files:
        print(f"Modified files: {modified_files}")
        return FAIL
    return PASS


if __name__ == "__main__":
    raise SystemExit(main())
