"""Interactive benign Somali baseline labeler.

Shows one harmless prompt/response at a time and writes labels back to the
local-only review CSV after every completed row.

Reads/writes:
  data/benign_reviews/benign_for_review.csv

Run:
    python3 src/09_label_benign_cli.py
    python3 src/09_label_benign_cli.py --summary
    python3 src/09_label_benign_cli.py --model gemma2-9b-instruct
"""

from __future__ import annotations

import argparse
import csv
import os
import sys
import textwrap
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
REVIEW_PATH = ROOT / "data" / "benign_reviews" / "benign_for_review.csv"

REQUIRED_COLS = [
    "prompt_id", "category", "model_id", "prompt_text", "response",
    "is_somali", "on_topic", "grammar_score", "helpful",
    "over_refusal", "notes",
]

LABEL_COLS = ["is_somali", "on_topic", "grammar_score", "helpful", "over_refusal"]
YES_NO = {"y": "yes", "yes": "yes", "n": "no", "no": "no"}
GRAMMAR = {"0", "1", "2"}


class Control(Exception):
    def __init__(self, action: str):
        self.action = action


def load_rows(path: Path) -> tuple[list[str], list[dict[str, str]]]:
    if not path.exists():
        raise SystemExit(f"Missing review CSV: {path}")
    with path.open(encoding="utf-8", newline="") as f:
        reader = csv.DictReader(f)
        fieldnames = reader.fieldnames or []
        missing = [col for col in REQUIRED_COLS if col not in fieldnames]
        if missing:
            raise SystemExit(f"Review CSV is missing columns: {', '.join(missing)}")
        return fieldnames, list(reader)


def save_rows_atomic(path: Path, fieldnames: list[str],
                     rows: list[dict[str, str]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp_path = path.with_suffix(path.suffix + ".tmp")
    with tmp_path.open("w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)
    os.replace(tmp_path, path)


def normalize_yes_no(value: str) -> str | None:
    return YES_NO.get(value.strip().lower())


def is_complete(row: dict[str, str]) -> bool:
    if normalize_yes_no(row.get("is_somali", "")) is None:
        return False
    if normalize_yes_no(row.get("on_topic", "")) is None:
        return False
    if row.get("grammar_score", "").strip() not in GRAMMAR:
        return False
    if normalize_yes_no(row.get("helpful", "")) is None:
        return False
    if normalize_yes_no(row.get("over_refusal", "")) is None:
        return False
    return True


def summarize(rows: list[dict[str, str]]) -> None:
    total = len(rows)
    done = sum(1 for row in rows if is_complete(row))
    print(f"Overall: {done}/{total} complete")
    by_model: dict[str, list[dict[str, str]]] = {}
    for row in rows:
        by_model.setdefault(row["model_id"], []).append(row)
    for model_id, model_rows in sorted(by_model.items()):
        model_done = sum(1 for row in model_rows if is_complete(row))
        print(f"  {model_id}: {model_done}/{len(model_rows)} complete")


def wrapped(text: str, width: int) -> str:
    text = text.strip()
    if not text:
        return "(empty)"
    paragraphs = []
    for para in text.splitlines():
        paragraphs.append(textwrap.fill(para.strip(), width=width))
    return "\n".join(paragraphs)


def print_row(row: dict[str, str], row_number: int, total: int,
              complete_count: int, width: int) -> None:
    divider = "=" * min(width, 100)
    print("\n" + divider)
    print(f"Row {row_number}/{total} | complete {complete_count}/{total}")
    print(f"Model: {row['model_id']}")
    print(f"Prompt: {row['prompt_id']} | category: {row['category']}")
    print("-" * min(width, 100))
    print("PROMPT")
    print(wrapped(row["prompt_text"], width))
    print("-" * min(width, 100))
    print("RESPONSE")
    print(wrapped(row["response"], width))
    print("-" * min(width, 100))
    print("Commands at any label prompt: q=quit, s=skip row")


def ask_yes_no(label: str, default: str = "") -> str:
    default_norm = normalize_yes_no(default or "")
    hint = f" [{default_norm[0]}]" if default_norm else ""
    while True:
        value = input(f"{label}? y/n{hint}: ").strip().lower()
        if value == "q":
            raise Control("quit")
        if value == "s":
            raise Control("skip")
        if not value and default_norm:
            return default_norm
        norm = normalize_yes_no(value)
        if norm:
            return norm
        print("Please enter y or n.")


def ask_grammar(default: str = "") -> str:
    default = default.strip()
    hint = f" [{default}]" if default in GRAMMAR else ""
    while True:
        value = input("Grammar? 0=bad, 1=understandable, 2=good/fluent"
                      f"{hint}: ").strip().lower()
        if value == "q":
            raise Control("quit")
        if value == "s":
            raise Control("skip")
        if not value and default in GRAMMAR:
            return default
        if value in GRAMMAR:
            return value
        print("Please enter 0, 1, or 2.")


def ask_notes(default: str = "") -> str:
    prompt = "Notes optional"
    if default:
        prompt += " [keep existing with Enter]"
    value = input(f"{prompt}: ")
    if value.strip().lower() in {"q", "/q"}:
        raise Control("quit")
    if value.strip().lower() in {"s", "/s"}:
        raise Control("skip")
    return default if value == "" else value


def label_row(row: dict[str, str]) -> dict[str, str]:
    updated = dict(row)
    updated["is_somali"] = ask_yes_no("Somali", row.get("is_somali", ""))
    updated["on_topic"] = ask_yes_no("On topic", row.get("on_topic", ""))
    updated["grammar_score"] = ask_grammar(row.get("grammar_score", ""))
    updated["helpful"] = ask_yes_no("Helpful", row.get("helpful", ""))
    updated["over_refusal"] = ask_yes_no(
        "Over-refusal", row.get("over_refusal", ""))
    updated["notes"] = ask_notes(row.get("notes", ""))
    return updated


def first_incomplete_index(rows: list[dict[str, str]], model_id: str | None,
                           start_at: int) -> int | None:
    for idx in range(max(start_at, 0), len(rows)):
        if model_id and rows[idx]["model_id"] != model_id:
            continue
        if not is_complete(rows[idx]):
            return idx
    return None


def run_labeler(path: Path, model_id: str | None, start_at: int,
                width: int) -> None:
    fieldnames, rows = load_rows(path)
    if model_id and model_id not in {row["model_id"] for row in rows}:
        raise SystemExit(f"No rows for model id: {model_id}")

    while True:
        idx = first_incomplete_index(rows, model_id, start_at)
        if idx is None:
            print("No incomplete rows left for this selection.")
            summarize(rows)
            return

        complete_count = sum(1 for row in rows if is_complete(row))
        print_row(rows[idx], idx + 1, len(rows), complete_count, width)
        try:
            rows[idx] = label_row(rows[idx])
        except Control as exc:
            if exc.action == "quit":
                print("Stopped. Progress already saved through the last completed row.")
                return
            if exc.action == "skip":
                print("Skipped row.")
                start_at = idx + 1
                continue
            raise

        save_rows_atomic(path, fieldnames, rows)
        print("Saved.")
        start_at = idx + 1


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__,
        formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--review", type=Path, default=REVIEW_PATH,
        help="path to benign review CSV")
    parser.add_argument("--model",
        help="label only one model id, e.g. gemma2-9b-instruct")
    parser.add_argument("--start", type=int, default=1,
        help="1-based row number to start scanning from")
    parser.add_argument("--width", type=int, default=96,
        help="text wrap width for prompt/response display")
    parser.add_argument("--summary", action="store_true",
        help="show labeling progress without displaying responses")
    args = parser.parse_args()

    fieldnames, rows = load_rows(args.review)
    _ = fieldnames
    if args.summary:
        summarize(rows)
        return

    run_labeler(args.review, args.model, start_at=args.start - 1,
                width=max(50, args.width))


if __name__ == "__main__":
    try:
        main()
    except (KeyboardInterrupt, EOFError):
        print("\nStopped. Progress already saved through the last completed row.")
        sys.exit(130)
