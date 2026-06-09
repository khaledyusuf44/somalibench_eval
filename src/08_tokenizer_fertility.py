"""Reviewer follow-up -- aggregate tokenizer fertility analysis.

This script compares how the subject-model tokenizers segment English and
Somali prompts. It supports the paper's mechanism discussion without writing
prompt text or model generations to public outputs.

Optional dependency:
    pip install "transformers>=4.41" sentencepiece

Reads:
  configs/eval_config.yaml
  SomaliBench v0 local prompt JSONL files
  data/benign_prompts/benign_so.jsonl

Writes public-safe aggregates:
  data/results/tokenizer_fertility_summary.csv
  data/results/tokenizer_fertility_summary.json

Writes local-only/gitignored diagnostics when tokenizers fail to load:
  data/tokenizer_errors/tokenizer_fertility_errors.json

Run:
    python3 src/08_tokenizer_fertility.py
    python3 src/08_tokenizer_fertility.py --include safety
    python3 src/08_tokenizer_fertility.py --models gemma2-9b-instruct qwen2.5-7b-instruct
"""

from __future__ import annotations

import argparse
import csv
import json
import os
import re
import sys
from collections import defaultdict
from pathlib import Path
from statistics import mean

import yaml

ROOT = Path(__file__).resolve().parent.parent
CONFIG_PATH = ROOT / "configs" / "eval_config.yaml"
BENIGN_SO_PATH = ROOT / "data" / "benign_prompts" / "benign_so.jsonl"
RESULTS_DIR = ROOT / "data" / "results"
ERROR_DIR = ROOT / "data" / "tokenizer_errors"

TOKENIZER_REPOS = {
    "llama3.1-8b-instruct": "meta-llama/Llama-3.1-8B-Instruct",
    "gemma2-9b-instruct": "google/gemma-2-9b-it",
    "qwen2.5-7b-instruct": "Qwen/Qwen2.5-7B-Instruct",
    "aya-23-8b": "CohereForAI/aya-23-8B",
}

WORD_RE = re.compile(r"\S+")


def load_config() -> dict:
    with CONFIG_PATH.open(encoding="utf-8") as f:
        return yaml.safe_load(f)


def resolve_repo_path(value: str) -> Path:
    path = Path(value)
    if path.is_absolute():
        return path
    return (ROOT / path).resolve()


def load_jsonl(path: Path, source: str, lang: str) -> list[dict]:
    if not path.exists():
        raise SystemExit(f"Missing input file: {path}")
    rows: list[dict] = []
    with path.open(encoding="utf-8") as f:
        for line in f:
            if not line.strip():
                continue
            rec = json.loads(line)
            rows.append({
                "id": rec.get("probe_id") or rec.get("prompt_id") or "",
                "source": source,
                "lang": lang,
                "text": rec["text"],
            })
    return rows


def load_datasets(include: str) -> list[dict]:
    cfg = load_config()
    dataset = cfg["dataset"]
    rows: list[dict] = []

    if include in {"all", "safety"}:
        rows.extend(load_jsonl(
            resolve_repo_path(dataset["local_en_jsonl"]),
            source="safety",
            lang="en",
        ))
        rows.extend(load_jsonl(
            resolve_repo_path(dataset["local_so_jsonl"]),
            source="safety",
            lang="so",
        ))

    if include in {"all", "benign"}:
        rows.extend(load_jsonl(BENIGN_SO_PATH, source="benign_control", lang="so"))

    return rows


def parse_custom_tokenizers(values: list[str] | None) -> dict[str, str]:
    mapping: dict[str, str] = {}
    for value in values or []:
        if "=" not in value:
            raise SystemExit("--tokenizer must use model_id=hf_repo format")
        model_id, repo = value.split("=", 1)
        mapping[model_id.strip()] = repo.strip()
    return mapping


def token_count(tokenizer, text: str) -> int:
    return len(tokenizer.encode(text, add_special_tokens=False))


def safe_div(num: float, den: float) -> float | None:
    if den == 0:
        return None
    return round(num / den, 4)


def percentile(values: list[int], p: float) -> float | None:
    if not values:
        return None
    ordered = sorted(values)
    idx = min(len(ordered) - 1, round((len(ordered) - 1) * p))
    return float(ordered[idx])


def words(text: str) -> list[str]:
    return WORD_RE.findall(text)


def summarize_group(tokenizer, records: list[dict], model_id: str,
                    tokenizer_repo: str) -> dict:
    n_texts = len(records)
    word_lengths: list[int] = []
    total_tokens = 0
    total_chars = 0
    total_words = 0
    unk_tokens = 0
    unk_id = getattr(tokenizer, "unk_token_id", None)

    for rec in records:
        text = rec["text"]
        token_ids = tokenizer.encode(text, add_special_tokens=False)
        total_tokens += len(token_ids)
        total_chars += len(text)
        if unk_id is not None:
            unk_tokens += sum(1 for tok_id in token_ids if tok_id == unk_id)

        text_words = words(text)
        total_words += len(text_words)
        for word in text_words:
            word_lengths.append(token_count(tokenizer, word))

    split_words = sum(1 for n in word_lengths if n > 1)
    heavily_split_words = sum(1 for n in word_lengths if n >= 3)

    return {
        "model_id": model_id,
        "tokenizer_repo": tokenizer_repo,
        "source": records[0]["source"],
        "lang": records[0]["lang"],
        "n_texts": n_texts,
        "n_words": total_words,
        "n_chars": total_chars,
        "n_tokens": total_tokens,
        "tokens_per_word": safe_div(total_tokens, total_words),
        "chars_per_token": safe_div(total_chars, total_tokens),
        "mean_word_pieces": round(mean(word_lengths), 4) if word_lengths else None,
        "p95_word_pieces": percentile(word_lengths, 0.95),
        "pct_words_split": safe_div(split_words, len(word_lengths)),
        "pct_words_3plus_pieces": safe_div(heavily_split_words, len(word_lengths)),
        "unk_token_rate": safe_div(unk_tokens, total_tokens),
        "vocab_size": getattr(tokenizer, "vocab_size", None),
    }


def write_outputs(rows: list[dict], errors: list[dict]) -> None:
    RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    csv_path = RESULTS_DIR / "tokenizer_fertility_summary.csv"
    json_path = RESULTS_DIR / "tokenizer_fertility_summary.json"

    if rows:
        fieldnames = list(rows[0].keys())
        with csv_path.open("w", encoding="utf-8", newline="") as f:
            writer = csv.DictWriter(f, fieldnames=fieldnames)
            writer.writeheader()
            writer.writerows(rows)
        json_path.write_text(json.dumps({
            "metric_note": (
                "Aggregate tokenizer fertility only. No prompt text or model "
                "generations are included in this file."
            ),
            "rows": rows,
        }, indent=2, ensure_ascii=False), encoding="utf-8")
        print(f"wrote {csv_path}")
        print(f"wrote {json_path}")
    else:
        print("No tokenizer summaries were produced.")

    if errors:
        ERROR_DIR.mkdir(parents=True, exist_ok=True)
        error_path = ERROR_DIR / "tokenizer_fertility_errors.json"
        error_path.write_text(json.dumps(errors, indent=2, ensure_ascii=False),
                              encoding="utf-8")
        print(f"wrote {error_path}")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__,
        formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--include", choices=["all", "safety", "benign"],
        default="all", help="which prompt sources to aggregate")
    parser.add_argument("--models", nargs="+",
        help="model ids to analyze; default is all configured models")
    parser.add_argument("--tokenizer", action="append",
        help="override or add tokenizer mapping as model_id=hf_repo")
    parser.add_argument("--local-files-only", action="store_true",
        help="load only tokenizers already present in the HF cache")
    args = parser.parse_args()

    try:
        from transformers import AutoTokenizer
    except ImportError:
        print("Missing optional dependency. Run:")
        print('  pip install "transformers>=4.41" sentencepiece')
        sys.exit(1)

    cfg = load_config()
    model_ids = [m["id"] for m in cfg["models"]]
    if args.models:
        unknown = sorted(set(args.models) - set(model_ids))
        if unknown:
            raise SystemExit(f"Unknown model ids: {', '.join(unknown)}")
        model_ids = args.models

    tokenizer_repos = {**TOKENIZER_REPOS, **parse_custom_tokenizers(args.tokenizer)}
    records = load_datasets(args.include)
    grouped: dict[tuple[str, str], list[dict]] = defaultdict(list)
    for rec in records:
        grouped[(rec["source"], rec["lang"])].append(rec)

    rows: list[dict] = []
    errors: list[dict] = []
    hf_token = os.environ.get("HF_TOKEN") or os.environ.get("HUGGINGFACE_TOKEN")

    for model_id in model_ids:
        repo = tokenizer_repos.get(model_id)
        if not repo:
            errors.append({"model_id": model_id, "error": "no tokenizer repo mapping"})
            continue
        try:
            tokenizer = AutoTokenizer.from_pretrained(
                repo,
                use_fast=True,
                trust_remote_code=True,
                token=hf_token,
                local_files_only=args.local_files_only,
            )
        except Exception as exc:
            errors.append({
                "model_id": model_id,
                "tokenizer_repo": repo,
                "error": str(exc),
            })
            continue

        for (_source, _lang), group_records in sorted(grouped.items()):
            rows.append(summarize_group(tokenizer, group_records, model_id, repo))

    write_outputs(rows, errors)


if __name__ == "__main__":
    main()
