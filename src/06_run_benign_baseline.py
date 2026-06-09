"""Reviewer follow-up -- run benign Somali competence prompts.

This is a non-safety control for the SomaliBench Eval paper. It asks the same
four subject models harmless Somali prompts so we can estimate whether high
Somali ``unclear'' rates on safety prompts reflect language competence, safety
conditioning, or both.

Reads:
  data/benign_prompts/benign_so.jsonl
  configs/eval_config.yaml

Writes, local-only/gitignored:
  data/benign_responses/<model_id>.jsonl

Run:
    python src/06_run_benign_baseline.py
    python src/06_run_benign_baseline.py --model gemma2-9b-instruct
    python src/06_run_benign_baseline.py --limit 5
"""

from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

import yaml
from tqdm import tqdm

ROOT = Path(__file__).resolve().parent.parent
CONFIG_PATH = ROOT / "configs" / "eval_config.yaml"
PROMPT_PATH = ROOT / "data" / "benign_prompts" / "benign_so.jsonl"
OUT_DIR = ROOT / "data" / "benign_responses"


def load_config() -> dict:
    with CONFIG_PATH.open(encoding="utf-8") as f:
        return yaml.safe_load(f)


def load_prompts() -> list[dict]:
    if not PROMPT_PATH.exists():
        raise SystemExit(f"Missing benign prompt file: {PROMPT_PATH}")
    with PROMPT_PATH.open(encoding="utf-8") as f:
        return [json.loads(line) for line in f if line.strip()]


def already_done(out_path: Path) -> set[str]:
    done: set[str] = set()
    if not out_path.exists():
        return done
    with out_path.open(encoding="utf-8") as f:
        for line in f:
            if line.strip():
                rec = json.loads(line)
                done.add(rec["prompt_id"])
    return done


def run_one(client, model_tag: str, system_prompt: str, user_prompt: str,
            temperature: float, max_tokens: int, seed: int,
            keep_alive: int | str = 0) -> dict:
    start = time.time()
    try:
        resp = client.chat(
            model=model_tag,
            messages=[
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": user_prompt},
            ],
            options={
                "temperature": temperature,
                "num_predict": max_tokens,
                "seed": seed,
            },
            keep_alive=keep_alive,
        )
        return {
            "response": resp["message"]["content"],
            "latency_s": round(time.time() - start, 2),
            "error": None,
        }
    except Exception as e:
        return {
            "response": "",
            "latency_s": round(time.time() - start, 2),
            "error": str(e),
        }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__,
        formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--model", help="run only this model id")
    parser.add_argument("--limit", type=int, help="cap prompts per model")
    args = parser.parse_args()

    cfg = load_config()
    prompts = load_prompts()

    try:
        import ollama
    except ImportError:
        print("`ollama` package not installed. pip install -r requirements.txt")
        sys.exit(1)

    models = cfg["models"]
    if args.model:
        models = [m for m in models if m["id"] == args.model]
        if not models:
            raise SystemExit(f"No model with id={args.model}")

    inf = cfg["inference"]
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    client = ollama.Client()

    print(f"Loaded {len(prompts)} benign Somali prompts")
    for model in models:
        model_id = model["id"]
        out_path = OUT_DIR / f"{model_id}.jsonl"
        done = already_done(out_path) if inf.get("resume", True) else set()
        todo = [p for p in prompts if p["prompt_id"] not in done]
        if args.limit:
            todo = todo[: args.limit]

        if not todo:
            print(f"\n[{model_id}] nothing to do ({len(done)} already saved)")
            continue

        print(f"\n[{model_id}] {len(todo)} benign prompts queued")
        with out_path.open("a", encoding="utf-8") as f:
            for row in tqdm(todo, desc=model_id, ncols=80):
                result = run_one(
                    client,
                    model_tag=model["ollama_tag"],
                    system_prompt=inf["system_prompt"],
                    user_prompt=row["text"],
                    temperature=inf["temperature"],
                    max_tokens=inf["max_tokens"],
                    seed=inf["seed"],
                    keep_alive=inf.get("keep_alive", 0),
                )
                rec = {
                    "prompt_id": row["prompt_id"],
                    "category": row["category"],
                    "lang": row["lang"],
                    "prompt_text": row["text"],
                    "model_id": model_id,
                    "ollama_tag": model["ollama_tag"],
                    **result,
                }
                f.write(json.dumps(rec, ensure_ascii=False) + "\n")
                f.flush()

        print(f"[{model_id}] wrote {out_path}")

    print("\nNext: python src/07_score_benign_baseline.py")


if __name__ == "__main__":
    main()
