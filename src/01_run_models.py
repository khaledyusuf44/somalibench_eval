"""Phase 1 — Run each open-weight model against the SomaliBench v0 prompts via Ollama.

Reads:
  configs/eval_config.yaml          — models, paths, inference settings
  <dataset>/probes_en.jsonl         — English prompts (100)
  <dataset>/probes_so.jsonl         — Somali prompts  (100)

Writes:
  data/responses/<model_id>.jsonl   — one line per (probe_id, lang, response)

Resume-safe: re-running skips (probe_id, model) pairs already saved.

Run:
    python src/01_run_models.py                     # all models, all prompts
    python src/01_run_models.py --model aya-23-8b   # one model only
    python src/01_run_models.py --limit 5           # smoke test (5 prompts/model)
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


def load_config() -> dict:
    with CONFIG_PATH.open(encoding="utf-8") as f:
        return yaml.safe_load(f)


def load_prompts(cfg: dict) -> list[dict]:
    """Return a flat list of {probe_id, category, source, lang, text} rows."""
    ds_cfg = cfg["dataset"]
    rows: list[dict] = []
    if ds_cfg.get("prefer_local", True):
        en_path = (ROOT / ds_cfg["local_en_jsonl"]).resolve()
        so_path = (ROOT / ds_cfg["local_so_jsonl"]).resolve()
        for path in (en_path, so_path):
            if not path.exists():
                print(f"ERROR: prompt file not found: {path}")
                sys.exit(1)
            with path.open(encoding="utf-8") as f:
                for line in f:
                    if line.strip():
                        rows.append(json.loads(line))
    else:
        # Fallback: load from Hugging Face hub
        from datasets import load_dataset
        ds = load_dataset(ds_cfg["hf_repo"], split="train")
        for row in ds:
            rows.append(dict(row))
    return rows


def already_done(out_path: Path) -> set[tuple[str, str]]:
    """Return the set of (probe_id, lang) pairs already saved in out_path."""
    done: set[tuple[str, str]] = set()
    if not out_path.exists():
        return done
    with out_path.open(encoding="utf-8") as f:
        for line in f:
            if line.strip():
                rec = json.loads(line)
                done.add((rec["probe_id"], rec["lang"]))
    return done


def run_one(client, model_tag: str, system_prompt: str, user_prompt: str,
            temperature: float, max_tokens: int, seed: int) -> dict:
    """Send a single prompt to an Ollama model and return a structured response.

    Returns: {response, latency_s, error}.
    Raises nothing — errors are captured in the dict so the loop is resume-safe.
    """
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
        )
        text = resp["message"]["content"]
        return {"response": text, "latency_s": round(time.time() - start, 2),
                "error": None}
    except Exception as e:
        return {"response": "", "latency_s": round(time.time() - start, 2),
                "error": str(e)}


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__,
        formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--model", help="run only this model id (from config)")
    parser.add_argument("--limit", type=int,
        help="cap prompts per model (smoke testing)")
    args = parser.parse_args()

    cfg = load_config()

    try:
        import ollama
    except ImportError:
        print("`ollama` package not installed. pip install -r requirements.txt")
        sys.exit(1)

    client = ollama.Client()
    prompts = load_prompts(cfg)
    print(f"Loaded {len(prompts)} prompts "
          f"({sum(1 for r in prompts if r['lang']=='en')} en, "
          f"{sum(1 for r in prompts if r['lang']=='so')} so)")

    models = cfg["models"]
    if args.model:
        models = [m for m in models if m["id"] == args.model]
        if not models:
            print(f"ERROR: no model with id={args.model} in config")
            sys.exit(1)

    inf = cfg["inference"]
    out_dir = ROOT / cfg["paths"]["responses_dir"]
    out_dir.mkdir(parents=True, exist_ok=True)

    for model in models:
        model_id = model["id"]
        tag = model["ollama_tag"]
        out_path = out_dir / f"{model_id}.jsonl"

        done = already_done(out_path) if inf.get("resume", True) else set()
        todo = [r for r in prompts if (r["probe_id"], r["lang"]) not in done]
        if args.limit:
            todo = todo[: args.limit]

        if not todo:
            print(f"\n[{model_id}] nothing to do (all {len(prompts)} already saved)")
            continue

        print(f"\n[{model_id}] {len(todo)} prompts queued (resume skipped {len(done)})")

        with out_path.open("a", encoding="utf-8") as f:
            for row in tqdm(todo, desc=model_id, ncols=80):
                result = run_one(
                    client,
                    model_tag=tag,
                    system_prompt=inf["system_prompt"],
                    user_prompt=row["text"],
                    temperature=inf["temperature"],
                    max_tokens=inf["max_tokens"],
                    seed=inf["seed"],
                )
                rec = {
                    "probe_id": row["probe_id"],
                    "category": row["category"],
                    "source": row.get("source", ""),
                    "lang": row["lang"],
                    "prompt_text": row["text"],
                    "model_id": model_id,
                    "ollama_tag": tag,
                    **result,
                }
                f.write(json.dumps(rec, ensure_ascii=False) + "\n")
                f.flush()

        print(f"[{model_id}] wrote {out_path}")

    print("\nDone. Next: python src/02_judge_responses.py")


if __name__ == "__main__":
    main()
