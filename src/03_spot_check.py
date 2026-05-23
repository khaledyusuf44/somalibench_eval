"""Phase 3 — Native-author spot check.

Random-samples N (default 80) judge classifications, stratified across the 7 safety
categories, and writes a CSV that the native author fills in manually. Re-importing
the filled CSV yields the judge-vs-human agreement number (paper's inter-judge
reliability metric).

Reads:
  data/classifications/<model>.jsonl   — Phase-2 judge labels

Writes:
  data/spot_checks/sample_for_review.csv   — empty `human_label` column for you to fill
  data/spot_checks/agreement.json          — after you fill the CSV and re-run with --score

Run:
    python src/03_spot_check.py                  # generate the sample CSV
    python src/03_spot_check.py --score          # after you fill it, compute agreement
"""

from __future__ import annotations

import argparse
import csv
import json
import random
import sys
from collections import defaultdict
from pathlib import Path

import yaml

ROOT = Path(__file__).resolve().parent.parent
CONFIG_PATH = ROOT / "configs" / "eval_config.yaml"


def load_config() -> dict:
    with CONFIG_PATH.open(encoding="utf-8") as f:
        return yaml.safe_load(f)


def gather_labels(cfg: dict) -> list[dict]:
    """Load every classification from every model into a flat list."""
    in_dir = ROOT / cfg["paths"]["classifications_dir"]
    rows: list[dict] = []
    for path in sorted(in_dir.glob("*.jsonl")):
        with path.open(encoding="utf-8") as f:
            for line in f:
                if line.strip():
                    rows.append(json.loads(line))
    return rows


def stratified_sample(rows: list[dict], n: int, seed: int = 0) -> list[dict]:
    """Stratified random sample by category — roughly n/7 per category."""
    random.seed(seed)
    by_cat: dict[str, list[dict]] = defaultdict(list)
    for r in rows:
        by_cat[r.get("category", "unknown")].append(r)

    n_cats = max(len(by_cat), 1)
    per_cat = max(1, n // n_cats)
    picked: list[dict] = []
    for cat, rs in by_cat.items():
        random.shuffle(rs)
        picked.extend(rs[:per_cat])

    # Top-up with extra random picks if we're under n
    pool = [r for r in rows if r not in picked]
    random.shuffle(pool)
    while len(picked) < n and pool:
        picked.append(pool.pop())
    return picked[:n]


def write_sample_csv(rows: list[dict], out_path: Path) -> None:
    out_path.parent.mkdir(parents=True, exist_ok=True)
    cols = ["probe_id", "category", "lang", "model_id",
            "judge_label", "human_label", "notes"]
    with out_path.open("w", encoding="utf-8", newline="") as f:
        w = csv.DictWriter(f, fieldnames=cols)
        w.writeheader()
        for r in rows:
            w.writerow({
                "probe_id":    r["probe_id"],
                "category":    r.get("category", ""),
                "lang":        r["lang"],
                "model_id":    r["model_id"],
                "judge_label": r["label"],
                "human_label": "",   # YOU FILL THIS
                "notes":       "",
            })


def score(csv_path: Path, out_json: Path) -> None:
    """Compute judge-vs-human agreement from the filled-in CSV."""
    with csv_path.open(encoding="utf-8") as f:
        rows = list(csv.DictReader(f))

    labeled = [r for r in rows if (r.get("human_label") or "").strip()]
    if not labeled:
        print("No human_label values filled in yet. Open the CSV and label them, "
              "then re-run with --score.")
        sys.exit(1)

    total = len(labeled)
    agree = sum(1 for r in labeled
                if (r["human_label"] or "").strip().lower()
                == (r["judge_label"] or "").strip().lower())

    # Per-class breakdown
    by_class: dict[str, dict[str, int]] = defaultdict(lambda: {"agree": 0, "total": 0})
    for r in labeled:
        h = (r["human_label"] or "").strip().lower()
        j = (r["judge_label"] or "").strip().lower()
        by_class[h]["total"] += 1
        if h == j:
            by_class[h]["agree"] += 1

    result = {
        "n_labeled":            total,
        "overall_agreement":    round(agree / total, 4),
        "per_class":            {k: {**v,
                                     "agreement": round(v["agree"]/v["total"], 4)
                                                  if v["total"] else None}
                                 for k, v in by_class.items()},
    }
    out_json.parent.mkdir(parents=True, exist_ok=True)
    with out_json.open("w", encoding="utf-8") as f:
        json.dump(result, f, indent=2, ensure_ascii=False)

    print(f"\nJudge vs human spot-check ({total} rows):")
    print(f"  overall agreement: {result['overall_agreement']:.2%}")
    print(f"  per-class:")
    for cls, stats in result["per_class"].items():
        print(f"    {cls:<10} {stats['agree']}/{stats['total']} "
              f"({stats['agreement']:.0%})" if stats['total'] else f"    {cls}: n/a")
    print(f"\nwrote {out_json}")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__,
        formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--score", action="store_true",
        help="after you've filled the CSV, compute judge-vs-human agreement")
    args = parser.parse_args()

    cfg = load_config()
    sc_cfg = cfg["spot_check"]
    csv_path = ROOT / sc_cfg["output_csv"]

    if args.score:
        out_json = csv_path.parent / "agreement.json"
        score(csv_path, out_json)
        return

    if csv_path.exists():
        print(f"WARNING: {csv_path} already exists. Move/delete it or it will be "
              f"overwritten.")
        sys.exit(1)

    rows = gather_labels(cfg)
    if not rows:
        print("ERROR: no classifications found in "
              f"{cfg['paths']['classifications_dir']}")
        print("Run `python src/02_judge_responses.py` first.")
        sys.exit(1)

    sample = stratified_sample(rows, n=sc_cfg["n_samples"],
                                seed=sc_cfg["random_seed"])
    write_sample_csv(sample, csv_path)

    print(f"Wrote {len(sample)} rows to {csv_path}")
    print("Open it in a spreadsheet, fill the `human_label` column with one of "
          "{refused, complied, unclear}, save, then re-run with --score.")


if __name__ == "__main__":
    main()
