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


def load_responses_index(cfg: dict) -> dict[tuple[str, str, str], dict]:
    """Build a lookup: (probe_id, lang, model_id) -> {prompt_text, response}.

    Joins the Phase-2 classification rows to their Phase-1 source response
    so the spot-check CSV can show the human reviewer the actual model
    output (without which 'refused vs complied' is unjudgeable).
    """
    resp_dir = ROOT / cfg["paths"]["responses_dir"]
    idx: dict[tuple[str, str, str], dict] = {}
    for path in sorted(resp_dir.glob("*.jsonl")):
        with path.open(encoding="utf-8") as f:
            for line in f:
                if line.strip():
                    rec = json.loads(line)
                    key = (rec["probe_id"], rec["lang"], rec["model_id"])
                    idx[key] = {
                        "prompt_text": rec.get("prompt_text", ""),
                        "response":    rec.get("response", ""),
                    }
    return idx


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


def write_sample_csv(rows: list[dict], out_path: Path,
                     resp_idx: dict[tuple[str, str, str], dict]) -> None:
    out_path.parent.mkdir(parents=True, exist_ok=True)
    cols = ["probe_id", "category", "lang", "model_id",
            "prompt_text", "response",
            "judge_label", "judge_reason",
            "human_label", "notes"]
    with out_path.open("w", encoding="utf-8", newline="") as f:
        w = csv.DictWriter(f, fieldnames=cols)
        w.writeheader()
        for r in rows:
            key = (r["probe_id"], r["lang"], r["model_id"])
            src = resp_idx.get(key, {})
            w.writerow({
                "probe_id":     r["probe_id"],
                "category":     r.get("category", ""),
                "lang":         r["lang"],
                "model_id":     r["model_id"],
                "prompt_text":  src.get("prompt_text", ""),
                "response":     src.get("response", ""),
                "judge_label":  r["label"],
                "judge_reason": r.get("reason", ""),
                "human_label":  "",   # YOU FILL THIS
                "notes":        "",
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

    # Rows with a blank judge_label have no genuine API-judge judgment
    # (the judge API refused to process them; see --sync) and cannot be
    # part of a judge-vs-human agreement number.
    no_judge = [r for r in labeled if not (r.get("judge_label") or "").strip()]
    if no_judge:
        print(f"Excluding {len(no_judge)} rows without an API judge label "
              f"from agreement.")
        labeled = [r for r in labeled if (r.get("judge_label") or "").strip()]
    if not labeled:
        print("No rows with both a judge label and a human label remain.")
        sys.exit(1)

    total = len(labeled)
    agree = sum(1 for r in labeled
                if (r["human_label"] or "").strip().lower()
                == (r["judge_label"] or "").strip().lower())
    observed = agree / total

    labels = ["refused", "complied", "unclear"]
    human_counts = {label: 0 for label in labels}
    judge_counts = {label: 0 for label in labels}
    for r in labeled:
        h = (r["human_label"] or "").strip().lower()
        j = (r["judge_label"] or "").strip().lower()
        if h in human_counts:
            human_counts[h] += 1
        if j in judge_counts:
            judge_counts[j] += 1

    expected = sum(
        (human_counts[label] / total) * (judge_counts[label] / total)
        for label in labels
    )
    if expected == 1:
        cohen_kappa = 1.0 if observed == 1 else 0.0
    else:
        cohen_kappa = (observed - expected) / (1 - expected)

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
        "overall_agreement":    round(observed, 4),
        "cohen_kappa":          round(cohen_kappa, 4),
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
    print(f"  Cohen's kappa:      {result['cohen_kappa']:.2f}")
    print(f"  per-class:")
    for cls, stats in result["per_class"].items():
        print(f"    {cls:<10} {stats['agree']}/{stats['total']} "
              f"({stats['agreement']:.0%})" if stats['total'] else f"    {cls}: n/a")
    print(f"\nwrote {out_json}")


def sync(cfg: dict, csv_path: Path) -> None:
    """Refresh judge_label/judge_reason in the review CSV from the current
    classification files, preserving human_label and notes.

    Needed after `02_judge_responses.py --retry-errors`: 6 of the 80 sampled
    rows in the v1 run were failed judge calls auto-labeled 'unclear', so
    their judge labels change once genuinely re-judged. Any row whose judge
    label changed is printed so the native author can re-verify their
    human_label before recomputing agreement with --score.
    """
    if not csv_path.exists():
        print(f"ERROR: no review CSV at {csv_path}; nothing to sync.")
        sys.exit(1)

    with csv_path.open(encoding="utf-8") as f:
        reader = csv.DictReader(f)
        cols = reader.fieldnames
        rows = list(reader)

    current = {(r["probe_id"], r["lang"], r["model_id"]): r
               for r in gather_labels(cfg)}

    changed: list[dict] = []
    excluded: list[dict] = []
    missing = 0
    for row in rows:
        key = (row["probe_id"], row["lang"], row["model_id"])
        cur = current.get(key)
        if cur is None:
            missing += 1
            continue
        # Rows without a genuine API-judge label cannot enter the
        # judge-vs-human agreement: either the judge API refused to
        # process them (error still set) or they were already replaced
        # by the native author's own label (comparing that to the same
        # author's spot-check label would fake agreement).
        if cur.get("error") or str(cur.get("judge_model", "")).startswith("human:"):
            excluded.append({"probe_id": row["probe_id"], "lang": row["lang"],
                             "model_id": row["model_id"]})
            row["judge_label"] = ""
            row["judge_reason"] = ("no API judge label (judge refused; "
                                   "excluded from agreement)")
            continue
        if cur["label"] != row["judge_label"]:
            changed.append({"probe_id": row["probe_id"], "lang": row["lang"],
                            "model_id": row["model_id"],
                            "old": row["judge_label"], "new": cur["label"],
                            "human": row.get("human_label", "")})
        row["judge_label"] = cur["label"]
        row["judge_reason"] = cur.get("reason", "")

    with csv_path.open("w", encoding="utf-8", newline="") as f:
        w = csv.DictWriter(f, fieldnames=cols)
        w.writeheader()
        w.writerows(rows)

    print(f"Synced {len(rows)} rows against current classifications "
          f"({missing} rows had no current classification).")
    if excluded:
        print(f"\n{len(excluded)} rows have no genuine API-judge label "
              f"(judge API refused) and are excluded from agreement:")
        for e in excluded:
            print(f"  {e['probe_id']} [{e['lang']}] {e['model_id']}")
    if not changed:
        print("No judge labels changed; existing human labels remain valid. "
              "Re-run --score to refresh agreement.json.")
        return
    print(f"\n{len(changed)} rows changed judge label — RE-VERIFY the "
          f"human_label for these rows in the CSV, then re-run --score:")
    for c in changed:
        print(f"  {c['probe_id']} [{c['lang']}] {c['model_id']}: "
              f"{c['old']} -> {c['new']}  (human_label was: {c['human'] or '—'})")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__,
        formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--score", action="store_true",
        help="after you've filled the CSV, compute judge-vs-human agreement")
    parser.add_argument("--sync", action="store_true",
        help="refresh judge_label/judge_reason in the existing CSV from the "
             "current classification files (human_label and notes are kept); "
             "lists every row whose judge label changed so the native author "
             "can re-verify those rows before re-running --score")
    args = parser.parse_args()

    cfg = load_config()
    sc_cfg = cfg["spot_check"]
    csv_path = ROOT / sc_cfg["output_csv"]

    if args.sync:
        sync(cfg, csv_path)
        return

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
    resp_idx = load_responses_index(cfg)
    write_sample_csv(sample, csv_path, resp_idx)

    print(f"Wrote {len(sample)} rows to {csv_path}")
    print("Open it in a spreadsheet, fill the `human_label` column with one of "
          "{refused, complied, unclear}, save, then re-run with --score.")


if __name__ == "__main__":
    main()
