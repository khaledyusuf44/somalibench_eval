"""Quick sanity check on Phase-2 judge output before running analysis.

Prints per-model label distributions and a sample of judge reasons,
so you can verify the judge classifications look sensible.

Run:
    python src/inspect_classifications.py
"""

from __future__ import annotations

import json
from collections import Counter
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
CLASS_DIR = ROOT / "data" / "classifications"


def main() -> None:
    print("=" * 80)
    print("JUDGE CLASSIFICATION SUMMARY — sanity check before analysis")
    print("=" * 80)

    paths = sorted(CLASS_DIR.glob("*.jsonl"))
    if not paths:
        print(f"\nNo files in {CLASS_DIR}. Run src/02_judge_responses.py first.")
        return

    for p in paths:
        with p.open(encoding="utf-8") as f:
            rows = [json.loads(line) for line in f if line.strip()]

        print(f"\n📊 {p.stem}  (n={len(rows)})")

        # Label distribution per language
        for lang in ("en", "so"):
            lr = [r for r in rows if r["lang"] == lang]
            c = Counter(r["label"] for r in lr)
            ref = c.get("refused", 0)
            com = c.get("complied", 0)
            unc = c.get("unclear", 0)
            print(f"   {lang}:  refused={ref:3d}  complied={com:3d}  unclear={unc:3d}  (n={len(lr)})")

        # API errors / Anthropic refusals
        errs = [r for r in rows if r.get("error")]
        if errs:
            print(f"   ⚠️  API errors / refusals: {len(errs)}")

        # One example per (language, label) combo
        print(f"   --- sample judge reasons ---")
        seen: set[tuple[str, str]] = set()
        for r in rows:
            key = (r["lang"], r["label"])
            if key in seen or r.get("error"):
                continue
            seen.add(key)
            reason = (r.get("reason") or "")[:85]
            print(f'     [{r["lang"]}/{r["label"]:<8}] {r["probe_id"]}: {reason}')


if __name__ == "__main__":
    main()
