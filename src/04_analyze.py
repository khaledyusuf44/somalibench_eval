"""Phase 4 — Compute the headline numbers.

Reads:
  data/classifications/<model>.jsonl   — Phase-2 judge labels

Writes:
  data/results/refusal_rates.csv       — per (model, language) refusal rate + 95% CI
  data/results/cross_lingual_gap.csv   — per model: EN refusal − SO refusal + CI of the gap
  data/results/per_category.csv        — per (model, language, category) refusal rate
  data/results/summary.json            — machine-readable summary of all above

All numbers reported with 95% bootstrap CIs (500 resamples, seed=0) — matches the
SomaliWeb v1 convention.

Run:
    python src/04_analyze.py
"""

from __future__ import annotations

import json
from collections import defaultdict
from pathlib import Path

import numpy as np
import pandas as pd
import yaml

ROOT = Path(__file__).resolve().parent.parent
CONFIG_PATH = ROOT / "configs" / "eval_config.yaml"


def load_config() -> dict:
    with CONFIG_PATH.open(encoding="utf-8") as f:
        return yaml.safe_load(f)


def load_labels(cfg: dict) -> pd.DataFrame:
    """Concatenate every classification JSONL into a single DataFrame."""
    in_dir = ROOT / cfg["paths"]["classifications_dir"]
    rows = []
    for path in sorted(in_dir.glob("*.jsonl")):
        with path.open(encoding="utf-8") as f:
            for line in f:
                if line.strip():
                    rows.append(json.loads(line))
    df = pd.DataFrame(rows)
    if df.empty:
        raise SystemExit(f"No classifications in {in_dir}. Run Phase 2 first.")
    df["is_refused"] = (df["label"] == "refused").astype(int)
    return df


def bootstrap_ci(values: np.ndarray, n_resamples: int, seed: int,
                  level: float) -> tuple[float, float]:
    """Percentile bootstrap CI for the mean of a 0/1 array."""
    if len(values) == 0:
        return (float("nan"), float("nan"))
    rng = np.random.default_rng(seed)
    boot = rng.choice(values, size=(n_resamples, len(values)), replace=True).mean(1)
    lo = float(np.quantile(boot, (1 - level) / 2))
    hi = float(np.quantile(boot, 1 - (1 - level) / 2))
    return (lo, hi)


def refusal_rate_with_ci(df: pd.DataFrame, n: int, seed: int,
                         level: float) -> dict:
    arr = df["is_refused"].to_numpy()
    rate = float(arr.mean()) if len(arr) else float("nan")
    lo, hi = bootstrap_ci(arr, n, seed, level)
    return {"n": int(len(arr)), "refusal_rate": round(rate, 4),
            "ci_low": round(lo, 4), "ci_high": round(hi, 4)}


def gap_with_ci(df_en: pd.DataFrame, df_so: pd.DataFrame, n: int, seed: int,
                 level: float) -> dict:
    """Bootstrap CI for (EN refusal rate − SO refusal rate)."""
    en = df_en["is_refused"].to_numpy()
    so = df_so["is_refused"].to_numpy()
    if len(en) == 0 or len(so) == 0:
        return {"gap": float("nan"), "ci_low": float("nan"),
                "ci_high": float("nan")}
    rng = np.random.default_rng(seed)
    boot_en = rng.choice(en, size=(n, len(en)), replace=True).mean(1)
    boot_so = rng.choice(so, size=(n, len(so)), replace=True).mean(1)
    diff = boot_en - boot_so
    return {
        "gap":     round(float(en.mean() - so.mean()), 4),
        "ci_low":  round(float(np.quantile(diff, (1 - level) / 2)), 4),
        "ci_high": round(float(np.quantile(diff, 1 - (1 - level) / 2)), 4),
    }


def main() -> None:
    cfg = load_config()
    stats = cfg["stats"]
    n_boot = stats["bootstrap_resamples"]
    seed = stats["bootstrap_seed"]
    level = stats["confidence_level"]

    out_dir = ROOT / cfg["paths"]["results_dir"]
    out_dir.mkdir(parents=True, exist_ok=True)

    df = load_labels(cfg)
    print(f"Loaded {len(df)} classifications across "
          f"{df['model_id'].nunique()} models and {df['lang'].nunique()} languages")

    # ----- 1. Per (model, language) refusal rate with CI -----
    rows = []
    for (mid, lang), g in df.groupby(["model_id", "lang"]):
        rec = {"model_id": mid, "lang": lang,
               **refusal_rate_with_ci(g, n_boot, seed, level)}
        rows.append(rec)
    rate_df = pd.DataFrame(rows).sort_values(["model_id", "lang"])
    rate_df.to_csv(out_dir / "refusal_rates.csv", index=False)
    print(f"wrote {out_dir / 'refusal_rates.csv'}")

    # ----- 2. Cross-lingual gap with CI (the headline) -----
    rows = []
    for mid, g in df.groupby("model_id"):
        g_en = g[g["lang"] == "en"]
        g_so = g[g["lang"] == "so"]
        rec = {"model_id": mid,
               "en_refusal":  round(float(g_en["is_refused"].mean()), 4),
               "so_refusal":  round(float(g_so["is_refused"].mean()), 4),
               **gap_with_ci(g_en, g_so, n_boot, seed, level)}
        rows.append(rec)
    gap_df = pd.DataFrame(rows).sort_values("gap", ascending=False)
    gap_df.to_csv(out_dir / "cross_lingual_gap.csv", index=False)
    print(f"wrote {out_dir / 'cross_lingual_gap.csv'}")

    # ----- 3. Per-category breakdown -----
    rows = []
    for (mid, lang, cat), g in df.groupby(["model_id", "lang", "category"]):
        rec = {"model_id": mid, "lang": lang, "category": cat,
               **refusal_rate_with_ci(g, n_boot, seed, level)}
        rows.append(rec)
    cat_df = pd.DataFrame(rows).sort_values(["model_id", "lang", "category"])
    cat_df.to_csv(out_dir / "per_category.csv", index=False)
    print(f"wrote {out_dir / 'per_category.csv'}")

    # ----- 4. Summary JSON (paper-ready numbers) -----
    summary = {
        "n_total_classifications":   int(len(df)),
        "models":                    sorted(df["model_id"].unique().tolist()),
        "stats":                     {"bootstrap_resamples": n_boot,
                                      "seed": seed,
                                      "confidence_level": level},
        "headline_gaps":             gap_df.to_dict(orient="records"),
        "refusal_rates_per_model_lang": rate_df.to_dict(orient="records"),
    }
    (out_dir / "summary.json").write_text(
        json.dumps(summary, indent=2, ensure_ascii=False), encoding="utf-8")
    print(f"wrote {out_dir / 'summary.json'}")

    # ----- Console headline -----
    print("\n=== HEADLINE: cross-lingual refusal gap ===")
    print(gap_df.to_string(index=False))
    print("\nNext: python src/05_make_figures.py")


if __name__ == "__main__":
    main()
