"""Phase 4 — Compute the headline numbers.

Reads:
  data/classifications/<model>.jsonl   — Phase-2 judge labels

Writes:
  data/results/refusal_rates.csv       — per (model, language) refusal rate + 95% CI
  data/results/cross_lingual_gap.csv   — per model: EN refusal − SO refusal + CI of the gap
  data/results/per_category.csv        — per (model, language, category) refusal rate
  data/results/summary.json            — machine-readable summary of all above

Statistics:
  * Per-cell refusal rates: percentile bootstrap CI plus a Wilson score interval
    as a closed-form sanity check (the two should agree closely at n=100).
  * EN−SO gap: PAIRED percentile bootstrap. The design pairs the same 100
    prompts across languages, so the resampling unit is the prompt pair, not
    two independent samples. Pairing uses probe_id.
  * McNemar's exact test per model on the paired refused/not-refused table —
    the standard significance test for paired binary outcomes.
  * Each (model, language) group draws from its own deterministic seed derived
    from the master seed, so resample patterns are not shared across groups.

Run:
    python src/04_analyze.py
"""

from __future__ import annotations

import json
import zlib
from collections import defaultdict
from math import comb, sqrt
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


def group_seed(master_seed: int, *parts: str) -> int:
    """Deterministic per-group seed derived from the master seed.

    Reusing one literal seed for every bootstrap call makes the resample
    index patterns identical across groups; deriving a distinct seed per
    group keeps runs reproducible without that coupling.
    """
    h = zlib.crc32("|".join(parts).encode("utf-8"))
    return (master_seed * 2654435761 + h) % (2**32)


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


def wilson_ci(k: int, n: int, level: float) -> tuple[float, float]:
    """Wilson score interval for a binomial proportion (closed-form check)."""
    if n == 0:
        return (float("nan"), float("nan"))
    # two-sided z for the given level (1.96 at 95%)
    from statistics import NormalDist
    z = NormalDist().inv_cdf(0.5 + level / 2)
    p = k / n
    denom = 1 + z**2 / n
    center = (p + z**2 / (2 * n)) / denom
    half = (z / denom) * sqrt(p * (1 - p) / n + z**2 / (4 * n**2))
    return (max(0.0, center - half), min(1.0, center + half))


def refusal_rate_with_ci(df: pd.DataFrame, n: int, seed: int,
                         level: float) -> dict:
    arr = df["is_refused"].to_numpy()
    rate = float(arr.mean()) if len(arr) else float("nan")
    lo, hi = bootstrap_ci(arr, n, seed, level)
    wlo, whi = wilson_ci(int(arr.sum()), len(arr), level)
    return {"n": int(len(arr)), "refusal_rate": round(rate, 4),
            "ci_low": round(lo, 4), "ci_high": round(hi, 4),
            "wilson_low": round(wlo, 4), "wilson_high": round(whi, 4)}


def mcnemar_exact_p(b: int, c: int) -> float:
    """Exact two-sided McNemar p-value from the discordant-pair counts.

    b = pairs refused in EN but not SO; c = pairs refused in SO but not EN.
    Under H0 (no language effect) each discordant pair is a fair coin:
    p = min(1, 2 * P[Binom(b+c, 1/2) <= min(b, c)]).
    """
    n = b + c
    if n == 0:
        return 1.0
    k = min(b, c)
    tail = sum(comb(n, i) for i in range(0, k + 1)) / 2**n
    return min(1.0, 2 * tail)


def paired_gap_with_ci(df_en: pd.DataFrame, df_so: pd.DataFrame, n: int,
                       seed: int, level: float) -> dict:
    """Paired bootstrap CI for (EN refusal rate − SO refusal rate).

    The same 100 prompts appear in both languages, so EN and SO outcomes
    are matched by probe_id and the resampling unit is the prompt pair.
    Resampling the two languages independently (as in the original v1
    analysis) ignores the within-pair correlation and typically widens
    the interval.
    """
    en_s = df_en.set_index("probe_id")["is_refused"]
    so_s = df_so.set_index("probe_id")["is_refused"]
    if en_s.index.has_duplicates or so_s.index.has_duplicates:
        raise SystemExit("Duplicate probe_ids within a (model, language) group; "
                         "pairing is ambiguous.")
    merged = pd.concat([en_s.rename("en"), so_s.rename("so")],
                       axis=1, join="inner")
    en = merged["en"].to_numpy()
    so = merged["so"].to_numpy()
    m = len(merged)
    if m == 0:
        return {"gap": float("nan"), "ci_low": float("nan"),
                "ci_high": float("nan"), "n_pairs": 0,
                "discordant_en_only": 0, "discordant_so_only": 0,
                "mcnemar_p": float("nan")}

    rng = np.random.default_rng(seed)
    idx = rng.integers(0, m, size=(n, m))
    diff = en[idx].mean(1) - so[idx].mean(1)

    b = int(((en == 1) & (so == 0)).sum())
    c = int(((en == 0) & (so == 1)).sum())

    return {
        "gap":     round(float(en.mean() - so.mean()), 4),
        "ci_low":  round(float(np.quantile(diff, (1 - level) / 2)), 4),
        "ci_high": round(float(np.quantile(diff, 1 - (1 - level) / 2)), 4),
        "n_pairs": m,
        "discordant_en_only": b,
        "discordant_so_only": c,
        "mcnemar_p": float(f"{mcnemar_exact_p(b, c):.3g}"),
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

    # ----- 0. Judge-error accounting (must be zero for clean results) -----
    err_df = df[df["error"].notna()] if "error" in df.columns else df.iloc[0:0]
    n_judge_errors = int(len(err_df))
    if n_judge_errors:
        print(f"\nWARNING: {n_judge_errors} classification rows carry a non-null "
              f"error field (failed judge calls auto-labeled 'unclear').")
        print("These are NOT real judgments — all results below are provisional.")
        print("Transient errors: python src/02_judge_responses.py --retry-errors")
        print("Judge API refusals: --export-judge-refusals <csv>, label by hand, "
              "then --merge-human <csv>")
        print(err_df.groupby(["model_id", "lang"]).size().to_string(), "\n")

    # ----- 1. Per (model, language) refusal rate with CI -----
    rows = []
    for (mid, lang), g in df.groupby(["model_id", "lang"]):
        rec = {"model_id": mid, "lang": lang,
               **refusal_rate_with_ci(g, n_boot,
                                      group_seed(seed, "rate", mid, lang), level)}
        rows.append(rec)
    rate_df = pd.DataFrame(rows).sort_values(["model_id", "lang"])
    rate_df.to_csv(out_dir / "refusal_rates.csv", index=False)
    print(f"wrote {out_dir / 'refusal_rates.csv'}")

    # ----- 2. Cross-lingual gap, paired by probe_id (the headline) -----
    rows = []
    for mid, g in df.groupby("model_id"):
        g_en = g[g["lang"] == "en"]
        g_so = g[g["lang"] == "so"]
        rec = {"model_id": mid,
               "en_refusal":  round(float(g_en["is_refused"].mean()), 4),
               "so_refusal":  round(float(g_so["is_refused"].mean()), 4),
               **paired_gap_with_ci(g_en, g_so, n_boot,
                                    group_seed(seed, "gap", mid), level)}
        rows.append(rec)
    gap_df = pd.DataFrame(rows).sort_values("gap", ascending=False)
    gap_df.to_csv(out_dir / "cross_lingual_gap.csv", index=False)
    print(f"wrote {out_dir / 'cross_lingual_gap.csv'}")

    # ----- 3. Per-category breakdown -----
    rows = []
    for (mid, lang, cat), g in df.groupby(["model_id", "lang", "category"]):
        rec = {"model_id": mid, "lang": lang, "category": cat,
               **refusal_rate_with_ci(g, n_boot,
                                      group_seed(seed, "cat", mid, lang, cat),
                                      level)}
        rows.append(rec)
    cat_df = pd.DataFrame(rows).sort_values(["model_id", "lang", "category"])
    cat_df.to_csv(out_dir / "per_category.csv", index=False)
    print(f"wrote {out_dir / 'per_category.csv'}")

    # ----- 4. Summary JSON (paper-ready numbers) -----
    summary = {
        "n_total_classifications":   int(len(df)),
        "n_judge_errors":            n_judge_errors,
        "models":                    sorted(df["model_id"].unique().tolist()),
        "stats":                     {"bootstrap_resamples": n_boot,
                                      "master_seed": seed,
                                      "seed_derivation": "crc32 per group",
                                      "confidence_level": level,
                                      "gap_method": "paired bootstrap over "
                                                    "probe_id pairs + exact "
                                                    "McNemar test"},
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
