"""Phase 5 — Generate paper figures from Phase-4 result tables.

Reads:
  data/results/refusal_rates.csv
  data/results/cross_lingual_gap.csv
  data/results/per_category.csv

Writes (in data/figures/):
  fig1_refusal_rates.pdf     — bar chart, EN vs SO per model, with CI whiskers
  fig2_gap_forest.pdf        — forest plot of the EN-SO gap per model
  fig3_per_category.pdf      — heatmap: gap by (model × category)

Run:
    python src/05_make_figures.py
"""

from __future__ import annotations

from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import yaml

ROOT = Path(__file__).resolve().parent.parent
CONFIG_PATH = ROOT / "configs" / "eval_config.yaml"


def load_config() -> dict:
    with CONFIG_PATH.open(encoding="utf-8") as f:
        return yaml.safe_load(f)


# ---- Figure 1: per-model refusal rate (EN vs SO) with CI bars ----------------
def fig_refusal_rates(rate_df: pd.DataFrame, out_path: Path) -> None:
    models = sorted(rate_df["model_id"].unique())
    x = np.arange(len(models))
    width = 0.35

    en = rate_df[rate_df["lang"] == "en"].set_index("model_id").loc[models]
    so = rate_df[rate_df["lang"] == "so"].set_index("model_id").loc[models]

    fig, ax = plt.subplots(figsize=(8, 4.5))
    ax.bar(x - width/2, en["refusal_rate"], width, label="English",
           yerr=[en["refusal_rate"] - en["ci_low"],
                 en["ci_high"] - en["refusal_rate"]],
           capsize=3, color="#1F4E79")
    ax.bar(x + width/2, so["refusal_rate"], width, label="Somali",
           yerr=[so["refusal_rate"] - so["ci_low"],
                 so["ci_high"] - so["refusal_rate"]],
           capsize=3, color="#C62828")

    ax.set_xticks(x)
    ax.set_xticklabels(models, rotation=20, ha="right")
    ax.set_ylabel("Refusal rate")
    ax.set_ylim(0, 1.05)
    ax.set_title("Refusal rate per model — English vs Somali (95% bootstrap CI)")
    ax.legend()
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    plt.tight_layout()
    plt.savefig(out_path, bbox_inches="tight")
    plt.close()
    print(f"wrote {out_path}")


# ---- Figure 2: forest plot of the EN-SO gap with CIs -------------------------
def fig_gap_forest(gap_df: pd.DataFrame, out_path: Path) -> None:
    g = gap_df.sort_values("gap")
    fig, ax = plt.subplots(figsize=(7.5, 0.6 * len(g) + 1.5))
    y = np.arange(len(g))

    err_low = g["gap"] - g["ci_low"]
    err_high = g["ci_high"] - g["gap"]
    ax.errorbar(g["gap"], y, xerr=[err_low, err_high], fmt="o",
                color="#1F4E79", capsize=4, markersize=7, linewidth=1.5)

    ax.axvline(0, color="#555", linestyle="--", linewidth=0.8)
    ax.set_yticks(y)
    ax.set_yticklabels(g["model_id"])
    ax.set_xlabel("EN − SO refusal rate (positive = refuses more in English)")
    ax.set_title("Cross-lingual refusal gap per model (95% bootstrap CI)")
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    plt.tight_layout()
    plt.savefig(out_path, bbox_inches="tight")
    plt.close()
    print(f"wrote {out_path}")


# ---- Figure 3: per-category gap heatmap (model × category) -------------------
def fig_per_category(cat_df: pd.DataFrame, out_path: Path) -> None:
    # Compute per (model, category) EN-SO gap
    pivot_en = cat_df[cat_df["lang"] == "en"].pivot_table(
        index="model_id", columns="category", values="refusal_rate")
    pivot_so = cat_df[cat_df["lang"] == "so"].pivot_table(
        index="model_id", columns="category", values="refusal_rate")
    gap = (pivot_en - pivot_so).reindex(sorted(pivot_en.index))

    fig, ax = plt.subplots(figsize=(0.9 * len(gap.columns) + 2, 0.7 * len(gap) + 2))
    im = ax.imshow(gap.values, cmap="RdBu_r", vmin=-1, vmax=1, aspect="auto")

    ax.set_xticks(np.arange(len(gap.columns)))
    ax.set_xticklabels(gap.columns, rotation=35, ha="right")
    ax.set_yticks(np.arange(len(gap.index)))
    ax.set_yticklabels(gap.index)

    for i in range(gap.shape[0]):
        for j in range(gap.shape[1]):
            v = gap.values[i, j]
            ax.text(j, i, f"{v:+.2f}" if not np.isnan(v) else "—",
                    ha="center", va="center",
                    color="white" if abs(v) > 0.5 else "black", fontsize=9)

    cbar = plt.colorbar(im, ax=ax, fraction=0.025)
    cbar.set_label("EN − SO refusal rate")
    ax.set_title("Cross-lingual gap per (model × safety category)")
    plt.tight_layout()
    plt.savefig(out_path, bbox_inches="tight")
    plt.close()
    print(f"wrote {out_path}")


def main() -> None:
    cfg = load_config()
    results_dir = ROOT / cfg["paths"]["results_dir"]
    figs_dir = ROOT / cfg["paths"]["figures_dir"]
    figs_dir.mkdir(parents=True, exist_ok=True)

    required = ["refusal_rates.csv", "cross_lingual_gap.csv", "per_category.csv"]
    missing = [f for f in required if not (results_dir / f).exists()]
    if missing:
        raise SystemExit(
            f"Missing result files in {results_dir}: {missing}\n"
            f"Run `python src/04_analyze.py` first.")

    rate_df = pd.read_csv(results_dir / "refusal_rates.csv")
    gap_df  = pd.read_csv(results_dir / "cross_lingual_gap.csv")
    cat_df  = pd.read_csv(results_dir / "per_category.csv")

    fig_refusal_rates(rate_df, figs_dir / "fig1_refusal_rates.pdf")
    fig_gap_forest(gap_df, figs_dir / "fig2_gap_forest.pdf")
    fig_per_category(cat_df, figs_dir / "fig3_per_category.pdf")

    print("\nFigures done.")


if __name__ == "__main__":
    main()
