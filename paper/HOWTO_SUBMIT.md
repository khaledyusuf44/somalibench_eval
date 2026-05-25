# arXiv Submission Checklist

## 1. Rebuild public-safe artifacts

From the repository root:

```bash
source .venv/bin/activate
python src/04_analyze.py
python src/05_make_figures.py
python src/03_spot_check.py --score
```

Then from `paper/`:

```bash
make figs
make zip
```

## 2. Verify the zip contents

The zip should contain only:

```text
main.tex
refs.bib
figures/fig1_refusal_rates.pdf
figures/fig2_gap_forest.pdf
figures/fig3_per_category.pdf
```

It must not contain:

```text
data/responses/
data/classifications/
data/spot_checks/sample_for_review.csv
.env
.venv/
*.log
```

## 3. Suggested arXiv metadata

Title:

```text
SomaliBench Eval: Measuring English-to-Somali Refusal Gaps in Open-Weight Language Models
```

Primary category:

```text
cs.CL
```

Secondary category:

```text
cs.AI
```

Comments:

```text
Preprint. Defensive multilingual safety evaluation. Aggregate statistics only; raw model generations are not released.
```

## 4. Upload flow

1. Upload `paper/arxiv-submission.zip`.
2. Let arXiv compile from source.
3. Check that all three figures render.
4. Check that the bibliography appears.
5. Confirm that no raw model responses or manual-review CSV are included.
6. Use a permissive paper license if desired; keep benchmark data under its CC-BY-NC-4.0 license.

## 5. If arXiv compile fails

This source intentionally uses only standard packages: `article`, `geometry`, `microtype`, `amsmath`, `graphicx`, `booktabs`, `enumitem`, `hyperref`, `natbib`, and `caption`.

Common fixes:

- If a figure is missing, run `make figs && make zip`.
- If references are missing on the first preview, click recompile once.
- If arXiv complains about generated files, remove local build products with `make clean` and rebuild the zip.
