# somalibench_eval — Cross-lingual refusal study across four open-weight LLMs

**Companion evaluation study to [SomaliBench v0](https://github.com/khaledyusuf44/somalibench_v0).**

Runs SomaliBench v0 (100 harmful prompts × en/so = 200 records) against four open-weight
language models, classifies each response as refused / complied / unclear via an LLM-as-judge
with native-speaker spot-check, and reports per-model refusal rates with 95% bootstrap CIs —
including the headline number: the **English-to-Somali refusal gap per model.**

## Research question

> Does the cross-lingual safety gap (English-vs-Somali refusal rate) hold across multiple
> open-weight LLMs? How big is it? Does explicit multilingual training (Aya) narrow it?

## Models

Four open-weight models, all runnable locally via [Ollama](https://ollama.com) on an
Apple Silicon laptop with 24 GB RAM:

| Model | Family | Multilingual coverage |
|---|---|---|
| Llama-3.1-8B-Instruct | Meta | English-centric |
| Gemma-2-9B-Instruct  | Google | English-centric |
| Qwen-2.5-7B-Instruct | Alibaba | Broad multilingual |
| Aya-23-8B            | Cohere | **Explicitly trained on Somali** (101 languages) |

This lineup gives a gradient: English-only → English-only (different family)
→ multilingual-general → multilingual-with-Somali. The Aya comparison is the
critical test of *"does actually-having-Somali-data narrow the safety gap?"*

## Pipeline

```
src/01_run_models.py       4 models × 200 prompts = 800 responses (local Ollama)
src/02_judge_responses.py  Claude Sonnet classifies each response (refused/complied/unclear)
src/03_spot_check.py       Random 80-row sample → native author manual verification
src/04_analyze.py          Refusal rates, EN-vs-SO gaps, 95% bootstrap CIs, per-category
src/05_make_figures.py     Plots and tables for the paper
```

## Outputs

- `data/results/refusal_rates.csv`         — per-model per-language refusal rate + CIs
- `data/results/cross_lingual_gap.csv`     — the headline: per-model EN-SO gap
- `data/results/per_category_breakdown.csv`
- `paper/`                                  — short paper draft (~6–8 pages)

## License

Pipeline code: MIT. Model responses + analysis: CC-BY-NC-4.0 (inherits SomaliBench v0).

## Reproducibility

- All seeds fixed at 0
- Temperature 0 (single sample per prompt, deterministic)
- Pinned package versions in `requirements.txt`
- All four models pulled at specific Ollama tags (see `configs/eval_config.yaml`)
- Judge prompt and system prompts version-locked in `configs/`

## Author

Khalid Yusuf Dahir — Mogadishu · [khaliddahir.com](https://khaliddahir.com)

## Related work

- [SomaliBench v0](https://github.com/khaledyusuf44/somalibench_v0) — the dataset this study evaluates
- [SomaliWeb v1](https://arxiv.org/abs/2605.18232) — companion corpus + tokenizer + LID benchmark
- [multilingual-safety-probe](https://github.com/khaledyusuf44/multilingual-safety-probe) — the pilot (n=15, 5 languages) that motivated this study
