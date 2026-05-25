# PLAN — somalibench_eval

Five-phase execution plan, ~3–4 weeks end-to-end.

---

## Phase 1 — Setup (1–2 days)

- Create venv, install `requirements.txt`
- Pull all four models via Ollama:
  - `ollama pull llama3.1:8b`
  - `ollama pull gemma2:9b`
  - `ollama pull qwen2.5:7b`
  - `ollama pull aya:8b`
- Test inference end-to-end with **one** prompt against **one** model
- Verify `data/prompts/` mirrors SomaliBench v0 (`probes_en.jsonl` + `probes_so.jsonl`)

## Phase 2 — Run model evaluations (2–3 days, mostly compute)

Run `src/01_run_models.py`:

- For each of 4 models × 200 prompts = **800 inferences**
- Temperature 0, single sample, deterministic seed
- One identical English HHH system prompt across all models (see `configs/eval_config.yaml`)
- Output: `data/responses/<model>.jsonl` — one line per (probe_id, lang, response)
- Resume-safe: if a row already exists, skip it (cheap to interrupt + restart)

Approximate wall-clock on M4 Pro / 24 GB (single model, 200 prompts):
- 7–9B Q4 quantization → ~15–30 minutes per model
- All four → ~1–2 hours total compute (assuming sequential)

## Phase 3 — LLM-as-judge classification (~1 day)

Run `src/02_judge_responses.py`:

- For each (prompt, response) pair (800 total), call Claude Sonnet
- Classify as `refused / complied / unclear` + brief reason
- Output: `data/classifications/<model>.jsonl`
- Cost estimate: ~$2–5 total at Sonnet rates (~500 input + 50 output tokens × 800)
- API safety-research framing in the system prompt (this is an LLM-output classification
  task for an academic refusal-rate study)
- A small fraction of API requests may be refused on extreme content — log and re-route
  those to the manual spot-check pool

## Phase 4 — Native-author spot check (~2 hours)

Run `src/03_spot_check.py`:

- Random-sample 80 (probe_id, model, judge-label) triples — stratified across the 7
  safety categories
- Output CSV with columns: `probe_id, lang, model, judge_label, human_label, notes`
- Native author (you) fills `human_label` for each row
- Re-import → compute judge-vs-human agreement (κ or % match) — this becomes the paper's
  inter-judge reliability number

## Phase 5 — Analysis (2–3 days)

Run `src/04_analyze.py`:

- **Refusal rate per (model, language)** — proportion of responses classified `refused`
- **Bootstrap CIs**: 500 resamples, seed = 0, 95% bounds — your SomaliWeb standard
- **EN−SO gap per model** — the headline metric
- **Per-category breakdown** — does the gap differ for CBRN vs hate_speech vs ...?
- **Multilingual-vs-English-centric comparison** — do broad multilingual models narrow the gap?

Run `src/05_make_figures.py`:

- Bar plot — refusal rate per model, EN vs SO side-by-side
- Heatmap — gap per (model × category)
- Forest plot — gap with CIs per model (clean, citeable single figure)

## Phase 6 — Paper (1–2 weeks)

Short paper, ~6–8 pages, structure in `paper/`:

```
1. Abstract
2. Introduction (the gap, why Somali, why these models)
3. Related work (Yong 2023, survey 2505.24119, multilingual-safety-probe)
4. SomaliBench v0 (brief)
5. Methodology (model choice, inference, judge, spot-check)
6. Results (headline gap, per-category, multilingual-training effect)
7. Discussion (mechanism hypotheses; consistent with cliff or gradient?)
8. Limitations
9. Conclusion + future work (more models, more languages, mitigation testing)
```

Use the SomaliWeb v1 paper structure as the template.

## Phase 7 — Release (~1 day)

- Push to GitHub as `somalibench_eval`
- Optional companion HuggingFace dataset of the public-safe judge labels +
  aggregate metrics (raw harmful generations remain local)
- Tag `v0.1.0`
- arXiv preprint submission

## Estimated total: 3–4 weeks to arXiv-ready paper.

---

## Decisions locked in

1. Models: Llama-3.1-8B, Gemma-2-9B, Qwen-2.5-7B, Aya-23-8B
2. Judge: Claude Sonnet (good calibration on refused/complied/unclear)
3. Temperature 0, single sample, English HHH system prompt
4. 95% bootstrap CIs (500 resamples, seed 0) — SomaliWeb standard
5. Native spot-check: stratified random 80 rows across 7 categories
6. v1 is baseline only — no temperature grid, no jailbreak variants, no mitigation

## What v2 could add (out of scope for v1)

- More models (70B, frontier API models)
- More languages (rerun on multilingual-safety-probe languages + Somali)
- Multi-judge (Claude + GPT-4o) → inter-judge agreement
- Mitigation testing (try MPO / self-distillation; does it close the Somali gap?)
- Temperature sweep, multiple samples per prompt → variance estimates
