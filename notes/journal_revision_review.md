# Deep review for journal revision — 2026-08-01

Findings from a full audit of paper/main.tex against the actual data and code,
going deeper than the mentor's skim. Ordered by severity. Khalid writes all
paper text per the working norm; this file only records what needs fixing and why.

---

## A. Data-integrity issues (must fix — these change reported numbers)

### A1. 38 judge API calls failed and were silently labeled "unclear"

- 38 of 800 classification rows (4.75%) have `error: "list index out of range"` —
  an exception in `src/02_judge_responses.py`, most likely `response.content[0]`
  on an empty content block. These rows were auto-labeled `unclear` without any
  actual judgment.
- Breakdown: 12 EN + 26 SO; per model: aya 13, gemma 6, llama 11, qwen 8.
- **Critical detail:** the paper's Table 4 shows exactly 3 EN "unclear" per model
  (12 total). All 12 are these error rows. So *every English "unclear" in the
  paper is a failed judge call*, not a judged response.
- The paper text ("in the completed run there are 800 classifications") implies
  zero judge-side failures. This is inaccurate as written.
- **Fix:** patch the exception handling, re-run the judge on the 38 failed rows,
  regenerate `04_analyze` + `05_make_figures`, and update every number in the
  paper. EN refusal rates will likely rise slightly (e.g., Llama EN could go
  0.97 → up to 1.00); headline gaps may shift by a point or two.
- Then document judge-failure handling honestly in the methodology.

### A2. Spot-check sample is contaminated by the error rows

- 6 of the 80 spot-check rows are judge-error rows. The human "agreed" with a
  label that was produced by an exception, not by the judge.
- The κ = 1.00 claim therefore rests on 74 genuine comparisons, not 80.
- **Fix:** after A1, re-draw (or at minimum re-score) the spot-check sample from
  the corrected classifications and recompute agreement. Update abstract,
  §4.4, and `data/spot_checks/agreement.json`.

### A3. "Empty" responses do not exist in the data

- The abstract, Table 5, and §5.3 describe an "empty" Somali failure mode.
  Audit result: **0 of 800 responses are empty/whitespace**, and 0 of the judge's
  reason strings mention emptiness. The genuine unclear modes in the data are:
  incoherent/repetitive text, generic deflection, prompt-echoing, clarifying
  questions, off-topic drift, and wrong-language output.
- This is exactly the kind of small unverified claim the mentor warned about.
- **Fix:** reword all "empty, wrong-language, or incoherent" claims to match the
  observed modes (verify "wrong-language" frequency too before keeping it).

---

## B. Statistical methodology (reviewers will push here)

### B1. Gap CI ignores the paired design

- `gap_with_ci` in `src/04_analyze.py:75-91` bootstraps EN and SO refusal
  *independently*, but the design is paired: the same 100 prompts appear in both
  languages. The correct resampling unit is the prompt pair.
- Independent resampling is likely conservative (wider CI) when outcomes are
  positively correlated, so the headline conclusion survives — but a journal
  reviewer will flag it, and the paired analysis is *stronger* for the paper.
- **Fix:** paired bootstrap (resample prompt indices, compute per-resample
  EN−SO difference) + McNemar's exact test per model on the paired
  refused/not-refused table. Report both.

### B2. Bootstrap details

- 500 resamples is low; 10,000 is standard and computationally free at n=100.
- The same seed (0) is reused for every group's resampling, so resample patterns
  are correlated across models/languages. Derive a distinct seed per group from
  a master seed.
- Optional sanity check: Wilson intervals alongside bootstrap for the per-cell
  rates (they should agree closely; saying so preempts a reviewer question).

### B3. Agreement statistic needs uncertainty

- Perfect agreement on n=80 should carry a CI (rule-of-three: 95% lower bound on
  the agreement rate ≈ 0.955 at 80/80). Report it; κ = 1.00 alone looks
  overconfident.

### B4. Spot-check stratification is category-only

- Stratifying only by category (config `stratify_by: category`) means model,
  language, and label coverage in the sample was luck (only 6 complied rows
  sampled). State the stratification exactly in the paper; for v2, stratify by
  label (oversample `complied`), since that is the class where judge errors are
  costliest.

---

## C. Methodology gaps a journal reviewer may raise

### C1. Judge competence in Somali is assumed, not established

The judge must read Somali to distinguish refusal/compliance/unclear. The native
spot-check partially validates this, but the paper should state the assumption
explicitly and note that judge failure on Somali would inflate `unclear`.

### C2. Truncation cannot be audited

`max_tokens: 512`, but the response JSONLs don't record `done_reason`, so
truncated generations can't be identified post-hoc. Truncation could turn a
partial compliance into an apparent non-answer. Log `done_reason` in v2; note
the limitation now.

### C3. Cross-model quantization confound

Aya ran at F16 while the other three ran at Q4 (Table 3). The current
limitations item frames quantization as an absolute-rate issue; the sharper
problem is *cross-model comparability* — Aya's numbers come from a
differently-degraded model than the others. Also, Ollama applies each model's
own chat template; template differences are part of the measured effect.

### C4. Tokenizer-fertility analysis is incomplete and unreported

- `08_tokenizer_fertility.py` succeeded only for Qwen; the other three tokenizer
  repos are gated on HF (`data/tokenizer_errors/tokenizer_fertility_errors.json`,
  401 errors). Fix: `huggingface-cli login` + accept the three gates, rerun.
- The Qwen result is a strong mechanism data point: Somali ≈ 2.50 tokens/word
  vs English ≈ 1.09 on the same safety prompts. Completing all four would let
  the Gemma-outlier discussion (§6) test its own tokenizer hypothesis instead
  of leaving it as speculation.

### C5. Translation validation is single-annotator

The Somali translations were produced and verified by the same person who is
the author and the spot-check annotator. SomaliBench v0 may document this, but
this paper should acknowledge the triple role explicitly as a limitation.

### C6. Missing related work / self-citation

- The pilot study (`multilingual-safety-probe`, n=15, 5 languages) motivated
  this work and is cited in the README but not in the paper.
- Related work is thin for a journal. Candidates: XSafety, MultiJail is cited
  (deng2024), PolygloToxicityPrompts, Aya Red-Teaming dataset, M-ALERT, RTP-LX,
  and work on refusal-direction/mechanism if the Discussion keeps hypotheses.

### C7. Benign-control table readability

Table 6 mixes proportions (0–1) with a 0–2 grammar scale in adjacent columns.
Normalize or visually separate; consider adding binomial CIs even for n=50.

---

## D. Structure and presentation (beyond the mentor's four points)

- D1. Related Work (§3) comes *after* the Benchmark section (§2) — unusual.
  Consider Intro → Related Work → Benchmark → Methodology, or fold §2 into §4.
- D2. Abstract is dense with numbers and contains the Code/Benchmark/License
  block — move links to a title footnote or a footnote at the end of the intro.
- D3. Table 5 is titled "Redacted examples" but contains no examples, only
  descriptions of patterns. Retitle honestly (e.g., observed failure modes).
- D4. `\date{May 2026}` — update on replacement; arXiv v2 will show its own date.
- D5. Hyphenation inconsistency: "native-author" (line 64) vs "native author"
  elsewhere; pick one.
- D6. Equations (1)–(2) are trivial definitional equations; fine, but they can
  be inline prose in a journal version.
- D7. §2 largely restates the SomaliBench v0 release; condense and cite rather
  than duplicate, keeping only what's needed to read the results.

Plus the mentor's four: contributions → prose at end of intro; limitations →
prose with depth; appendices → prose; LLM-use disclosure statement.

---

## Status update — 2026-08-01 (code side done)

- A1 investigated: the 38 "errors" split into 4 transient failures (re-judged
  successfully, all confirmed `unclear`) and **34 judge-API refusals** —
  `stop_reason: "refusal"` with an empty content block. The judge model's
  safety layer declines to process these (prompt, response) pairs. This is the
  scenario PLAN.md Phase 3 predicted; the code now detects it explicitly.
- The 34 rows are exported to `data/spot_checks/judge_refusals_for_review.csv`
  (gitignored) for native-author manual labeling → merge back with
  `python src/02_judge_responses.py --merge-human <csv>`, then re-run
  `04_analyze` + `05_make_figures` for final numbers.
- B1/B2 implemented: paired bootstrap over probe_id pairs, exact McNemar test,
  10k resamples, per-group derived seeds, Wilson sanity intervals.
- A2 resolved: spot-check re-scored on the 74 rows with genuine API-judge
  labels (6 judge-refusal rows excluded). Agreement remains 100%, κ = 1.00,
  n = 74 (41 refused / 27 unclear / 6 complied). Paper must report n = 74.
- Provisional headline (34 pending rows counted as unclear): gaps and McNemar
  all confirm the effect — llama 0.90 [0.84, 0.95] p≈2e-27; aya 0.75
  [0.64, 0.85] p≈4e-19; qwen 0.69 [0.59, 0.78] p≈6e-19; gemma 0.38
  [0.29, 0.48] p≈7e-12. Numbers finalize after the 34 manual labels.
- New paper content required: methodology must describe judge-API refusals and
  the manual-labeling fallback; reliability section reports n = 74.

## Suggested order of work

1. Fix judge script bug; re-run 38 failed calls (pennies of API cost).
2. Regenerate analysis + figures; recompute spot-check agreement (A1, A2).
3. Implement paired bootstrap + McNemar, 10k resamples (B1, B2).
4. Complete tokenizer fertility for all four models (C4).
5. Rewrite paper text (Khalid) with corrected numbers, mentor's structural
   changes, and the reworded failure-mode claims (A3) — critique pass after
   each section.
6. Rebuild, re-zip, submit arXiv v2; then format for target journal.
