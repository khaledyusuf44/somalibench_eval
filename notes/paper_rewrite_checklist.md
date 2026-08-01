# Paper rewrite checklist — journal revision (arXiv v2)

Khalid writes every step; Claude critiques after each one (the Catherine rule).
Workflow loop for every step: **edit `paper/main.tex` → say "build" → say
"critique" → revise → commit.**

All numbers referenced below are final and verified — sources:
`data/results/summary.json`, `data/results/*.csv`,
`data/spot_checks/agreement.json`, `notes/journal_revision_review.md`.

---

## Phase 1 — Skeleton (no new prose, just moving)

- [ ] **1.1** Delete the `\paragraph{Contributions.}` C1–C4 bullet block
      (currently sits between the abstract and Section 1 — most blog-like
      element in the paper). It gets rewritten as prose in Step 5.
- [ ] **1.2** Move the Related Work section above the Benchmark section.
      New order: Intro → Related Work → Benchmark → Methodology.
- [ ] **1.3** Condense the Benchmark section: cite SomaliBench v0 instead of
      restating its release details; keep only what a reader needs to
      interpret the results.
- [ ] Build + critique pass.

## Phase 2 — Sections with new facts (write these first)

### Step 2 — Methodology rewrite  *(mentor: less blog-like; audit: new facts)*

- [ ] **2.1** New paragraph: judge-API refusals. Facts: 34/800 rows (4.25%)
      returned `stop_reason: refusal` — the judge's safety layer declined to
      process the (prompt, response) pair. Routed to native-author manual
      labeling: 12 EN refused, 2 SO refused, 3 SO complied, 17 SO unclear;
      recorded with `judge_model = human:native-author` provenance. Also: 4
      transient API failures resolved by clean retries.
- [ ] **2.2** Statistics subsection: paired percentile bootstrap over prompt
      pairs (the same 100 prompts appear in both languages, so the resampling
      unit is the pair, matched by probe_id); exact McNemar test on the
      discordant pairs; 10,000 resamples; per-group derived seeds; Wilson
      score intervals reported where the bootstrap is degenerate (the two
      1.00 refusal cells → Wilson [0.96, 1.00]).
- [ ] **2.3** Spot-check subsection: agreement is n = 74 (not 80) — 6 sampled
      rows had no genuine API-judge label; 100% agreement, κ = 1.00
      (41 refused / 27 unclear / 6 complied). State stratification honestly:
      category-only, so label coverage in the sample was not controlled.
- [ ] **2.4** Inference bullets (settings list) → one or two sentences of prose.
- [ ] Build + critique pass.

### Step 3 — Results rewrite  *(all numbers change)*

- [ ] **3.1** Rates table: EN — Llama 1.00, Gemma 1.00, Qwen 0.96, Aya 0.83;
      SO — Llama 0.07, Gemma 0.60, Qwen 0.25, Aya 0.05. Use Wilson CIs for
      the 1.00 cells.
- [ ] **3.2** Gaps table: Llama 0.93 [0.88, 0.98]; Aya 0.78 [0.68, 0.87];
      Qwen 0.71 [0.61, 0.80]; Gemma 0.40 [0.31, 0.50]. Add discordant pairs
      + McNemar p (Llama: 93 vs 0, p = 2.0e-28; Aya: 82 vs 4, p = 5.8e-20;
      Qwen: 73 vs 2, p = 1.5e-19; Gemma: 40 vs 0, p = 1.8e-12).
- [ ] **3.3** Label-distribution table: EN unclear is 0 everywhere — delete
      every claim built on the old "3 EN unclear per model" (it was a judge
      crash artifact). New counts in data/classifications (refused/complied/
      unclear): aya en 83/17/0, so 5/4/91; gemma en 100/0/0, so 60/10/30;
      llama en 100/0/0, so 7/12/81; qwen en 96/4/0, so 25/4/71.
- [ ] **3.4** Remove ALL "empty response" claims (abstract, failure-mode
      table, §5.3) — zero empty responses exist in the data. Real unclear
      modes: incoherent/repetitive, generic deflection, prompt-echo,
      clarifying questions, wrong-language.
- [ ] **3.5** Retitle the failure-modes table (currently "Redacted examples"
      — it contains no examples, only pattern descriptions).
- [ ] **3.6** Compliance-coherence paragraph: 30 SO complied (was 27);
      9 coherent Somali-dominant — 9 of Gemma's 10, so "all of Gemma's were
      coherent" is no longer exact; 21 mixed/degraded incl. the 3 recovered
      rows (Somali-dominant but grammatically broken, not mixed-language).
- [ ] **3.7** Per-category: new order — privacy 0.80, harmful_instruction
      0.76, CBRN 0.73, self_harm 0.73, cybersecurity 0.70, hate_speech 0.70,
      misinformation 0.53. The old "CBRN is small (0.57)" sentence is wrong.
- [ ] Build + critique pass.

### Step 4 — Discussion: tokenizer fertility  *(new evidence)*

- [ ] **4.1** Add fertility results (data/results/tokenizer_fertility_summary):
      EN ~1.09 tokens/word for every model vs SO 2.20 (gemma), 2.34 (aya),
      2.49 (llama), 2.51 (qwen); 68–77% of Somali words split into pieces.
- [ ] **4.2** Upgrade the Gemma-outlier paragraph from speculation to
      "consistent with": Gemma has the lowest SO fertility AND the highest
      SO refusal. Keep the honest counterweight: Aya has second-lowest
      fertility but the LOWEST SO refusal — fertility is a factor, not the
      explanation.
- [ ] Build + critique pass.

## Phase 3 — Mentor's structural items

### Step 5 — Contributions as prose, end of introduction  *(mentor #3)*

- [ ] One flowing paragraph closing the intro — same four ideas
      (measurement, headline gap, validated judge, release discipline), no
      labels/bullets/bold. Reader now has context because it comes last.
- [ ] Add pilot-study citation (multilingual-safety-probe, n=15, 5 languages)
      to the intro's motivation.
- [ ] Build + critique pass.

### Step 6 — Limitations as deep prose  *(mentor #4)*

Merge 8 bullets into 4–5 paragraphs; each answers: how big is the effect,
which direction does it bias, what would rule it out. Groupings:

- [ ] **6.1** Benchmark size + no category-level CIs.
- [ ] **6.2** Single judge + judge-refusal fallback + the author's triple
      role (author = translator = spot-check annotator) stated explicitly.
- [ ] **6.3** Quantization reframed as cross-model comparability (Aya F16 vs
      others Q4) + untracked truncation (done_reason not logged).
- [ ] **6.4** English HHH system-prompt conditioning.
- [ ] **6.5** No severity scoring / no jailbreaks / raw generations local-only.
      (Mine notes/decisions.md — much of this reasoning already exists in
      Khalid's own words.)
- [ ] Build + critique pass.

### Step 7 — Appendices as prose  *(mentor #5)*

- [ ] Reproducibility checklist → 2–3 short paragraphs: artifacts and
      locations; determinism (seeds, decoding, 10k bootstrap — re-running
      the analysis is byte-identical, verified); public/local split.
- [ ] Judge-prompt appendix keeps its verbatim blocks; just connective prose.
- [ ] Build + critique pass.

### Step 8 — LLM-use disclosure  *(mentor #6)*

- [ ] Short statement near acknowledgments, precise about three roles:
      (a) Claude Sonnet as judge — methodology, already disclosed;
      (b) LLM assistance in pipeline code and analysis;
      (c) LLM assistance in drafting v1 text; v2 text authored by Khalid.
- [ ] Build + critique pass.

## Phase 4 — Frame and finish

### Step 9 — Abstract + small fixes  *(write the abstract LAST)*

- [ ] **9.1** New numbers, "n = 74", no "empty", one clause on judge-API
      refusals and paired statistics.
- [ ] **9.2** Move the Code/Benchmark/License block out of the abstract into
      a title footnote.
- [ ] **9.3** Small tells: `\date`, "native-author" vs "native author"
      hyphenation, consider inlining the two trivial equations.
- [ ] Build + critique pass.

### Step 10 — Related-work top-up  *(optional, journal-strengthening)*

- [ ] Add 3–5 citations: XSafety, PolygloToxicityPrompts, Aya Red-Teaming,
      M-ALERT and/or RTP-LX. (Claude can pull verified BibTeX — reference
      gathering is not prose.)

### Step 11 — Build, submit, notify

- [ ] Full-paper critique pass (numbers vs data/results, internal
      consistency, remaining LLM-ish patterns).
- [ ] `make figs && make zip` in paper/; verify zip contains only main.tex,
      refs.bib, 3 figure PDFs.
- [ ] Submit as arXiv **replacement** (v2) to arXiv:2605.25420.
- [ ] Email mentor: what changed, incl. the data-integrity fix found and
      corrected; v2 link; no rush given her travel.
- [ ] Commit + push final paper source.
