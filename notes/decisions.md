# Design decisions — somalibench_eval

One paragraph of rationale per non-obvious choice.

## Why these four models?

A controlled comparison across open-weight model families that run locally.
Llama-3.1 and Gemma-2 are English-centric (different families controls for
family-specific effects). Qwen-2.5 is broadly multilingual. Aya-23 is a strong
23-language multilingual comparator, but the Aya 23 report does not list Somali
among its covered languages, so we do not treat it as Somali-specific. All four
are 7–9B-class models and run locally on M-series hardware, making the study
reproducible by anyone with a laptop.

## Why temperature 0, single sample?

Reproducibility. Temperature 0 + fixed seed means anyone re-running this exactly
reproduces the numbers in the paper. v2 can add a temperature sweep to study
variance; v1 nails down the deterministic baseline first.

## Why an English HHH system prompt instead of none?

Two reasons. (1) Several open-weight instruction-tuned models behave noticeably
differently with no system prompt vs a benign one — and "no system prompt" is not
how these models are typically deployed. (2) Using one identical neutral prompt
across all four models removes the system-prompt confound from the cross-model
comparison.

## Why Claude Sonnet as judge?

Three-class classification (refused / complied / unclear) on short text snippets
is well within Sonnet's capability. Opus is overkill at ~5x the cost. Haiku is
slightly cheaper but more confident-but-wrong on edge cases. Anthropic models are
documented as more honest about classification uncertainty than GPT-4-class, which
matters here because the "unclear" category is real and we want it called when it
is real. Native spot-check on 80 verifies the judge.

## Why a stratified random sample for the spot-check (not pure random)?

The seven safety categories are unequal (20 / 15 / 15 / 10 / 15 / 15 / 10) and
some categories are more ambiguous than others. Pure random would under-sample
the small categories. Stratified by category (≈11–12 per category) keeps the
human-verification signal balanced across category types.

## Why bootstrap CIs and not parametric?

Per-model per-language refusal rate is a proportion on n=100. Parametric Wilson
or normal-approx CIs work but bootstrap is robust to category-mix shifts and is
the standard used in the SomaliWeb v1 paper. Same machinery, same convention,
500 resamples + seed 0.

## Why local-only for raw responses?

When a model fails to refuse a harmful prompt, the response is the model's
generation. Aggregate refusal rates are the legitimate research output;
publishing the full unfiltered generations would be redistributing material with
no research benefit beyond what the numbers already convey. Keep raw on disk for
reproducibility; publish counts and rates.

## What v1 explicitly does NOT cover

- No fine-tuning, no mitigation, no defense — pure measurement.
- No jailbreak variants, adversarial wrappers, or prompt injection — v1 reports
  baseline refusal on the prompts as written.
- No temperature sweep, no multi-sample variance estimates.
- No frontier API models (GPT-4, Claude as a *subject* of evaluation, Gemini) —
  v1 sticks to open-weight reproducible-locally.

## What v2 will fix

- Add a 70B-class open-weight model + 1–2 frontier API models for scale comparison.
- Multi-judge (Claude + GPT-4o) → inter-judge agreement number.
- Re-run on the multilingual-safety-probe languages (en/fr/ar/sw/so) at n=100.
- Test whether a mitigation method (MPO / self-distillation) narrows the gap when
  applied to one of the open-weight bases.
- Companion paper.
