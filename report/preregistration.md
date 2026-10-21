# Pre-registration

Committed **before** any model was run against the eval set. The git history is
the evidence: this file's commit precedes the first commit containing results.

The point of writing this first is that it is much easier to find a story in
data you have already seen. Fixing the hypotheses, the primary comparison and
the analysis plan in advance means the headline finding is whatever the data
says, including "no difference".

## Hypotheses

**H1 (language effect).** Task completion in Croatian (`HR_HR`) is lower than in
English (`EN_EN`) on the same scenarios and the same model.

**H2 (harness localisation).** Part of any Croatian deficit is attributable to
the harness rather than the model: `HR_HR` outperforms `HR_EN`, where the user
writes Croatian but the system prompt, tool descriptions and error messages are
English.

**H3 (tier interaction).** Any Croatian deficit is larger for smaller models —
i.e. the gap between `HR_HR` and `EN_EN` widens as the tier drops.

**H4 (linguistic axes).** Failures concentrate in the Croatian-specific stress
axes — time idiom, inflection, diacritic stripping — rather than being spread
uniformly across buckets.

**H5 (token inflation).** Croatian consumes more tokens than English for the
same task, making Croatian deployments more expensive at identical quality.

## Primary comparison

Exactly one, fixed in advance:

> **Per-field argument accuracy, `HR_HR` vs `EN_EN`, on the largest pinned
> tier, paired by scenario, tested with Wilcoxon signed-rank at α = 0.05.**

Per-field rather than per-scenario because at n = 36 scenarios, per-scenario
completion is underpowered; field-level comparison yields several hundred paired
observations. Everything else reported in this project — H2 through H5, every
other tier, every other condition pair, every per-axis breakdown — is
**exploratory** and will be labelled as such. No multiplicity correction is
applied to exploratory results because no inferential claim is made from them.

## Analysis plan

- Completion rates reported with **Wilson 95% CIs**.
- Paired condition comparisons use the per-scenario success *rate* across the 3
  seeded repeats (0, ⅓, ⅔, 1) with **Wilcoxon signed-rank**, not binary McNemar.
- Step-efficiency deltas reported with **bootstrap 95% CIs** (10,000 resamples).
- Judge-derived metrics are reported only alongside Cohen's κ against blind
  human labels, measured on a held-out 40 of 60 labelled items. **If κ < 0.6,
  both judge metrics are demoted to "indicative only"** and no claim rests on
  them.
- Runs with `stop_reason = provider_error` are excluded from scoring and their
  count is reported. No other exclusions.

## Power, stated up front

At n = 36 paired scenarios this design detects **large effects only**. A paired
McNemar test distinguishing 0.85 from 0.70 yields roughly 7 discordant pairs and
does not reach significance. Null results will therefore be reported as
*"no difference detectable at this sample size"*, never as *"no difference"*.

## Pre-committed interpretation of a null result

If Croatian performance matches English, that is a publishable finding and will
be reported as the headline without reframing: the first public agentic evidence
that these multilingual claims extend to a lower-resource EU language. This
sentence exists so that a null cannot later be quietly spun into something more
dramatic.

## What would falsify the method itself

- If `HR_EN` ≈ `HR_HR` on every tier, the harness-localisation factor carries no
  signal and should be reported as a negative methodological result.
- If κ < 0.6 for every candidate judge, LLM-judged naturalness is not measurable
  at this scale and the project reports deterministic metrics only.
