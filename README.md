# croatian-agent-eval

An agentic evaluation of LLM tool use in Croatian. A model plays a salon booking assistant, talks to a customer, calls tools, and either completes the booking correctly or does not. Scenarios are authored natively in Croatian, and a harness-localisation ablation separates "the model is worse in Croatian" from "our English-only scaffolding is worse for Croatian".

**Status:** harness, scenarios, scorer and pre-registration are done. The full matrix has not been run yet, so there are no results in this repo. Traces and the results database will be committed once the full matrix has run, not before.

## Why

Multilingual claims for LLMs are mostly backed by static benchmarks. Agentic behaviour (picking the right tool, filling arguments correctly, resolving "sutra" or "u subotu" to a real date, not finalising too early) is rarely tested in lower-resource languages. Croatian is a good stress case: heavy inflection, diacritics that users often drop, and time idioms that do not map cleanly to English.

## Design

**Domain.** A deterministic salon booking backend with five tools: `search_services`, `get_available_slots`, `create_booking_hold`, `confirm_booking`, `cancel_booking`. The clock is frozen (see `config/run.toml`), so relative dates resolve identically on every run.

**Conditions.**

| Condition | User language | System prompt, tool descriptions, errors |
|-----------|---------------|------------------------------------------|
| `EN_EN` | English | English |
| `HR_EN` | Croatian | English |
| `HR_HR` | Croatian | Croatian |

Only descriptions, enum labels and error messages are translated. JSON parameter keys stay English in every condition, so the scorer is language-independent and schema parsing is not confounded with language comprehension.

**Scenarios.** 36 scenarios in five buckets: happy path, linguistic stress, code-switching, multi-turn, adversarial. Each one is tagged with linguistic axes (diacritics, inflection, time idiom, relative dates, register and dialect). Expectations are constraints, not scripts: required tools, forbidden tools, causally necessary orderings and expected final arguments. Legitimate alternative paths are not penalised.

**Scoring.** Scoring reads saved traces and never calls a model, so a scorer bug costs a re-score rather than a re-run.

- Primary metric: per-field argument accuracy on the final booking.
- Also reported: scenario completion, tool correctness, step efficiency against a hand-derived minimum, language drift (detected with `lingua`), and a failure-cause taxonomy (`premature_finalize`, `bad_arg_date`, `bad_arg_entity`, `hallucinated_tool`, and so on).
- A judge model from a different provider than the models under test scores naturalness. It is only reported alongside Cohen's kappa against blind human labels, and demoted to "indicative only" if kappa is below 0.6.

**Measurement integrity.**

- Infrastructure failures (429, 5xx, network) are retried. A bad model response is never retried.
- Tool dispatch never repairs bad input.
- Runs that end in a provider error are excluded and counted, nothing else is.
- Models are pinned to dated snapshots in `config/models.lock.json`, never to `-latest` aliases.

## Pre-registration

Hypotheses, the single primary comparison and the analysis plan were committed before any model was run: [`report/preregistration.md`](report/preregistration.md). The primary test is per-field argument accuracy, `HR_HR` vs `EN_EN`, paired by scenario, Wilcoxon signed-rank at alpha = 0.05. Everything else is labelled exploratory. A null result will be reported as a null result.

## Quickstart

```bash
cp .env.example .env     # add MISTRAL_API_KEY and GEMINI_API_KEY
make install
make test
make validate            # structural check of the eval set
make smoke               # check models are reachable, resolve and pin snapshots
make run                 # resumable; optional MODEL=, COND=, LIMIT=
make score
make judge
```

The tested models are served through Mistral's API; the judge runs on Gemini. A free API tier is enough for the pilot, but note that some tiers grant zero quota on premier models, in which case the runner fails fast with a clear message instead of retrying.

## Layout

```
config/            frozen run configuration and model lock
report/            pre-registration and report
src/hreval/
  domain/          deterministic booking store, localised tool specs
  harness/         agent loop and prompts
  scenarios/       schema, eval set (JSONL), validator
  scoring/         deterministic scorer, judge, human agreement
  providers.py     model API client with retry discipline
  runner.py        resumable matrix runner
tests/             offline tests using a scripted provider
traces/ results/   populated after the full run
```

## Limitations

- 36 scenarios detect large effects only. Null results mean "no difference detectable at this sample size".
- One domain (booking) and one language pair.
- Scenarios were written by one native speaker.
