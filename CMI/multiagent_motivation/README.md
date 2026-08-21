# Multi-agent motivation experiment

This experiment tests the structural hypothesis that local worker utility and
downstream team utility need not have the same sign:

`U_local(m) = S(worker_with_memory) - S(worker_without_memory)`

`U_team(m) = S(synthesizer(worker_with_memory_report)) - S(synthesizer(worker_without_memory_report))`

It is a minimal two-agent setup. Existing worker outputs are reused; the new
Synthesizer sees only the task and the worker report. It never sees the memory
content, construction label, or expected answer. This prevents the team layer
from directly reading the intervention.

## Prerequisites

The source run must contain `memory_interventions.jsonl` with five
`no_memory_outputs` and five `with_memory_outputs` per intervention. The default
local config uses `llama3:8b` as Synthesizer and `qwen2.5:14b` as the independent
team utility judge. Pull both models first if they are not already present.

## Smoke test

Run from the repository root after activating the project environment:

```bash
PYTHONPATH=. python multiagent_motivation/run_team_pilot.py \
  --source-dir motivation_experiment/results/qwen_llama_rollout5_fresh \
  --dataset causal_locomo_final.jsonl \
  --max-interventions 1 \
  --rollouts 1 \
  --config multiagent_motivation/config_local.yaml \
  --output-dir multiagent_motivation/results/smoke \
  --utility-scorer hybrid \
  --worker-model qwen2.5:7b \
  --require-independent-roles
```

Analyze the smoke output:

```bash
PYTHONPATH=. python multiagent_motivation/analyze_team_results.py \
  --input-dir multiagent_motivation/results/smoke
```

## Full run

This reuses all 60 existing interventions and all five worker rollouts. It adds
600 Synthesizer generations and 600 team-utility judge calls. The two conditions
are intentionally non-paired in the same way as the original local utility
estimate: this estimates the difference of two Monte Carlo means.

```bash
PYTHONPATH=. python multiagent_motivation/run_team_pilot.py \
  --source-dir motivation_experiment/results/qwen_llama_rollout5_fresh \
  --dataset causal_locomo_final.jsonl \
  --config multiagent_motivation/config_local.yaml \
  --output-dir multiagent_motivation/results/qwen_llama_team_rollout5 \
  --utility-scorer hybrid \
  --worker-model qwen2.5:7b \
  --require-independent-roles

PYTHONPATH=. python multiagent_motivation/analyze_team_results.py \
  --input-dir multiagent_motivation/results/qwen_llama_team_rollout5 \
  --output-dir multiagent_motivation/results/qwen_llama_team_rollout5/analysis
```

## Main outputs

- `team_interventions.jsonl`: worker reports, team answers, rollout scores,
  `local_utility`, and `team_utility` for every intervention.
- `analysis/summary.json`: cluster-bootstrap mismatch rate, the 2x2 sign table,
  local/team utility correlation, and task-family stratification.
- `analysis/mismatch_cases.jsonl`: cases for manual inspection.

Generate figures for an existing run (no LLM calls):

```bash
MPLBACKEND=Agg PYTHONPATH=. python multiagent_motivation/plot_team_results.py \
  --input-dir multiagent_motivation/results/qwen_llama_team_rollout5_same_judge
```

This writes `analysis/figures/utility_scatter.png`,
`analysis/figures/sign_contingency.png`, and
`analysis/figures/mismatch_by_task_family.png`.

The key go/no-go statistic is `mismatch_rate`. The two directions must be kept
separate: `local_positive_team_nonpositive` indicates downstream misuse or loss
of a locally useful memory; `local_nonpositive_team_positive` indicates that the
Synthesizer corrected a locally harmful/noisy worker result. Do not pool these
directions in the interpretation. With only 20 question clusters, treat a wide
bootstrap interval crossing zero as inconclusive rather than as evidence that
local and team utility are equivalent.

## Same-judge validation

The first full run uses the original Llama judge for `U_local` and a Qwen judge
for `U_team`. To rule out scorer-family differences, rescore only the existing
local worker answers with the same Qwen judge already used for the team scores.
The team scores are reused, so this adds about 600 judge calls and no generation.

```bash
PYTHONPATH=. python multiagent_motivation/rescore_same_judge.py \
  --input-dir multiagent_motivation/results/qwen_llama_team_rollout5 \
  --dataset causal_locomo_final.jsonl \
  --config multiagent_motivation/config_local.yaml \
  --output-dir multiagent_motivation/results/qwen_llama_team_rollout5_same_judge \
  --rollouts 5 \
  --utility-scorer hybrid

PYTHONPATH=. python multiagent_motivation/analyze_team_results.py \
  --input-dir multiagent_motivation/results/qwen_llama_team_rollout5_same_judge \
  --output-dir multiagent_motivation/results/qwen_llama_team_rollout5_same_judge/analysis
```

Compare the new `analysis/summary.json` to the original. Also run a sensitivity
check with a pre-declared practical utility tolerance, for example
`--utility-epsilon 0.05`, rather than treating tiny values around zero as robust
sign changes.

## R/B/U substitution experiment

`analyze_rbu_substitution.py` turns the already-evaluated interventions into a
per-question memory-selection experiment. Each selector may choose exactly one
of the same retrieved candidate memories, so R, B, R+B, and the U oracle have
the same memory budget. `U_oracle` selects the candidate with the highest
observed `team_utility`; it is an evaluation upper bound, not a deployable
method.

Run it on a completed team run without making further LLM calls:

```bash
PYTHONPATH=. python multiagent_motivation/analyze_rbu_substitution.py \
  --input-dir multiagent_motivation/results/qwen_llama_team_rollout5_same_judge \
  --output-dir multiagent_motivation/results/qwen_llama_team_rollout5_same_judge/rbu_analysis
```

The default definitions are `R=hybrid_relevance`, `B=behavioral_reliance` from
the worker intervention, and `U=team_utility`. `R+B` normalizes R and B within
each question before adding them, so neither metric wins merely because of its
numeric scale. For a stricter downstream version in which B is measured from
the synthesizer's paired answers, pass:

```bash
  --b-field team_behavioral_reliance
```

This uses a derived lexical divergence from the stored `team_with_memory_outputs`
and `team_no_memory_outputs`; it does not make a new model call.

The primary result is `conditions["R+B"].score_gap_vs_u_oracle` in
`rbu_summary.json`. A negative estimate whose bootstrap interval remains below
zero means the R+B selector underperformed observed causal utility under the
same one-memory budget. Report this alongside `selection_match_rate_to_u_oracle`
and the ordered errors in `rbu_regret_cases.jsonl`; do not interpret the oracle
as a train-free deployment baseline.

To test whether the worker's own intervention utility transfers better than R/B
to the final team outcome, use the same analysis with:

```bash
  --u-field local_utility
```

The selector then uses worker-level U, while `mean_team_score` and
`score_gap_vs_u_oracle` remain downstream team outcomes. State the selected U
field explicitly in the paper; the output records it under `fields.u`.

Generate motivation-style plots from the R/B/U output:

```bash
CMI/.venv/bin/python multiagent_motivation/plot_rbu_results.py \
  --input-dir multiagent_motivation/results/qwen_llama_team_rollout5_same_judge/rbu_analysis \
  --output-dir multiagent_motivation/results/qwen_llama_team_rollout5_same_judge/rbu_analysis/figures
```

This writes selector performance, regret versus the U oracle, R/B-versus-U
scatter plots, positive-versus-negative-U prediction (ROC AUC and average
precision), and per-question selector scores. The scatter plot uses the
intervention labels (`useful`, `harmful`, `irrelevant`) as colors, matching the
motivation-experiment figures. It only reads JSON outputs and does not call a
model.
