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

The key go/no-go statistic is `mismatch_rate`. The two directions must be kept
separate: `local_positive_team_nonpositive` indicates downstream misuse or loss
of a locally useful memory; `local_nonpositive_team_positive` indicates that the
Synthesizer corrected a locally harmful/noisy worker result. Do not pool these
directions in the interpretation. With only 20 question clusters, treat a wide
bootstrap interval crossing zero as inconclusive rather than as evidence that
local and team utility are equivalent.
