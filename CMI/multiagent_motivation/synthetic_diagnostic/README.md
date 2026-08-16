# Synthetic mismatch diagnostic

This directory is intentionally isolated from the real 60-intervention results.
It contains eight hand-constructed cases with known ground truth:

- four `synthesizer_correction` cases, where the worker becomes wrong after the
  memory intervention and the team coordinator corrects it;
- four `resource_cost` cases, where the final answer remains correct but the
  with-memory path imposes a known verification/tool/review cost.

The existing answer scorer should detect the first category. It cannot detect the
second category unless a team-cost term is added, which is precisely the intended
diagnosis.

## Oracle measurement test

Run this first. It does not call Ollama and feeds the manually specified team
outputs through the same answer scoring function:

```bash
PYTHONPATH=. python multiagent_motivation/synthetic_diagnostic/run_synthetic_diagnostic.py \
  --mode oracle \
  --utility-scorer deterministic \
  --output-dir multiagent_motivation/synthetic_diagnostic/results/oracle
```

Read `results/oracle/summary.json`. The expected interpretation is:

- standard answer-level metric detects the four correction cases;
- standard answer-level metric does not detect the four resource-only cases;
- the resource-adjusted metric detects all eight structural cases.

This proves whether the measurement design, independent of model behavior, can
represent the intended phenomenon.

## Optional Ollama behavior test

After the oracle test, feed the worker reports to the actual Synthesizer:

```bash
PYTHONPATH=. python multiagent_motivation/synthetic_diagnostic/run_synthetic_diagnostic.py \
  --mode ollama \
  --utility-scorer hybrid \
  --config multiagent_motivation/config_local.yaml \
  --output-dir multiagent_motivation/synthetic_diagnostic/results/ollama
```

This tests whether the current local models reproduce the expected correction
behavior. It is separate from the oracle measurement test and must not be merged
with the real benchmark results.
