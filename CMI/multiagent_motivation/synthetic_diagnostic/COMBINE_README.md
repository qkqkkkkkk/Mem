# Combined diagnostic

This command combines the 60 real same-judge interventions with the 8 isolated
synthetic cases into a new directory. It is for pipeline diagnostics only. The
synthetic cases are not natural benchmark observations and must not be presented
as additional benchmark evidence. Every row has `origin=real` or
`origin=synthetic`.

```bash
PYTHONPATH=. python \
  multiagent_motivation/synthetic_diagnostic/combine_real_and_synthetic.py \
  --real-dir multiagent_motivation/results/qwen_llama_team_rollout5_same_judge \
  --synthetic-dir multiagent_motivation/synthetic_diagnostic/results/oracle \
  --output-dir multiagent_motivation/synthetic_diagnostic/results/combined_diagnostic
```

Outputs:

- `team_interventions.jsonl`: 68 rows with provenance markers.
- `analysis_answer_only/summary.json`: existing answer-only team utility.
- `analysis_structural/summary.json`: synthetic resource costs folded into team utility.
- `combined_summary.json`: both summaries and source paths.

Interpret the real and synthetic rows separately. The useful diagnostic is whether
the answer-only analysis detects the four correction cases and misses the four
resource-only cases; the structural analysis should recover all eight by applying
the explicit synthetic resource-cost field.
