# CMI motivation pilot

This folder contains the pilot suggested in `chatgpt-export_08_12 15_33 Memos 的优化思路.md`.
It reuses the CMI intervention setup and adds a behavioral-reliance measurement:

\[
B_i = D(y_{with(m_i)}, y_{no}),\qquad
U_i = score(y_{with(m_i)}) - score(y_{no}).
\]

For every retrieved memory, the runner records retrieval relevance `R`, behavioral
reliance `B`, causal utility `U`, CMI stability, labels, and the raw intervention
answers. It then reports the `B-U` correlation, the fraction of negative-utility
memories among highly relevant memories, and the four reliance/utility quadrants.
Bootstrap intervals are included for the reported proportions and correlation.

The enhanced pilot keeps separate measurements instead of treating one proxy as
ground truth:

- `embedding_relevance`: query-memory cosine similarity only.
- `hybrid_relevance`: embedding + recency + lexical overlap, matching CMI retrieval.
- `deterministic_utility`, `llm_utility`, and optional `human_utility`.
- `lexical_answer_divergence`, LLM-judged decision change, and optional
  human-judged decision change.
- question-cluster bootstrap confidence intervals for all reported correlations.
- within-label `R-U` correlations to separate group differences from gradients.
- leave-one-question-out logistic models using `R`, `B`, and `R+B` to predict
  positive versus negative utility (neutral interventions are reported and excluded).

Install the repository dependencies first with `pip install -r requirements.txt`.
Here `U` is the repository's deterministic (or hybrid, if the caller changes the
agent/scorer configuration) task score difference, `score(with memory) -
score(without memory)`. It is an answer-level pilot, not a sequential trajectory
return estimate.

## Quick start (no API key)

### Ollama + Qwen (recommended local setup)

Install Ollama, then pull a chat model. The default local config uses your installed Qwen model:

```bash
ollama serve
ollama pull qwen2.5:latest
# Optional, for neural retrieval rather than the deterministic fallback:
ollama pull nomic-embed-text
```

Run the existing CMI pipeline with the local model:

```bash
python scripts/03_run_all_experiments.py \
  --config config/local_ollama.yaml \
  --dataset data/generated/live_pilot_20.jsonl \
  --max_examples 5 \
  --agents cmi
```

Run the motivation pilot with the same model:

```bash
python motivation_experiment/run_pilot.py \
  --config config/local_ollama.yaml \
  --dataset data/generated/live_pilot_20.jsonl \
  --max-examples 5 \
  --output-dir motivation_experiment/results/qwen_smoke
```

The Ollama provider talks to `http://127.0.0.1:11434` and does not require an
OpenAI key. Change `openai.agent_model` in `config/local_ollama.yaml` to any
model shown by `ollama list` (for example `qwen3:8b`).

## Enhanced motivation experiment

Install a real local embedding model first:

```bash
ollama pull nomic-embed-text
```

Run a two-example check using embedding-only relevance, a 50/50 deterministic +
local-Qwen utility judge, and decision-level behavioral reliance:

```bash
python motivation_experiment/run_pilot.py \
  --config config/local_ollama.yaml \
  --dataset causal_locomo_final.jsonl \
  --max-examples 2 \
  --top-k 3 \
  --retrieval-metric hybrid \
  --relevance-metric embedding \
  --require-neural-embeddings \
  --utility-scorer hybrid \
  --behavior-scorer llm_decision \
  --deterministic-weight 0.5 \
  --judge-weight 0.5 \
  --no-perturbation \
  --output-dir motivation_experiment/results/qwen_enhanced_check
```

The comparative judge sees the task, gold behavior, and the two outputs, but not
the memory role label. It scores both outputs and distinguishes a material answer,
recommendation, tool, or action change from a paraphrase. Using the same Qwen model
for generation and judging is inexpensive but not independent; human review is
recommended for claims in a paper.

### Reanalyze an existing run without calling Ollama

The primary correlation intervals use a cluster bootstrap that resamples complete
`example_id` groups. The prediction analysis fits logistic regression inside
leave-one-question-out cross-validation, so interventions from a held-out question
never enter its training fold.

```bash
python motivation_experiment/analyze_results.py \
  --output-dir motivation_experiment/results/qwen_enhanced_check
```

The resulting `summary.json` contains:

- `correlations`: primary question-cluster bootstrap estimates and intervals.
- `naive_correlations`: row-level bootstrap results retained only for comparison.
- `within_label_correlations`: separate `R-U` estimates for each constructed label.
- `utility_sign_prediction`: out-of-fold AUC, average precision, and cluster intervals
  for `R`, `B`, and `R+B`.

### Independent local judge and repeated rollouts

Pull a model from a different family once:

```bash
ollama pull gemma3:4b
```

Then run five independently cached rollouts per condition:

```bash
python motivation_experiment/run_pilot.py \
  --config config/local_ollama_independent_judge.yaml \
  --dataset causal_locomo_final.jsonl \
  --max-examples 20 \
  --top-k 3 \
  --rollouts 5 \
  --generation-temperature 0.3 \
  --require-independent-judge \
  --retrieval-metric hybrid \
  --relevance-metric embedding \
  --require-neural-embeddings \
  --utility-scorer hybrid \
  --behavior-scorer llm_decision \
  --deterministic-weight 0.5 \
  --judge-weight 0.5 \
  --no-perturbation \
  --output-dir motivation_experiment/results/qwen_gemma_rollout5
```

Every generation call includes a distinct rollout identifier in its cache key.
For each intervention, the JSONL records `rollout_utilities`, their mean, standard
deviation, and a rollout bootstrap interval. A multi-rollout run fails early when
the generation temperature is zero, and `--require-independent-judge` fails when
the configured judge and agent model names are identical.

### Diagnose and repair a saturated B measurement

Do this before expanding the dataset when `B` is nearly all one or when the old
judge reports normalization warnings. These commands reuse the existing Qwen
answers: they do not rerun retrieval or generation.

First export a blinded review bundle. The pair-level file combines repeated
rollouts, so 54 rollout warnings in the current fresh result become 14 manual
review units.

```bash
PYTHONPATH=. python motivation_experiment/make_diagnostic_bundle.py \
  --input-dir motivation_experiment/results/qwen_llama_rollout5_fresh
```

Review `diagnostics/judge_warning_pair_review.csv` using only the two answers:
enter 0 / 0.5 / 1 for conclusion and factual change, set whether an action is
applicable, then record its action-change score. This file intentionally excludes
the constructed memory label. Review `diagnostics/anomaly_review.csv` separately:
it contains all harmful items, the harmful-positive-utility outliers, and every
item with a five-rollout utility interval completely below zero.

Next, keep the same independent judge model but replace the ambiguous single
decision score with a three-dimensional rubric. Its overall B is the arithmetic
mean of the applicable conclusion, factual, and action-change scores. The rubric
does not receive the expected answer, scoring criteria, or memory label, so B is
not contaminated by correctness/utility judgment.

```bash
PYTHONPATH=. python motivation_experiment/rejudge_multidim.py \
  --input-dir motivation_experiment/results/qwen_llama_rollout5_fresh \
  --output-dir motivation_experiment/results/qwen_llama_rollout5_multidim \
  --config config/local_ollama_independent_judge.yaml \
  --judge-model llama3:8b
```

This makes one deterministic local judge call for each existing answer pair
(60 interventions x 5 rollouts = about 300 calls) and caches them under the new
output directory. Keep the output directory separate from the original run. The
new `summary.json` has `diagnostic_comparison`: compare old/new B distributions,
the normalization-warning count, and cluster-bootstrap `B-U` correlation before
making any claim about behavioral reliance.

### Human judging

Every run writes `human_annotations.csv`. Annotate these columns on a `[0,1]`
scale:

- `human_score_without`: task correctness without the memory.
- `human_score_with`: task correctness with the memory.
- `human_decision_change`: `0` same final decision, `1` contradictory/reversed
  decision, intermediate values only for a partial material change.

Do not edit IDs or outputs. After annotation, rerun the same command and output
directory so model outputs are read from cache, replacing the two scorer options
with:

```bash
  --utility-scorer human \
  --behavior-scorer human_decision \
  --human-annotations motivation_experiment/results/qwen_enhanced_check/human_annotations.csv
```

The run rewrites `memory_interventions.jsonl`, `summary.json`, and both figures
using human `U` and decision-level `B` while preserving the filled annotation
columns.

From the repository root:

```bash
python motivation_experiment/run_pilot.py \
  --dataset causal_locomo_final.jsonl \
  --max-examples 20 \
  --output-dir motivation_experiment/results/local_20
```

The default local answer generator is deterministic and costs no API calls. A
small generated dataset is useful for a smoke test:

```bash
python motivation_experiment/run_pilot.py \
  --dataset data/generated/live_pilot_20.jsonl \
  --max-examples 5 \
  --output-dir motivation_experiment/results/smoke
```

To use the hosted OpenAI client instead, set `OPENAI_API_KEY` and pass
`--config config/live_pilot.yaml --use-api`. The number of independent samples per
intervention can be increased with `--rollouts`; local Ollama calls are free but
still increase runtime approximately linearly.

## Outputs

- `memory_interventions.jsonl`: one row per `(example, retrieved memory)`.
- `human_annotations.csv`: blinded human-scoring template and optional input.
- `summary.json`: aggregate statistics and bootstrap 95% intervals.
- `summary.csv`: compact summary for tables.
- `figures/utility_vs_reliance.png`: `B-U` scatter with quadrant boundaries.
- `figures/relevance_vs_utility.png`: relevance/utility plot with the top-relevance
  band highlighted.

The four quadrants use the top and bottom `--b-quantile` bands (default 0.8 / 0.2)
so tied zero-divergence answers do not silently count as both high and low reliance.

`--no-perturbation` disables the optional CMI perturbed-memory condition. Rows are
still valid for the `B-U` pilot; `stability` is then omitted.
