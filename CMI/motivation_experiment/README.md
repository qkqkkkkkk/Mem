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
- `summary.json`: aggregate statistics and bootstrap 95% intervals.
- `summary.csv`: compact summary for tables.
- `figures/utility_vs_reliance.png`: `B-U` scatter with quadrant boundaries.
- `figures/relevance_vs_utility.png`: relevance/utility plot with the top-relevance
  band highlighted.

The four quadrants use the top and bottom `--b-quantile` bands (default 0.8 / 0.2)
so tied zero-divergence answers do not silently count as both high and low reliance.

`--no-perturbation` disables the optional CMI perturbed-memory condition. Rows are
still valid for the `B-U` pilot; `stability` is then omitted.
