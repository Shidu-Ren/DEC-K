<h1 align="center">DEC-K</h1>

<p align="center"><b>Diverse Evidence with Calibrated K for Long-Term Memory Multimodal Agents</b></p>

<p align="center">
  <a href="https://github.com/Shidu-Ren/DEC-K/actions/workflows/tests.yml"><img alt="Tests" src="https://github.com/Shidu-Ren/DEC-K/actions/workflows/tests.yml/badge.svg"></a>
  <a href="https://www.python.org/"><img alt="Python 3.10+" src="https://img.shields.io/badge/python-3.10%2B-3776AB.svg"></a>
  <a href="LICENSE"><img alt="Apache-2.0" src="https://img.shields.io/badge/license-Apache--2.0-3DA639.svg"></a>
  <a href="https://github.com/Shidu-Ren/DEC-K"><img alt="Code" src="https://img.shields.io/badge/code-available-6F42C1.svg"></a>
</p>

<p align="center">
  <a href="https://github.com/Shidu-Ren">Shidu Ren</a> ·
  Rongcheng Tu · Hang Zhou · Xiao Luo
</p>

<p align="center"><i>Paper link will be added after the arXiv record is available.</i></p>

![Why long-term memory retrieval needs evidence diversity and calibrated depth](assets/teaser.png)

## TL;DR

Long-term agent memory contains repeated observations, related records, and weak tail
evidence. Plain top-k retrieval can therefore return several views of the same event, while a
single fixed depth can be too short for one question and distracting for another. DEC-K is a
training-free retrieval layer that:

1. constructs one candidate per evidence clip, including query-aware aggregation for
   structured memories;
2. orders candidates with sequential maximal marginal relevance (MMR); and
3. selects a query-specific prefix from length-calibrated drops in the MMR marginal sequence.

DEC-K leaves memory construction and answer generation unchanged. It trains no selector and
adds no foundation-model call.

![DEC-K method overview](assets/method.png)

## Main Results

All comparisons below use matched realized retrieval budgets. Accuracy and clip counts are
reported as percentages and mean clips per ordinary retrieval call, respectively.

### M3-Agent

| Method | Robot | Web | VideoMME-Long | Average | Mean clips |
|---|---:|---:|---:|---:|---:|
| Original | 30.7 | 48.9 | 53.0 | 44.2 | 1.8 |
| Fixed-4 | 38.4 | 56.4 | 56.9 | 50.6 | 4.0 |
| Fixed-5 | 38.9 | 56.7 | 57.6 | 51.0 | 4.9 |
| Adaptive-k | 38.8 | 55.3 | 57.9 | 50.7 | 4.1 |
| Adaptive-RAG | 38.3 | 55.1 | 56.8 | 50.1 | 4.1 |
| **DEC-K** | **40.0** | **57.3** | **58.4** | **51.9** | **4.1** |

### SiLVR

| Method | Robot | Web | VideoMME-Long | Average | Mean clips |
|---|---:|---:|---:|---:|---:|
| Fixed-7 | 40.8 | 55.9 | 56.9 | 51.2 | 7.0 |
| Fixed-8 | 41.3 | **57.6** | 58.0 | 52.3 | 8.0 |
| Adaptive-k | 41.4 | 56.1 | 56.2 | 51.2 | 7.0 |
| Adaptive-RAG | 41.0 | 56.3 | 57.2 | 51.5 | 7.1 |
| **DEC-K** | **42.1** | 56.9 | **58.6** | **52.5** | **7.1** |

The complete per-benchmark table, component ablations, lambda sweep, returned-depth counts,
and case-study records are in [`results/`](results/).

## Method

For clip candidate `i`, DEC-K combines min-max-scaled query relevance with redundancy to the
selected set:

$$
g_t(i)=\lambda\bar r_i-(1-\lambda)\max_{j\in S_{t-1}}\cos(z_i,z_j).
$$

The selected MMR values form a marginal sequence `g1, g2, ...`. For every admissible boundary,
DEC-K evaluates

$$
a_k=\left(\frac{g_k-g_{k+1}}{g_1}\right)^{1/k},
\qquad
K^*=\arg\max_k a_k.
$$

Only strict positive drops are eligible. If no valid boundary exists, DEC-K returns the largest
feasible depth. One additional MMR value is observed to score the `K_max` boundary; this
look-ahead clip is never returned to the answer model.

## Installation

The selector itself is lightweight and runs on CPU:

```bash
git clone https://github.com/Shidu-Ren/DEC-K.git
cd DEC-K
python -m venv .venv
source .venv/bin/activate
pip install -e .
```

Install the optional local embedding stack when using Qwen3-Embedding-4B directly:

```bash
pip install -e ".[models]"
```

## Quick Start

Run DEC-K on the included prepared example:

```bash
deck select \
  --config configs/silvr/deck.yaml \
  --input examples/prepared_candidates.jsonl \
  --output outputs/demo-selected.jsonl
```

The input row contains a question and clip candidates with query relevance and cached
embeddings. The output preserves the question, selected evidence, selected depth, MMR trace,
and every calibrated boundary score.

Python users can call the selector directly:

```python
import numpy as np
from deck import Candidate, select_deck

candidates = [
    Candidate("0", 0.91, np.array([1.0, 0.0]), text="first memory"),
    Candidate("1", 0.88, np.array([0.99, 0.01]), text="near duplicate"),
    Candidate("2", 0.80, np.array([0.0, 1.0]), text="complementary memory"),
]

result = select_deck(
    candidates,
    candidate_pool=200,
    min_k=2,
    max_k=8,
    start_k=2,
    lambda_=0.85,
)
print(result.selected_k, [item.clip_id for item in result.selected])
```

## End-to-End Caption-Memory Pipeline

The commands below reproduce the model-independent pipeline. They contain no cluster scheduler
assumptions.

### 1. Join 30-second captions and transcripts with questions

```bash
python pipelines/prepare_caption_memory.py \
  --memory data/caption_asr.jsonl \
  --questions data/questions.jsonl \
  --output data/qa_documents.jsonl
```

### 2. Encode queries and documents

```bash
deck embed \
  --backend local \
  --model Qwen/Qwen3-Embedding-4B \
  --input data/qa_documents.jsonl \
  --output data/qa_candidates.jsonl
```

An OpenAI-compatible embedding server can be used with `--backend openai --base-url ...`.

### 3. Select evidence

```bash
deck select \
  --config configs/silvr/deck.yaml \
  --input data/qa_candidates.jsonl \
  --output outputs/deck-selected.jsonl
```

### 4. Answer with an OpenAI-compatible local Qwen endpoint

```bash
deck answer \
  --model Qwen/Qwen3-VL-8B-Instruct \
  --base-url http://127.0.0.1:8000 \
  --input outputs/deck-selected.jsonl \
  --output outputs/deck-predictions.jsonl
```

### 5. Evaluate

```bash
# VideoMME-Long
deck evaluate --mode multiple_choice \
  --input outputs/deck-predictions.jsonl \
  --metrics outputs/metrics.json

# M3Bench open-ended QA with the same judge prompt used by all methods
deck evaluate --mode judge \
  --judge-model gpt-4o \
  --input outputs/deck-predictions.jsonl \
  --output outputs/deck-judged.jsonl \
  --metrics outputs/metrics.json
```

Credentials are read only from the environment variable named by `--api-key-env`. No API key,
cookie, model weight, or private dataset is stored in this repository.

## Reproducing the Paper

The exact selector settings are versioned in [`configs/`](configs/):

| Framework | DEC-K setting | Adaptive-k | Adaptive-RAG |
|---|---|---|---|
| M3-Agent | `N=200`, `K=2..5`, `K_start=1`, `lambda=.85` | `+2` clips | `A/B/C = 2/4/5` |
| SiLVR | `N=200`, `K=2..8`, `K_start=2`, `lambda=.85` | `+5` clips | `A/B/C = 2/7/8` |

See [`docs/REPRODUCTION.md`](docs/REPRODUCTION.md) for benchmark setup and the complete run
matrix. Host-specific hooks are documented in [`docs/INTEGRATION.md`](docs/INTEGRATION.md).

## Repository Layout

```text
DEC-K/
├── src/deck/              # selector, candidate construction, baselines, CLI
├── configs/               # exact paper settings and ablations
├── pipelines/             # generic preparation and experiment runners
├── results/               # paper results in machine-readable form
├── tests/                 # deterministic unit and CLI tests
├── docs/                  # data, integration, and reproduction guides
├── examples/              # small prepared smoke-test input
└── assets/                # paper figures used by this README
```

## Tests

```bash
pip install -e ".[dev]"
ruff check src tests pipelines
pytest
```

## Citation

```bibtex
@misc{ren2026deck,
  title        = {DEC-K: Diverse Evidence with Calibrated K for Long-Term Memory Multimodal Agents},
  author       = {Ren, Shidu and Tu, Rongcheng and Zhou, Hang and Luo, Xiao},
  year         = {2026},
  note         = {Preprint},
  howpublished = {\url{https://github.com/Shidu-Ren/DEC-K}}
}
```

## Acknowledgments

DEC-K is evaluated with [M3-Agent](https://github.com/ByteDance-Seed/m3-agent) and
[SiLVR](https://github.com/CeeZh/SILVR). Baselines use
[Adaptive-k Retrieval](https://github.com/megagonlabs/adaptive-k-retrieval) and
[Adaptive-RAG](https://github.com/starsuzi/Adaptive-RAG). See
[`THIRD_PARTY.md`](THIRD_PARTY.md) for licenses and attribution.
