# Paper reproduction

## Environment

The selector and evaluation utilities require Python 3.10 or newer.

```bash
conda env create -f environment.yml
conda activate deck
```

The core tests do not require a GPU. Captioning, local embeddings, and Qwen3-VL answering follow
the hardware requirements of their model servers.

## Shared memory and model settings

- non-overlapping 30-second clips;
- Qwen3-VL-8B clip descriptions containing four to eight sentences;
- only transcript utterances present in the clip are supplied to caption generation;
- caption and transcript are indexed together with Qwen3-Embedding-4B;
- question plus answer options form the retrieval query for VideoMME-Long;
- Qwen3-VL-8B is the answer backbone;
- GPT-4o judges M3Bench with one shared Yes/No semantic-correctness prompt;
- VideoMME-Long uses exact option matching.

Keep prompts, decoding, clip boundaries, caption memory, and answer generation fixed within every
paired comparison.

## Main experiment matrix

### M3-Agent

| Method | Configuration |
|---|---|
| Original | released threshold 0.5 and chain-of-retrieval behavior |
| Fixed-4 | `configs/m3_agent/fixed4.yaml` |
| Fixed-5 | `configs/m3_agent/fixed5.yaml` |
| Adaptive-k | `configs/m3_agent/adaptive_k.yaml`, fixed `+2` buffer |
| Adaptive-RAG | `configs/m3_agent/adaptive_rag.yaml`, A/B/C = 2/4/5 |
| DEC-K | `configs/m3_agent/deck.yaml` |

### SiLVR

| Method | Configuration |
|---|---|
| Fixed-7 | `configs/silvr/fixed7.yaml` |
| Fixed-8 | `configs/silvr/fixed8.yaml` |
| Adaptive-k | `configs/silvr/adaptive_k.yaml`, fixed `+5` buffer |
| Adaptive-RAG | `configs/silvr/adaptive_rag.yaml`, A/B/C = 2/7/8 |
| DEC-K | `configs/silvr/deck.yaml` |

Adaptive-k buffers and Adaptive-RAG budget mappings are selected once to align the average clip
budget within each framework and are then held fixed across Robot, Web, and VideoMME-Long.

## Ablations

- Evidence ordering: relevance Fixed-7 versus MMR Fixed-7 on all three SiLVR benchmarks.
- Component analysis: relevance/MMR ordering crossed with fixed/calibrated depth on SiLVR +
  VideoMME-Long.
- Structured candidate construction: max-node versus full clip aggregation on M3-Agent.
- Lambda sensitivity: `0.70, 0.73, 0.76, 0.79, 0.82, 0.85, 0.88, 0.91` on SiLVR +
  VideoMME-Long.
- Returned-depth analysis: DEC-K and Adaptive-k on the three SiLVR benchmarks.

Expected aggregate outputs are versioned in `results/`. A reproduction is complete only when the
question count and realized average clip count also match; accuracy alone is insufficient.

## Generic run matrix

`pipelines/run_paper_matrix.py` accepts a YAML manifest and executes selection, optional answer
generation, and evaluation without a scheduler. Each entry specifies a prepared input JSONL and
one versioned selector config.

```yaml
runs:
  - name: videomme_silvr_deck
    input: ../data/videomme_candidates.jsonl
    config: ../configs/silvr/deck.yaml
    answer:
      model: Qwen/Qwen3-VL-8B-Instruct
      base_url: http://127.0.0.1:8000
    evaluation:
      mode: multiple_choice
```

```bash
python pipelines/run_paper_matrix.py \
  --manifest configs/my_matrix.yaml \
  --output-dir outputs/paper
```

## Verification

```bash
ruff check src tests pipelines
pytest
deck select --config configs/silvr/deck.yaml \
  --input examples/prepared_candidates.jsonl \
  --output outputs/smoke.jsonl
```
