# Host-system integration

DEC-K requires only clip identifiers, query relevance, and one redundancy vector per clip. The
answer model and memory-building code remain in the host project.

## SiLVR

SiLVR already produces `docs`, relevance `scores`, and document embedding `vectors`. Replace the
final top-k slice with:

```python
from deck.integrations import select_silvr_documents

selected_docs, trace = select_silvr_documents(
    docs,
    scores,
    vectors,
    candidate_pool=200,
    min_k=2,
    max_k=8,
    start_k=2,
    lambda_=0.85,
)
```

Pass `selected_docs` through SiLVR's existing temporal formatting and answer prompt. The helper
already restores temporal order, but preserving the host formatter keeps prompt text identical
between methods.

## M3-Agent

The M3-Agent retriever returns query-scored memory-node hits. Export each hit as:

```python
{
    "node_id": node_id,
    "clip_id": clip_id,
    "relevance": node_score,
    "vector": node_embedding,
    "memory_type": node.type,
    "text": node_text,
    "start": clip_start,
    "end": clip_end,
}
```

Then call:

```python
from deck.config import ExperimentConfig
from deck.integrations import select_m3_agent_nodes

config = ExperimentConfig.from_yaml("configs/m3_agent/deck.yaml")
selected_clip_ids, trace = select_m3_agent_nodes(node_hits, config=config)
```

Use the returned clip IDs in M3-Agent's existing memory translation and control loop. Ordinary
search uses DEC-K; entity lookups and framework-specific searches remain unchanged. For reported
clip budgets, average only non-empty ordinary memory-search calls.

## Candidate construction details

For every clip, M3-Agent candidate construction:

1. retains the eight highest-scoring nodes in host retrieval order;
2. greedily clusters each node against score-weighted cluster centroids with cosine threshold
   0.85;
3. scores up to four cluster leaders with weights `(1, .35, .15, .05)`;
4. adds the cross-type bonus only when the leading three clusters contain both episodic and
   semantic records; and
5. combines leader centroids into one redundancy vector.

The pure-Python implementation is in `src/deck/candidates.py` and has no dependency on M3-Agent
internals.

## Optional upstream checkouts

```bash
python pipelines/setup_backends.py --output third_party
```

This clones the public host projects for reference. It does not modify them or install model
weights.
