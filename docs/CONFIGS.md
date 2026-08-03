# Configuration reference

Every selector run is controlled by one YAML file.

| Field | Meaning |
|---|---|
| `method` | `deck`, `fixed`, `mmr_fixed`, `relevance_deck`, `adaptive_k`, or `adaptive_rag` |
| `candidate_pool` | Number of relevance-ranked candidates exposed to the selector |
| `min_k`, `max_k` | Minimum and maximum returned evidence depth |
| `start_k` | First adjacent boundary considered by DEC-K |
| `lambda` | Relevance weight in sequential MMR |
| `temporal_output` | Restore selected clips to temporal order before answering |
| `fixed_k` | Requested depth for fixed relevance/MMR baselines |
| `adaptive_extra` | Fixed post-cutoff buffer used by Adaptive-k |
| `adaptive_budgets` | Mapping from Adaptive-RAG classes A/B/C to clip counts |
| `structured_memory` | M3-Agent clip-level node aggregation parameters |

DEC-K observes at most `max_k + 1` MMR marginals. The additional item is used only to evaluate
the boundary after `max_k`; it is not returned.

The paper configurations are fixed across all three benchmarks within each framework. There is
no benchmark-specific tuning.
