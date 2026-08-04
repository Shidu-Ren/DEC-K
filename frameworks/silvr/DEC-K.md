# SiLVR integration

This directory contains the complete SiLVR inference and evaluation runtime used by DEC-K.
It is based on [CeeZh/SILVR](https://github.com/CeeZh/SILVR) and retains the upstream MIT
license. `retrieval_selectors.py` adds dense Qwen retrieval, MMR, DEC-K, Adaptive-k, and the
official Adaptive-RAG classifier path. `main.py` exposes these selectors without changing the
SiLVR answer prompt or evaluator.

Upstream source revision: `ee6403d761065237e35514cc50437527a6fefbf6`, with the DEC-K
experiment changes applied from the authors' reproducibility workspace.

Run the paper configuration from the repository root with `python reproduce.py silvr ...`.
