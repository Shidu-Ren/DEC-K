# M3-Agent integration

This directory contains the complete M3-Agent memorization and control runtime used by DEC-K.
It is based on [ByteDance-Seed/m3-agent](https://github.com/ByteDance-Seed/m3-agent) and retains
the upstream Apache-2.0 license. The control path adds clip-internal memory aggregation,
diversity-aware MMR selection, calibrated depth, and the paper baselines.

Source snapshot: `4750c01fe3b4403e61d4bfbfe201819bc5198783`, with the DEC-K experiment changes
applied from the authors' reproducibility workspace.

Download `M3-Agent-Control` and `M3-Agent-Memorization` into `models/`, or place compatible
symbolic links there. Downloaded M3-Bench memory graphs belong under
`data/memory_graphs/{robot,web,videomme_long}`. Run the paper configuration from the repository
root with `python reproduce.py m3-agent ...`.
