# Data preparation

This repository does not redistribute benchmark videos, model weights, API credentials, or
third-party memory graphs. Obtain each resource from its official source and follow its license.

## Benchmarks

### M3Bench Robot and Web

- Dataset: [ByteDance-Seed/M3-Bench](https://huggingface.co/datasets/ByteDance-Seed/M3-Bench)
- Host system: [ByteDance-Seed/m3-agent](https://github.com/ByteDance-Seed/m3-agent)
- Robot evaluation: 100 videos and 1,276 open-ended questions.
- M3-Agent Web evaluation: the complete released 3,214-question structured-memory set.
- SiLVR Web evaluation: a frozen 906-video, 3,171-question subset. Fourteen YouTube sources
  were private, removed, or otherwise unavailable when caption memory was reconstructed.

Use the same frozen question manifest for every paired selector comparison. Missing videos
must not be silently replaced by questions without memory.

### VideoMME-Long

- Dataset: [Video-MME](https://video-mme.github.io/home_page.html)
- Evaluation: 300 long videos and 900 multiple-choice questions.

## Caption-ASR memory schema

`pipelines/prepare_caption_memory.py` accepts JSONL memory rows in either form:

```json
{"video_id":"v1","clips":[{"clip_id":"0","start":0,"end":30,"caption":"...","asr":"..."}]}
```

or one clip per row:

```json
{"video_id":"v1","clip_id":"0","start":0,"end":30,"caption":"...","transcript":"..."}
```

Questions require `video_id`, `question`, and optionally `options`, `ground_truth`, and a stable
`question_id`. The preparation script emits a `documents` list that can be passed to `deck
embed`.

## Prepared candidate schema

The model-independent selector consumes one JSON object per question:

```json
{
  "question_id": "v1_q1",
  "question": "What happened after the person entered?",
  "candidates": [
    {
      "clip_id": "7",
      "relevance": 0.63,
      "vector": [0.1, 0.2],
      "text": "Visual description: ...\nTranscript: ...",
      "start": 210,
      "end": 240
    }
  ]
}
```

For structured memory, replace `candidates` with query-scored `nodes`. Every node requires
`node_id`, `clip_id`, `relevance`, `vector`, and `memory_type` (`episodic` or `semantic`).
