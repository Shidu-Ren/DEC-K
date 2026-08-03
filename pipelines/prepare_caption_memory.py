#!/usr/bin/env python3
from __future__ import annotations

import argparse
from collections import defaultdict
from pathlib import Path
from typing import Any

from deck.io import read_jsonl, write_jsonl


def _video_id(value: dict[str, Any]) -> str:
    for key in ("video_id", "video", "vid"):
        if value.get(key) is not None:
            return str(value[key])
    raise ValueError("Row requires video_id, video, or vid")


def prepare(memory_path: Path, question_path: Path) -> list[dict[str, Any]]:
    memory: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for row in read_jsonl(memory_path):
        video_id = _video_id(row)
        clips = row.get("clips")
        if clips is None:
            clips = [row]
        memory[video_id].extend(clips)

    output = []
    for question in read_jsonl(question_path):
        video_id = _video_id(question)
        documents = []
        for index, clip in enumerate(memory.get(video_id, [])):
            caption = str(clip.get("caption") or clip.get("description") or "").strip()
            transcript = str(clip.get("asr") or clip.get("transcript") or "").strip()
            parts = []
            if caption:
                parts.append(f"Visual description: {caption}")
            if transcript:
                parts.append(f"Transcript: {transcript}")
            documents.append(
                {
                    "clip_id": str(clip.get("clip_id", index)),
                    "start": clip.get("start", index * 30.0),
                    "end": clip.get("end", (index + 1) * 30.0),
                    "text": "\n".join(parts),
                }
            )
        output.append({**question, "video_id": video_id, "documents": documents})
    return output


def main() -> None:
    parser = argparse.ArgumentParser(description="Join 30-second caption-ASR memory with QA")
    parser.add_argument("--memory", type=Path, required=True)
    parser.add_argument("--questions", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    write_jsonl(args.output, prepare(args.memory, args.questions))


if __name__ == "__main__":
    main()
