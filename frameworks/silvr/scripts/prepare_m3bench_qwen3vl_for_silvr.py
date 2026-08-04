#!/usr/bin/env python
from __future__ import annotations

import argparse
import json
import os
import re
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any


M3_ROOT = Path(os.environ.get("M3_AGENT_ROOT", Path(__file__).resolve().parents[2] / "m3_agent"))
SILVR_ROOT = Path(__file__).resolve().parents[1]


def default_caption_root(dataset: str) -> Path:
    if dataset == "web":
        return M3_ROOT / "data/captions/qwen3vl8b_web_m3style_20260614"
    if dataset == "robot":
        return M3_ROOT / "data/captions/qwen3vl8b_robot_m3style_20260617"
    raise ValueError(f"Unknown dataset: {dataset}")


def format_seconds(seconds: float) -> str:
    seconds = max(0, int(round(seconds)))
    hours = seconds // 3600
    minutes = (seconds % 3600) // 60
    secs = seconds % 60
    return f"{hours:02d}:{minutes:02d}:{secs:02d}"


def format_range(clip_idx: int, clip_length: int) -> str:
    start = clip_idx * clip_length
    end = start + clip_length
    return f"{format_seconds(start)} --> {format_seconds(end)}"


def clean_text(value: Any) -> str:
    text = str(value or "").replace("\r\n", "\n").replace("\r", "\n").strip()
    text = re.sub(r"\n{3,}", "\n\n", text)
    return text


def load_json(path: Path) -> Any:
    with path.open("r", encoding="utf-8") as f:
        return json.load(f)


def save_json(path: Path, data: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as f:
        json.dump(data, f, indent=2, ensure_ascii=False)


def iter_caption_records(caption_root: Path):
    per_clip = caption_root / "per_clip"
    if not per_clip.exists():
        raise FileNotFoundError(f"Missing per_clip caption directory: {per_clip}")
    for path in sorted(per_clip.glob("*/*.json")):
        try:
            record = load_json(path)
        except Exception:
            continue
        record["_source_path"] = str(path)
        yield record


def build_video_memories(caption_root: Path, clip_length: int) -> tuple[dict[str, dict[str, Any]], dict[str, Any]]:
    by_video: dict[str, dict[int, dict[str, str]]] = defaultdict(dict)
    status_counts: Counter[str] = Counter()
    total = 0
    for record in iter_caption_records(caption_root):
        total += 1
        status_counts[str(record.get("status") or "unknown")] += 1
        video_id = str(record.get("video_id") or "").strip()
        if not video_id:
            continue
        try:
            clip_idx = int(record.get("clip_idx"))
        except Exception:
            source_path = str(record.get("path") or "")
            match = re.search(r"/(\d+)\.[A-Za-z0-9]+$", source_path)
            if not match:
                continue
            clip_idx = int(match.group(1))
        caption = clean_text(record.get("caption"))
        asr = clean_text(record.get("asr_dialogue"))
        if not caption and not asr:
            continue
        by_video[video_id][clip_idx] = {
            "caption": caption,
            "asr": asr,
            "source_path": str(record.get("_source_path") or ""),
        }

    memories: dict[str, dict[str, Any]] = {}
    for video_id, clips in by_video.items():
        caption_blocks = []
        subtitle_blocks = []
        for clip_idx in sorted(clips):
            clip = clips[clip_idx]
            time_range = format_range(clip_idx, clip_length)
            caption = clip["caption"] or "No visual caption available for this clip."
            caption_blocks.append(f"{time_range}\n[CLIP_{clip_idx}]\n{caption}")
            if clip["asr"]:
                subtitle_blocks.append(f"{time_range}\n[CLIP_{clip_idx} heard dialogue]\n{clip['asr']}")
        memories[video_id] = {
            "caption": "\n\n".join(caption_blocks),
            "subtitle": "\n\n".join(subtitle_blocks),
            "num_caption_clips": len(caption_blocks),
            "num_asr_clips": len(subtitle_blocks),
            "clip_indices": sorted(clips),
        }
    meta = {
        "caption_root": str(caption_root),
        "total_caption_records": total,
        "status_counts": dict(status_counts),
        "videos_with_memory": len(memories),
        "clip_length": clip_length,
    }
    return memories, meta


def flatten_m3bench(
    dataset: str,
    annotation_path: Path,
    caption_root: Path,
    output_path: Path,
    clip_length: int,
    drop_missing_memory: bool,
) -> None:
    annotation = load_json(annotation_path)
    memories, meta = build_video_memories(caption_root, clip_length)
    rows = []
    missing_memory_videos = []

    for video_id, video in annotation.items():
        memory = memories.get(video_id)
        if memory is None:
            missing_memory_videos.append(video_id)
            if drop_missing_memory:
                continue
            memory = {
                "caption": "",
                "subtitle": "",
                "num_caption_clips": 0,
                "num_asr_clips": 0,
                "clip_indices": [],
            }
        for qa in video.get("qa_list", []):
            question_id = qa.get("question_id") or f"{video_id}_Q{len(rows) + 1}"
            row = {
                "global_idx": len(rows),
                "id": question_id,
                "question_id": question_id,
                "video_id": video_id,
                "dataset": f"m3bench_{dataset}",
                "question": qa.get("question", ""),
                "answer": qa.get("answer", ""),
                "reasoning": qa.get("reasoning", ""),
                "task_type": qa.get("type", []),
                "question_type": "open-ended",
                "options": [],
                "caption": memory["caption"],
                "subtitle": memory["subtitle"],
                "num_caption_clips": memory["num_caption_clips"],
                "num_asr_clips": memory["num_asr_clips"],
                "clip_indices": memory["clip_indices"],
            }
            if "timestamp" in qa:
                row["timestamp"] = qa["timestamp"]
            if "before_clip" in qa:
                row["before_clip"] = qa["before_clip"]
            rows.append(row)

    save_json(output_path, rows)
    meta.update(
        {
            "dataset": dataset,
            "annotation_path": str(annotation_path),
            "output_path": str(output_path),
            "videos_in_annotation": len(annotation),
            "qas": len(rows),
            "missing_memory_videos": missing_memory_videos,
            "missing_memory_video_count": len(missing_memory_videos),
            "drop_missing_memory": bool(drop_missing_memory),
        }
    )
    save_json(output_path.with_suffix(output_path.suffix + ".meta.json"), meta)
    print(json.dumps(meta, indent=2, ensure_ascii=False))


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset", required=True, choices=["web", "robot"])
    parser.add_argument("--annotation-path", default="")
    parser.add_argument("--caption-root", default="")
    parser.add_argument("--output-path", default="")
    parser.add_argument("--clip-length", default=30, type=int)
    parser.add_argument("--drop-missing-memory", action="store_true")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    dataset = args.dataset
    annotation_path = Path(args.annotation_path) if args.annotation_path else M3_ROOT / f"data/annotations/{dataset}.json"
    caption_root = Path(args.caption_root) if args.caption_root else default_caption_root(dataset)
    output_path = (
        Path(args.output_path)
        if args.output_path
        else SILVR_ROOT / f"data/m3bench/{dataset}_qwen3vl8b_m3style_caption_asr_silvr.json"
    )
    flatten_m3bench(
        dataset,
        annotation_path,
        caption_root,
        output_path,
        args.clip_length,
        args.drop_missing_memory,
    )


if __name__ == "__main__":
    main()
