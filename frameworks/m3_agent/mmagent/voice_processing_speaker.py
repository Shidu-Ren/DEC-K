"""Speaker-aware wrapper for audio diarization/tracking outputs.

This module keeps the original voice pipeline untouched and adds an
experimental post-processing layer for audio-only runs:
- bind each utterance to a stable speaker tag (`<voice_x>`)
- estimate speaker confidence from node-level voice embeddings
- optionally merge adjacent same-speaker turns to reduce diarization noise
"""

from __future__ import annotations

import os
from typing import Dict, List

import numpy as np

from .voice_processing import process_voices


def _parse_mmss(ts: str) -> int:
    mm, ss = ts.split(":")
    return int(mm) * 60 + int(ss)


def _to_mmss(total_seconds: int) -> str:
    total_seconds = max(0, int(total_seconds))
    mm = total_seconds // 60
    ss = total_seconds % 60
    return f"{mm:02d}:{ss:02d}"


def _cosine_similarity(a: List[float], b: List[float]) -> float:
    va = np.asarray(a, dtype=np.float32)
    vb = np.asarray(b, dtype=np.float32)
    if va.size == 0 or vb.size == 0:
        return 0.0
    na = np.linalg.norm(va)
    nb = np.linalg.norm(vb)
    if na <= 1e-12 or nb <= 1e-12:
        return 0.0
    return float(np.dot(va, vb) / (na * nb))


def _confidence_from_similarity(sim: float) -> float:
    # Map cosine [-1, 1] to confidence [0, 1]
    return max(0.0, min(1.0, (sim + 1.0) / 2.0))


def _speaker_centroid(video_graph, voice_node_id: int) -> List[float]:
    node = video_graph.nodes.get(voice_node_id)
    if node is None or node.type != "voice" or not node.embeddings:
        return []
    emb = np.asarray(node.embeddings, dtype=np.float32)
    return np.mean(emb, axis=0).tolist()


def _attach_speaker_meta(video_graph, id2voices: Dict[int, List[dict]]) -> Dict[int, List[dict]]:
    out: Dict[int, List[dict]] = {}
    for voice_node_id, segments in id2voices.items():
        centroid = _speaker_centroid(video_graph, voice_node_id)
        speaker_tag = f"<voice_{voice_node_id}>"
        enriched = []
        for seg in segments:
            new_seg = dict(seg)
            emb = new_seg.get("embedding", [])
            sim = _cosine_similarity(emb, centroid) if centroid else 0.0
            new_seg["speaker_id"] = speaker_tag
            new_seg["speaker_confidence"] = round(_confidence_from_similarity(sim), 4)
            enriched.append(new_seg)
        out[voice_node_id] = enriched
    return out


def _merge_same_speaker_turns(
    id2voices: Dict[int, List[dict]],
    merge_gap_seconds: float,
    min_speaker_confidence: float,
) -> Dict[int, List[dict]]:
    merged_all: Dict[int, List[dict]] = {}
    gap_limit = max(0.0, float(merge_gap_seconds))

    for voice_node_id, segments in id2voices.items():
        if not segments:
            merged_all[voice_node_id] = []
            continue

        valid = []
        for seg in segments:
            try:
                s = _parse_mmss(seg["start_time"])
                e = _parse_mmss(seg["end_time"])
                if e <= s:
                    continue
                conf = float(seg.get("speaker_confidence", 0.0))
                if conf < float(min_speaker_confidence):
                    continue
                row = dict(seg)
                row["_start_sec"] = s
                row["_end_sec"] = e
                valid.append(row)
            except Exception:
                continue

        if not valid:
            merged_all[voice_node_id] = []
            continue

        valid.sort(key=lambda x: (x["_start_sec"], x["_end_sec"]))
        merged = [valid[0]]
        for curr in valid[1:]:
            prev = merged[-1]
            gap = curr["_start_sec"] - prev["_end_sec"]
            if gap <= gap_limit:
                prev["_end_sec"] = max(prev["_end_sec"], curr["_end_sec"])
                prev["end_time"] = _to_mmss(prev["_end_sec"])
                prev["duration"] = max(0, prev["_end_sec"] - prev["_start_sec"])
                # Keep transcript order and preserve original wording.
                prev_asr = str(prev.get("asr", "")).strip()
                curr_asr = str(curr.get("asr", "")).strip()
                if curr_asr:
                    prev["asr"] = f"{prev_asr} | {curr_asr}" if prev_asr else curr_asr
                prev["speaker_confidence"] = round(
                    max(float(prev.get("speaker_confidence", 0.0)), float(curr.get("speaker_confidence", 0.0))),
                    4,
                )
                continue
            merged.append(curr)

        cleaned = []
        for row in merged:
            row = dict(row)
            row.pop("_start_sec", None)
            row.pop("_end_sec", None)
            cleaned.append(row)
        merged_all[voice_node_id] = cleaned

    return merged_all


def process_voices_speaker_aware(
    video_graph,
    base64_audio,
    base64_video,
    save_path,
    preprocessing=None,
    merge_gap_seconds: float = 1.0,
    min_speaker_confidence: float = 0.0,
):
    """Run original voice pipeline and enrich outputs with speaker metadata.

    This function preserves original node-building behavior by delegating to
    `process_voices`, then only post-processes per-utterance records used by
    downstream memory generation.
    """
    if preprocessing is None:
        preprocessing = []
    # The base pipeline currently skips when the intermediate file does not exist.
    # For speaker-aware runs, ensure a placeholder exists so it can fall back to
    # fresh diarization in its own error-handling branch.
    if save_path and not os.path.exists(save_path):
        os.makedirs(os.path.dirname(save_path) or ".", exist_ok=True)
        with open(save_path, "w", encoding="utf-8") as f:
            f.write("")
    id2voices = process_voices(
        video_graph,
        base64_audio,
        base64_video,
        save_path=save_path,
        preprocessing=preprocessing,
    )
    if not id2voices:
        return {}

    enriched = _attach_speaker_meta(video_graph, id2voices)
    return _merge_same_speaker_turns(
        enriched,
        merge_gap_seconds=merge_gap_seconds,
        min_speaker_confidence=min_speaker_confidence,
    )
