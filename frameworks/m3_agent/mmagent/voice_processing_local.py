from __future__ import annotations

import base64
import io
import logging
import math
import os
import sys
import tempfile
from pathlib import Path
from typing import Callable, List

import numpy as np
import torch
from pydub import AudioSegment
from pydub.silence import detect_nonsilent

logger = logging.getLogger(__name__)

_LOCAL_DIARIZER = None
_LOCAL_WHISPER_MODEL = None
_LOCAL_WHISPER_MODEL_NAME = None


def _segmenter_backend() -> str:
    return os.getenv("M3_LOCAL_SEGMENTER", "whisper").strip().lower() or "whisper"


def _safe_float_env(name: str, default: float) -> float:
    try:
        return float(os.getenv(name, str(default)))
    except Exception:
        return default


def _voice_device() -> str:
    explicit = os.getenv("M3_LOCAL_VOICE_DEVICE", "").strip()
    if explicit:
        return explicit
    if torch.cuda.is_available():
        return "cuda"
    return "cpu"


def _ensure_speakerlab_import():
    repo_root = Path(__file__).resolve().parents[1]
    speakerlab_root = repo_root / "speakerlab"
    speakerlab_root_str = str(speakerlab_root)
    if speakerlab_root_str not in sys.path:
        sys.path.insert(0, speakerlab_root_str)
    from speakerlab.bin.infer_diarization import Diarization3Dspeaker

    return Diarization3Dspeaker


def _get_diarizer():
    global _LOCAL_DIARIZER
    if _LOCAL_DIARIZER is None:
        Diarization3Dspeaker = _ensure_speakerlab_import()
        device = _voice_device()
        logger.info("Initializing local diarizer on %s", device)
        _LOCAL_DIARIZER = Diarization3Dspeaker(device=device, include_overlap=False)
    return _LOCAL_DIARIZER


def _get_whisper_model():
    global _LOCAL_WHISPER_MODEL, _LOCAL_WHISPER_MODEL_NAME

    model_name = os.getenv("M3_LOCAL_WHISPER_MODEL", "tiny.en").strip() or "tiny.en"
    if _LOCAL_WHISPER_MODEL is None or _LOCAL_WHISPER_MODEL_NAME != model_name:
        import whisper

        device = "cuda" if _voice_device().startswith("cuda") and torch.cuda.is_available() else "cpu"
        logger.info("Loading local whisper model %s on %s", model_name, device)
        _LOCAL_WHISPER_MODEL = whisper.load_model(model_name, device=device)
        _LOCAL_WHISPER_MODEL_NAME = model_name
    return _LOCAL_WHISPER_MODEL


def _decode_audio(base64_audio: bytes | str) -> AudioSegment:
    if isinstance(base64_audio, bytes):
        payload = base64.b64decode(base64_audio)
    else:
        payload = base64.b64decode(base64_audio.encode("utf-8"))
    return AudioSegment.from_file(io.BytesIO(payload), format="wav")


def _merge_segments(segments: List[dict], gap_seconds: float) -> List[dict]:
    if not segments:
        return []

    segments = sorted(segments, key=lambda x: (x["start"], x["end"]))
    merged = [dict(segments[0])]
    for seg in segments[1:]:
        prev = merged[-1]
        same_speaker = seg.get("speaker") == prev.get("speaker")
        close_enough = seg["start"] <= prev["end"] + gap_seconds
        if same_speaker and close_enough:
            prev["end"] = max(prev["end"], seg["end"])
            continue
        merged.append(dict(seg))
    return merged


def _to_mmss(seconds: float, *, rounding: str) -> str:
    if rounding == "floor":
        total = int(math.floor(max(0.0, seconds)))
    else:
        total = int(math.ceil(max(0.0, seconds)))
    mm = total // 60
    ss = total % 60
    return f"{mm:02d}:{ss:02d}"


def _transcribe_segment(audio_segment: AudioSegment) -> str:
    model = _get_whisper_model()
    audio_segment = audio_segment.set_frame_rate(16000).set_channels(1)
    samples = np.frombuffer(audio_segment.raw_data, np.int16).astype(np.float32) / 32768.0
    transcribe_kwargs = {
        "language": os.getenv("M3_LOCAL_WHISPER_LANGUAGE", "en"),
        "fp16": _voice_device().startswith("cuda") and torch.cuda.is_available(),
        "temperature": 0.0,
        "condition_on_previous_text": False,
    }
    result = model.transcribe(samples, **transcribe_kwargs)
    return str(result.get("text", "")).strip()


def _whisper_segments(audio: AudioSegment, filter: Callable[[dict], bool] | None = None) -> List[dict]:
    min_silence_len = int(_safe_float_env("M3_LOCAL_VAD_MIN_SILENCE_MS", 600))
    silence_offset_db = _safe_float_env("M3_LOCAL_VAD_SILENCE_OFFSET_DB", 16.0)
    merge_gap_ms = int(_safe_float_env("M3_LOCAL_VOICE_MERGE_GAP", 0.75) * 1000)
    pad_ms = int(_safe_float_env("M3_LOCAL_VAD_PAD_MS", 200))
    silence_thresh = audio.dBFS - silence_offset_db

    logger.info(
        "Running local VAD before whisper (min_silence_ms=%d, silence_thresh=%.2f dBFS)",
        min_silence_len,
        silence_thresh,
    )
    spans = detect_nonsilent(
        audio,
        min_silence_len=min_silence_len,
        silence_thresh=silence_thresh,
        seek_step=10,
    )
    if not spans:
        logger.info("No nonsilent spans found; falling back to whole clip")
        spans = [[0, len(audio)]]

    merged = []
    for start_ms, end_ms in spans:
        start_ms = max(0, start_ms - pad_ms)
        end_ms = min(len(audio), end_ms + pad_ms)
        if not merged or start_ms > merged[-1][1] + merge_gap_ms:
            merged.append([start_ms, end_ms])
        else:
            merged[-1][1] = max(merged[-1][1], end_ms)

    rows = []
    for start_ms, end_ms in merged:
        if end_ms <= start_ms:
            continue
        segment_audio = audio[start_ms:end_ms]
        text = _transcribe_segment(segment_audio)
        if not text.strip():
            continue

        start_sec = int(math.floor(start_ms / 1000.0))
        end_sec = int(math.ceil(end_ms / 1000.0))
        if end_sec <= start_sec:
            end_sec = start_sec + 1
        row = {
            "start_time": _to_mmss(start_sec, rounding="floor"),
            "end_time": _to_mmss(end_sec, rounding="ceil"),
            "asr": text.strip(),
            "duration": end_sec - start_sec,
        }
        if filter is None or filter(row):
            rows.append(row)

    logger.info("Whisper+VAD kept %d segments from %d detected spans", len(rows), len(merged))
    return rows


def _speakerlab_segments(audio: AudioSegment, filter: Callable[[dict], bool] | None = None) -> List[dict]:
    logger.info("Decoding clip audio for local diarization")
    duration_seconds = len(audio) / 1000.0

    with tempfile.NamedTemporaryFile(suffix=".wav") as tmp:
        audio.export(tmp.name, format="wav")
        logger.info("Running local speaker diarization on temporary wav %s", tmp.name)
        diarizer = _get_diarizer()
        try:
            diarized = diarizer(tmp.name)
        except ValueError as exc:
            if "empty sequence" not in str(exc):
                raise
            logger.warning(
                "Local speaker diarization produced no valid chunks; falling back to whisper VAD for this clip"
            )
            return _whisper_segments(audio, filter=filter)
    logger.info("Local diarization produced %d raw segments", len(diarized))

    segments = []
    for start, end, speaker in diarized:
        start = max(0.0, float(start))
        end = min(duration_seconds, float(end))
        if end <= start:
            continue
        segments.append({"start": start, "end": end, "speaker": str(speaker)})

    gap_seconds = _safe_float_env("M3_LOCAL_VOICE_MERGE_GAP", 0.75)
    merged = _merge_segments(segments, gap_seconds)
    logger.info("Merged diarization into %d candidate segments (gap %.2fs)", len(merged), gap_seconds)

    results = []
    for seg in merged:
        start_sec = int(math.floor(seg["start"]))
        end_sec = int(math.ceil(seg["end"]))
        if end_sec <= start_sec:
            end_sec = start_sec + 1

        start_ms = start_sec * 1000
        end_ms = min(len(audio), end_sec * 1000)
        if end_ms <= start_ms:
            continue

        segment_audio = audio[start_ms:end_ms]
        logger.info(
            "Transcribing local segment %s-%s (speaker=%s, %.2fs)",
            row_start := _to_mmss(start_sec, rounding="floor"),
            row_end := _to_mmss(end_sec, rounding="ceil"),
            seg["speaker"],
            end_sec - start_sec,
        )
        text = _transcribe_segment(segment_audio).strip()
        if not text:
            continue
        row = {
            "start_time": row_start,
            "end_time": row_end,
            "asr": text,
            "duration": end_sec - start_sec,
        }
        if filter is None or filter(row):
            results.append(row)

    logger.info("Local backend kept %d final segments after filtering", len(results))
    return results


def diarize_audio_local(base64_audio: bytes | str, filter: Callable[[dict], bool] | None = None) -> List[dict]:
    audio = _decode_audio(base64_audio)
    backend = _segmenter_backend()
    if backend == "speakerlab":
        return _speakerlab_segments(audio, filter=filter)
    if backend == "whisper":
        return _whisper_segments(audio, filter=filter)
    raise ValueError(f"Unsupported local segmenter backend: {backend}")
