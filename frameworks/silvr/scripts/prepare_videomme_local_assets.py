#!/usr/bin/env python3
"""Prepare local VideoMME-long captions/subtitles for the SiLVR format."""

import argparse
import json
import os
import shutil
from pathlib import Path


DEFAULT_CAPTION_ROOT = os.environ.get("VIDEOMME_CAPTION_ROOT", "data/videomme/captions_source")
DEFAULT_SUBTITLE_SRC = os.environ.get("VIDEOMME_SUBTITLE_ROOT", "data/videomme/subtitles_source")


def normalize_caption_text(text, keep_asr=False):
    if not keep_asr:
        text = text.split("ASR transcript:", 1)[0]
    text = text.replace("Visual caption:", "", 1).strip()
    return " ".join(text.split())


def write_caption_file(src_json, dst_txt, keep_asr=False):
    with open(src_json, "r", encoding="utf-8") as f:
        entries = json.load(f)

    lines = []
    for entry in entries:
        text = entry["text"] if isinstance(entry, dict) and "text" in entry else str(entry)
        lines.append(normalize_caption_text(text, keep_asr=keep_asr))

    dst_txt.parent.mkdir(parents=True, exist_ok=True)
    with open(dst_txt, "w", encoding="utf-8") as f:
        for line in lines:
            f.write(f"{line}\n")
    return len(lines)


def link_or_copy(src, dst, copy=False, overwrite=False):
    dst.parent.mkdir(parents=True, exist_ok=True)
    if dst.exists() or dst.is_symlink():
        if not overwrite:
            return False
        dst.unlink()
    if copy:
        shutil.copy2(src, dst)
    else:
        os.symlink(src, dst)
    return True


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--caption_root", default=DEFAULT_CAPTION_ROOT)
    parser.add_argument("--caption_scale", default="30sec")
    parser.add_argument("--out_caption_dir", default="")
    parser.add_argument("--subtitle_src", default=DEFAULT_SUBTITLE_SRC)
    parser.add_argument("--out_subtitle_dir", default="data/videomme/subtitles_asr_m3")
    parser.add_argument("--keep_asr_in_caption", action="store_true")
    parser.add_argument("--copy_subtitles", action="store_true")
    parser.add_argument("--overwrite", action="store_true")
    args = parser.parse_args()

    caption_root = Path(args.caption_root)
    if not args.out_caption_dir:
        suffix = "caption_asr" if args.keep_asr_in_caption else "visual"
        args.out_caption_dir = f"data/videomme/captions_{args.caption_scale}_m3_qwen3vl8b_{suffix}"
    out_caption_dir = Path(args.out_caption_dir)
    subtitle_src = Path(args.subtitle_src)
    out_subtitle_dir = Path(args.out_subtitle_dir)

    converted = 0
    total_clips = 0
    missing_scale = []
    for video_dir in sorted(p for p in caption_root.iterdir() if p.is_dir()):
        src_json = video_dir / f"{args.caption_scale}.json"
        if not src_json.exists():
            missing_scale.append(video_dir.name)
            continue
        n_lines = write_caption_file(
            src_json,
            out_caption_dir / f"{video_dir.name}.txt",
            keep_asr=args.keep_asr_in_caption,
        )
        converted += 1
        total_clips += n_lines

    linked = 0
    if subtitle_src.exists():
        for src in sorted(subtitle_src.glob("*.srt")):
            if link_or_copy(
                src.resolve(),
                out_subtitle_dir / src.name,
                copy=args.copy_subtitles,
                overwrite=args.overwrite,
            ):
                linked += 1

    print(f"caption_root={caption_root}")
    print(f"caption_scale={args.caption_scale}")
    print(f"out_caption_dir={out_caption_dir.resolve()}")
    print(f"converted_videos={converted}")
    print(f"total_caption_lines={total_clips}")
    print(f"missing_scale_count={len(missing_scale)}")
    print(f"subtitle_src={subtitle_src}")
    print(f"out_subtitle_dir={out_subtitle_dir.resolve()}")
    print(f"linked_or_copied_subtitles={linked}")


if __name__ == "__main__":
    main()
