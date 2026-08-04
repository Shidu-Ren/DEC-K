# Copyright (2025) Bytedance Ltd. and/or its affiliates
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
import argparse
import json
import os

from mmagent.robot_dev import export_robot_dev_contexts_for_video
from mmagent.utils.general import load_video_graph


def _resolve_clip_dir(sample, clip_root):
    clip_dir = sample.get("clip_path")
    if clip_dir and os.path.isdir(clip_dir):
        return clip_dir
    video_id = os.path.splitext(os.path.basename(sample["mem_path"]))[0]
    return os.path.join(clip_root, video_id)


def _resolve_intermediate_dir(sample, intermediate_root):
    intermediate_dir = sample.get("intermediate_outputs")
    if intermediate_dir and os.path.isdir(intermediate_dir):
        return intermediate_dir
    video_id = os.path.splitext(os.path.basename(sample["mem_path"]))[0]
    return os.path.join(intermediate_root, video_id)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--data_file", type=str, default="data/annotations/robot.json")
    parser.add_argument(
        "--list_file",
        type=str,
        default=None,
        help="Optional file with video IDs (one per line) to export.",
    )
    parser.add_argument(
        "--clip_root",
        type=str,
        default="data/clips/robot",
        help="Fallback root for clip directories when not present in the annotation file.",
    )
    parser.add_argument(
        "--intermediate_root",
        type=str,
        default="data/intermediate_outputs/robot",
        help="Fallback root for intermediate output directories when not present in the annotation file.",
    )
    parser.add_argument(
        "--context_root",
        type=str,
        default="data/vl_contexts/robot",
        help="Output directory for exported robot dev VL contexts.",
    )
    parser.add_argument(
        "--overwrite",
        action="store_true",
        help="Overwrite existing context JSON files.",
    )
    args = parser.parse_args()

    with open(args.data_file, "r", encoding="utf-8") as f:
        datas = json.load(f)

    id_list = None
    if args.list_file:
        with open(args.list_file, "r", encoding="utf-8") as f:
            id_list = [line.strip() for line in f if line.strip() and not line.startswith("#")]

    items = ((vid, datas[vid]) for vid in id_list if vid in datas) if id_list else datas.items()
    total_written = 0
    total_videos = 0

    for video_id, sample in items:
        mem_path = sample["mem_path"]
        video_graph = load_video_graph(mem_path)
        if video_graph is None:
            print(f"[WARN] skip {video_id}: missing graph {mem_path}")
            continue

        clip_dir = _resolve_clip_dir(sample, args.clip_root)
        intermediate_dir = _resolve_intermediate_dir(sample, args.intermediate_root)
        written = export_robot_dev_contexts_for_video(
            video_graph=video_graph,
            mem_path=mem_path,
            clip_dir=clip_dir,
            intermediate_dir=intermediate_dir,
            context_root=args.context_root,
            overwrite=args.overwrite,
        )
        total_written += written
        total_videos += 1
        print(f"[INFO] {video_id}: wrote {written} clip context files")

    print(f"[DONE] exported {total_written} clip context files across {total_videos} videos")
