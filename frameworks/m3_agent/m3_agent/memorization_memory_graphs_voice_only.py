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
import os
import json
import logging
import argparse
import glob
import pickle

from mmagent.videograph import VideoGraph
from mmagent.utils.video_processing import process_video_clip
from mmagent.voice_processing import process_voices
from mmagent.memory_processing_qwen import process_memories, generate_memories

logger = logging.getLogger(__name__)
processing_config = json.load(open("configs/processing_config.json"))
memory_config = json.load(open("configs/memory_config.json"))

preprocessing = []

def _with_suffix(path, suffix):
    if not suffix:
        return path
    root, ext = os.path.splitext(path)
    return f"{root}{suffix}{ext}"

def process_segment(
    video_graph,
    base64_video,
    base64_frames,
    base64_audio,
    clip_id,
    clip_path,
    save_path,
):
    os.makedirs(save_path, exist_ok=True)

    voices_path = os.path.join(save_path, f"clip_{clip_id}_voices.json")
    id2voices = process_voices(
        video_graph,
        base64_audio,
        base64_video,
        save_path=voices_path,
        preprocessing=[],
    )

    # Voice-only memory graphs: skip face processing entirely.
    id2faces = {}

    episodic_memories, semantic_memories = generate_memories(
        base64_frames,
        id2faces,
        id2voices,
        clip_path,
    )

    process_memories(video_graph, episodic_memories, clip_id, type="episodic")
    process_memories(video_graph, semantic_memories, clip_id, type="semantic")

def streaming_process_video(video_graph, sample, output_path, intermediate_outputs_path):
    """Process video segments at specified intervals with given fps.

    Args:
        video_graph (VideoGraph): Graph object to store video information
        sample (dict): Input sample containing clip path and outputs
        output_path (str): Path to write the memory graph
    """
    clips = glob.glob(sample["clip_path"] + "/*")
    for clip_path in clips:
        clip_id = int(clip_path.split("/")[-1].split(".")[0])
        base64_video, base64_frames, base64_audio = process_video_clip(clip_path)

        if base64_frames:
            process_segment(
                video_graph,
                base64_video,
                base64_frames,
                base64_audio,
                clip_id,
                clip_path,
                intermediate_outputs_path,
            )

    video_graph.refresh_equivalences()
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    with open(output_path, "wb") as f:
        pickle.dump(video_graph, f)

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--data_file", type=str, default="data/data.jsonl")
    parser.add_argument(
        "--output_suffix",
        type=str,
        default="_voice_only",
        help="Suffix inserted before file extension for output mem_path.",
    )
    parser.add_argument(
        "--intermediate_suffix",
        type=str,
        default="",
        help="Suffix appended to intermediate_outputs directory.",
    )
    args = parser.parse_args()

    with open(args.data_file, "r") as f:
        for line in f:
            sample = json.loads(line)
            output_path = _with_suffix(sample["mem_path"], args.output_suffix)
            intermediate_outputs_path = _with_suffix(
                sample["intermediate_outputs"], args.intermediate_suffix
            )
            if not os.path.exists(output_path):
                video_graph = VideoGraph(**memory_config)
                streaming_process_video(
                    video_graph, sample, output_path, intermediate_outputs_path
                )
