# Copyright (2025) Bytedance Ltd. and/or its affiliates

# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at

#     http://www.apache.org/licenses/LICENSE-2.0

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

from mmagent.utils.video_processing import process_video_clip
from mmagent.face_processing import process_faces
from mmagent.voice_processing import process_voices

# Configure logger
logger = logging.getLogger(__name__)

processing_config = json.load(open("configs/processing_config.json"))
memory_config = json.load(open("configs/memory_config.json"))
SKIP_VOICES = os.getenv("M3_SKIP_VOICES", "0") == "1"
SKIP_FACES = os.getenv("M3_SKIP_FACES", "0") == "1"


def _write_empty_intermediate(sample, clip_id):
    save_root = sample["intermediate_outputs"]
    os.makedirs(save_root, exist_ok=True)
    if not SKIP_VOICES:
        with open(os.path.join(save_root, f"clip_{clip_id}_voices.json"), "w") as f:
            json.dump([], f)
    if not SKIP_FACES:
        with open(os.path.join(save_root, f"clip_{clip_id}_faces.json"), "w") as f:
            json.dump([], f)


def _write_empty_voice(sample, clip_id):
    if SKIP_VOICES:
        return
    save_root = sample["intermediate_outputs"]
    os.makedirs(save_root, exist_ok=True)
    with open(os.path.join(save_root, f"clip_{clip_id}_voices.json"), "w") as f:
        json.dump([], f)


def _write_empty_face(sample, clip_id):
    if SKIP_FACES:
        return
    save_root = sample["intermediate_outputs"]
    os.makedirs(save_root, exist_ok=True)
    with open(os.path.join(save_root, f"clip_{clip_id}_faces.json"), "w") as f:
        json.dump([], f)

def process_segment(
    video_graph,
    base64_video,
    base64_frames,
    base64_audio,
    clip_id,
    sample
):
    save_path = sample["intermediate_outputs"]

    if not SKIP_VOICES:
        try:
            process_voices(
                video_graph,
                base64_audio,
                base64_video,
                save_path=os.path.join(save_path, f"clip_{clip_id}_voices.json"),
                preprocessing=["voice"],
            )
        except Exception as exc:
            logger.warning("Voice intermediate generation failed for %s clip %s: %s", sample["video_id"], clip_id, exc)
            _write_empty_voice(sample, clip_id)

    if not SKIP_FACES:
        try:
            process_faces(
                video_graph,
                base64_frames,
                save_path=os.path.join(save_path, f"clip_{clip_id}_faces.json"),
                preprocessing=["face"],
            )
        except Exception as exc:
            logger.warning("Face intermediate generation failed for %s clip %s: %s", sample["video_id"], clip_id, exc)
            _write_empty_face(sample, clip_id)


def streaming_process_video(sample):
    """Process video segments at specified intervals with given fps.

    Args:
        video_graph (VideoGraph): Graph object to store video information
        video_path (str): Path to the video file or directory containing clips
        interval_seconds (float): Time interval between segments in seconds
        fps (float): Frames per second to extract from each segment

    Returns:
        None: Updates video_graph in place with processed segments
    """

    # Process each interval
    clips = sorted(
        glob.glob(sample["clip_path"] + "/*"),
        key=lambda p: int(os.path.basename(p).split(".")[0]),
    )
    for clip_path in clips:
        clip_id = int(clip_path.split("/")[-1].split(".")[0])
        try:
            base64_video, base64_frames, base64_audio = process_video_clip(clip_path)
        except Exception as exc:
            logger.warning("Skipping unreadable clip %s: %s", clip_path, exc)
            _write_empty_intermediate(sample, clip_id)
            continue

        # Process frames for this interval
        if base64_frames:
            process_segment(
                None,
                base64_video,
                base64_frames,
                base64_audio,
                clip_id,
                sample,
            )

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--data_file", type=str, default="data/data.jsonl")
    args = parser.parse_args()

    with open(args.data_file, "r") as f:
        for line in f:
            streaming_process_video(json.loads(line))
