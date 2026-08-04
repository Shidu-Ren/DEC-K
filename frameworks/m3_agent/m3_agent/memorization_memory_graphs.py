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
import pickle
import subprocess
from pathlib import Path
from contextlib import contextmanager

from mmagent.videograph import VideoGraph
from mmagent.utils.video_processing import process_video_clip
from mmagent.face_processing import process_faces
from mmagent.voice_processing import process_voices
from mmagent.memory_processing_qwen import process_memories, generate_memories

logger = logging.getLogger(__name__)
processing_config = json.load(open("configs/processing_config.json"))
memory_config = json.load(open("configs/memory_config.json"))

preprocessing = []
# Qwen Omni's local video reader can hang on very short tail clips, so skip them.
MIN_CLIP_DURATION_SECONDS = 2.0
MIN_CLIP_FRAMES = 24


def _clip_sort_key(clip_path):
    clip_name = Path(clip_path).stem
    try:
        return (0, int(clip_name))
    except ValueError:
        return (1, clip_name)


def _probe_clip_stats(clip_path):
    cmd = [
        "ffprobe",
        "-v",
        "error",
        "-print_format",
        "json",
        "-show_streams",
        "-show_format",
        clip_path,
    ]
    try:
        proc = subprocess.run(cmd, capture_output=True, text=True, check=True)
        data = json.loads(proc.stdout)
    except Exception:
        logger.exception("Failed to probe clip stats for %s", clip_path)
        return None

    duration = None
    video_frames = None

    fmt = data.get("format", {})
    try:
        duration = float(fmt.get("duration"))
    except (TypeError, ValueError):
        duration = None

    for stream in data.get("streams", []):
        if stream.get("codec_type") != "video":
            continue
        try:
            video_frames = int(stream.get("nb_frames"))
        except (TypeError, ValueError):
            video_frames = None
        break

    return {"duration": duration, "video_frames": video_frames}


def _should_skip_memory_clip(clip_path):
    stats = _probe_clip_stats(clip_path)
    if not stats:
        return False

    duration = stats.get("duration")
    video_frames = stats.get("video_frames")
    if duration is not None and duration < MIN_CLIP_DURATION_SECONDS:
        logger.warning(
            "Skip ultra-short clip during memory stage: %s (duration=%.2fs, frames=%s)",
            clip_path,
            duration,
            video_frames,
        )
        return True
    if video_frames is not None and video_frames < MIN_CLIP_FRAMES:
        logger.warning(
            "Skip ultra-short clip during memory stage: %s (duration=%s, frames=%s)",
            clip_path,
            duration,
            video_frames,
        )
        return True
    return False


@contextmanager
def _memory_lock(mem_path):
    lock_path = f"{mem_path}.lock"
    fd = None
    acquired = False
    try:
        fd = os.open(lock_path, os.O_CREAT | os.O_EXCL | os.O_WRONLY)
        payload = {
            "pid": os.getpid(),
            "cwd": os.getcwd(),
        }
        os.write(fd, json.dumps(payload).encode("utf-8"))
        os.close(fd)
        fd = None
        acquired = True
        yield True
    except FileExistsError:
        yield False
    finally:
        if fd is not None:
            os.close(fd)
        if acquired and os.path.exists(lock_path):
            try:
                os.remove(lock_path)
            except FileNotFoundError:
                pass

def process_segment(
    video_graph,
    base64_video,
    base64_frames,
    base64_audio,
    clip_id,
    sample,
    clip_path
):
    save_path = sample["intermediate_outputs"]
    os.makedirs(save_path, exist_ok=True)

    voices_path = os.path.join(save_path, f"clip_{clip_id}_voices.json")
    id2voices = process_voices(
        video_graph,
        base64_audio,
        base64_video,
        save_path=voices_path,
        preprocessing=[],
    )

    faces_path = os.path.join(save_path, f"clip_{clip_id}_faces.json")
    id2faces = process_faces(
        video_graph,
        base64_frames,
        save_path=faces_path,
        preprocessing=[],
    )

    episodic_memories, semantic_memories = generate_memories(
        base64_frames,
        id2faces,
        id2voices,
        clip_path,
    )

    process_memories(video_graph, episodic_memories, clip_id, type="episodic")
    process_memories(video_graph, semantic_memories, clip_id, type="semantic")

def streaming_process_video(video_graph, sample):
    """Process video segments at specified intervals with given fps.

    Args:
        video_graph (VideoGraph): Graph object to store video information
        video_path (str): Path to the video file or directory containing clips
        interval_seconds (float): Time interval between segments in seconds
        fps (float): Frames per second to extract from each segment

    Returns:
        None: Updates video_graph in place with processed segments
    """
    clips = sorted(glob.glob(sample["clip_path"] + "/*"), key=_clip_sort_key)
    for clip_path in clips:
        clip_id = int(Path(clip_path).stem)
        try:
            if _should_skip_memory_clip(clip_path):
                continue
            base64_video, base64_frames, base64_audio = process_video_clip(clip_path)

            # Process frames for this interval
            if base64_frames:
                process_segment(
                    video_graph,
                    base64_video,
                    base64_frames,
                    base64_audio,
                    clip_id,
                    sample,
                    clip_path
                )
            else:
                logger.warning("No frames extracted for clip %s, skip.", clip_path)
        except Exception:
            logger.exception("Failed to process memory graph clip %s, skip and continue.", clip_path)

    video_graph.refresh_equivalences()
    os.makedirs(os.path.dirname(sample["mem_path"]), exist_ok=True)
    with open(sample["mem_path"], "wb") as f:
        pickle.dump(video_graph, f)

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--data_file", type=str, default="data/data.jsonl")
    args = parser.parse_args()
    video_inputs = []

    with open(args.data_file, "r") as f:
        for line in f:
            sample = json.loads(line)
            mem_path = sample["mem_path"]
            if os.path.exists(mem_path):
                logger.info("Skip existing memory graph: %s", mem_path)
                continue
            with _memory_lock(mem_path) as acquired:
                if not acquired:
                    logger.info("Skip locked memory graph: %s", mem_path)
                    continue
                if os.path.exists(mem_path):
                    logger.info("Skip existing memory graph after lock: %s", mem_path)
                    continue
                video_graph = VideoGraph(**memory_config)
                streaming_process_video(video_graph, sample)
