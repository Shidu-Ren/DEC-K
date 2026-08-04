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
"""
Audio-only memory graph generation — V4: Context-Aware Memorization.
Injects rolling context from prior clips into the LLM prompt, so each
clip's memory generation is aware of previously identified voices, events,
and equivalences.  Uses VideoGraphDev for voice embedding verification.
"""
import os
import json
import logging
import argparse
import glob
import pickle
from collections import deque

from mmagent.videograph_dev import VideoGraphDev
from mmagent.utils.video_processing import process_audio_clip
from mmagent.voice_processing import process_voices
from mmagent.memory_processing_qwen import (
    process_memories,
    generate_audio_context,
    generate_all_memories,
)
from mmagent.prompts import prompt_generate_memory_with_ids_sft

logger = logging.getLogger(__name__)
processing_config = json.load(open("configs/processing_config.json"))
memory_config = json.load(open("configs/memory_config.json"))

# Maximum number of prior clips to keep in the rolling context
CONTEXT_WINDOW = 3


def _with_suffix(path, suffix):
    if not suffix:
        return path
    root, ext = os.path.splitext(path)
    return f"{root}{suffix}{ext}"


def generate_memories_audio_only_context_aware(
    voices_list, video_path, prior_context
):
    """Generate memories with rolling context from prior clips injected into the prompt.

    Args:
        voices_list: id2voices mapping for the current clip
        video_path: path to the clip file
        prior_context: list of strings summarizing prior clips' semantic memories
    """
    audio_context = generate_audio_context(voices_list)

    # Build the context-aware prompt
    if prior_context:
        context_block = (
            "[PRIOR CONTEXT FROM EARLIER IN THIS VIDEO]\n"
            "The following is a summary of what happened earlier. Use this to:\n"
            "- Identify returning speakers (if a new voice sounds like a previously seen voice, "
            "output an Equivalence statement)\n"
            "- Maintain narrative continuity\n"
            "- Avoid contradicting established facts\n\n"
            + "\n".join(f"- {c}" for c in prior_context)
            + "\n\n[END OF PRIOR CONTEXT]\n\n"
        )
        prompt_text = context_block + prompt_generate_memory_with_ids_sft
    else:
        prompt_text = prompt_generate_memory_with_ids_sft

    episodic_memories, semantic_memories = generate_all_memories(
        audio_context,
        video_path=video_path,
        prompt_text=prompt_text,
    )
    return episodic_memories, semantic_memories


class RollingContext:
    """Maintains a sliding window of semantic memories from prior clips."""

    def __init__(self, window_size=CONTEXT_WINDOW):
        self.window_size = window_size
        self.buffer = deque(maxlen=window_size)

    def get_context(self):
        """Return flattened list of context strings."""
        result = []
        for clip_memories in self.buffer:
            result.extend(clip_memories)
        return result

    def add_clip(self, semantic_memories):
        """Add semantic memories from the latest clip.

        Args:
            semantic_memories: list of strings (raw semantic memory texts from LLM)
        """
        clip_summary = []
        for mem in semantic_memories:
            if isinstance(mem, str) and mem.strip():
                clip_summary.append(mem.strip())
        self.buffer.append(clip_summary)


def process_segment(
    video_graph,
    base64_audio,
    clip_id,
    clip_path,
    save_path,
    rolling_context,
):
    os.makedirs(save_path, exist_ok=True)

    voices_path = os.path.join(save_path, f"clip_{clip_id}_voices.json")
    id2voices = process_voices(
        video_graph,
        base64_audio,
        base64_video=None,
        save_path=voices_path,
        preprocessing=[],
    )

    # Get prior context
    prior_context = rolling_context.get_context()
    if prior_context:
        logger.info(
            f"Clip {clip_id}: injecting {len(prior_context)} prior context items"
        )

    # Generate memories with context awareness
    episodic_memories, semantic_memories = generate_memories_audio_only_context_aware(
        id2voices,
        clip_path,
        prior_context,
    )

    process_memories(video_graph, episodic_memories, clip_id, type="episodic")
    process_memories(video_graph, semantic_memories, clip_id, type="semantic")

    # Update rolling context with this clip's semantic memories
    rolling_context.add_clip(semantic_memories)


def streaming_process_video(video_graph, sample, output_path, intermediate_outputs_path):
    """Process video segments with context-aware memorization."""
    clips = glob.glob(sample["clip_path"] + "/*")

    # Sort clips numerically for temporal order
    try:
        clips.sort(key=lambda x: int(os.path.basename(x).split(".")[0]))
    except Exception:
        clips.sort()

    rolling_context = RollingContext(window_size=CONTEXT_WINDOW)

    for clip_path in clips:
        clip_id = int(clip_path.split("/")[-1].split(".")[0])
        base64_audio = process_audio_clip(clip_path)
        if base64_audio:
            process_segment(
                video_graph,
                base64_audio,
                clip_id,
                clip_path,
                intermediate_outputs_path,
                rolling_context,
            )

    video_graph.refresh_equivalences()

    # Log rejected equivalences for debugging
    if video_graph.rejected_equivalences:
        logger.info(
            f"=== Rejected {len(video_graph.rejected_equivalences)} equivalences ==="
        )
        for rej in video_graph.rejected_equivalences:
            logger.info(f"  {rej['equivalence']} (sim={rej['similarity']:.4f})")

    # Log character mappings
    logger.info("=== Character mappings ===")
    for char_id, tags in video_graph.character_mappings.items():
        name = video_graph.character_names.get(char_id, "?")
        logger.info(f"  {char_id} ({name}): {tags}")

    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    with open(output_path, "wb") as f:
        pickle.dump(video_graph, f)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--data_file", type=str, default="data/data.jsonl")
    parser.add_argument(
        "--list_file",
        type=str,
        default=None,
        help="Optional file with video IDs (one per line) to limit processing.",
    )
    parser.add_argument(
        "--output_suffix",
        type=str,
        default="_audio_dev_v4",
        help="Suffix inserted before file extension for output mem_path.",
    )
    parser.add_argument(
        "--intermediate_suffix",
        type=str,
        default="",
        help="Suffix appended to intermediate_outputs directory.",
    )
    parser.add_argument(
        "--voice_equiv_threshold",
        type=float,
        default=0.5,
        help="Minimum cosine similarity for voice-voice equivalence verification.",
    )
    parser.add_argument(
        "--context_window",
        type=int,
        default=3,
        help="Number of prior clips to include as context (default: 3).",
    )
    args = parser.parse_args()

    CONTEXT_WINDOW = args.context_window

    # Load optional video ID filter
    id_filter = None
    if args.list_file:
        with open(args.list_file) as f:
            id_filter = set(
                line.strip()
                for line in f
                if line.strip() and not line.strip().startswith("#")
            )
        print(f"Filtering to {len(id_filter)} video IDs from {args.list_file}")

    with open(args.data_file, "r") as f:
        for line in f:
            sample = json.loads(line)
            vid = sample.get("id") or sample.get("video_id") or ""
            if id_filter and vid not in id_filter:
                continue
            output_path = _with_suffix(sample["mem_path"], args.output_suffix)
            intermediate_outputs_path = _with_suffix(
                sample["intermediate_outputs"], args.intermediate_suffix
            )
            if not os.path.exists(output_path):
                video_graph = VideoGraphDev(
                    voice_equiv_threshold=args.voice_equiv_threshold,
                    **memory_config,
                )
                streaming_process_video(
                    video_graph, sample, output_path, intermediate_outputs_path
                )
