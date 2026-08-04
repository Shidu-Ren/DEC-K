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
Audio-only memory graph generation — V5: Lightweight Prior Summary.
Only injects a single concise event summary from the *previous* clip
(generated directly from the raw audio input, not from stored memories)
into the current clip's memorization prompt.
Uses VideoGraphDev for voice embedding verification.
"""
import os
import json
import logging
import argparse
import glob
import pickle

from mmagent.videograph_dev import VideoGraphDev
from mmagent.utils.video_processing import process_audio_clip
from mmagent.voice_processing import process_voices
from mmagent.utils.chat_qwen import generate_messages, get_response as qwen_get_response
from mmagent.memory_processing_qwen import (
    process_memories,
    generate_audio_context,
    generate_all_memories,
    generate_memories_audio_only,
)
from mmagent.prompts import prompt_generate_memory_with_ids_sft

logger = logging.getLogger(__name__)
processing_config = json.load(open("configs/processing_config.json"))
memory_config = json.load(open("configs/memory_config.json"))


SUMMARIZE_CLIP_PROMPT = (
    "You are given the transcript of a short video clip with speaker labels. "
    "Write a brief 1-2 sentence summary of what happened in this clip. "
    "Focus only on concrete events and actions (who said what, what occurred). "
    "You may use the speaker IDs (e.g. <voice_X>) to refer to speakers. "
    "Do NOT speculate about emotions, intentions, or relationships. "
    "Output ONLY the summary, nothing else."
)


def _with_suffix(path, suffix):
    if not suffix:
        return path
    root, ext = os.path.splitext(path)
    return f"{root}{suffix}{ext}"


def generate_clip_summary(audio_context):
    """Generate a 1-2 sentence event summary from raw audio input.

    Args:
        audio_context: the raw audio context (voice features list) for the clip.

    Returns:
        A short string summarizing what happened, or None on failure.
    """
    input_msg = [
        {"type": "text", "content": SUMMARIZE_CLIP_PROMPT},
    ] + audio_context

    messages = generate_messages(input_msg)
    response = qwen_get_response(messages)

    if response and response[0]:
        summary = response[0].strip()
        if summary:
            logger.info(f"Clip summary: {summary[:150]}")
            return summary
    return None


def generate_memories_with_prior_summary(
    voices_list, video_path, prior_summary
):
    """Generate memories, optionally prepending a prior clip summary to the prompt.

    Args:
        voices_list: id2voices mapping for the current clip
        video_path: path to the clip file
        prior_summary: a short string summarizing the previous clip, or None
    """
    audio_context = generate_audio_context(voices_list)

    if prior_summary:
        context_block = (
            "[WHAT HAPPENED IN THE PREVIOUS CLIP]\n"
            f"{prior_summary}\n"
            "[END OF PREVIOUS CLIP SUMMARY]\n\n"
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


def process_segment(
    video_graph,
    base64_audio,
    clip_id,
    clip_path,
    save_path,
    prior_summary,
):
    """Process a single clip and return the raw audio_context for summary generation.

    Returns:
        audio_context for this clip (to generate its summary for the *next* clip).
    """
    os.makedirs(save_path, exist_ok=True)

    voices_path = os.path.join(save_path, f"clip_{clip_id}_voices.json")
    id2voices = process_voices(
        video_graph,
        base64_audio,
        base64_video=None,
        save_path=voices_path,
        preprocessing=[],
    )

    if prior_summary:
        logger.info(f"Clip {clip_id}: injecting prior summary ({len(prior_summary)} chars)")

    # Generate memories with optional prior summary
    episodic_memories, semantic_memories = generate_memories_with_prior_summary(
        id2voices,
        clip_path,
        prior_summary,
    )

    process_memories(video_graph, episodic_memories, clip_id, type="episodic")
    process_memories(video_graph, semantic_memories, clip_id, type="semantic")

    # Build audio_context from raw input for summary generation
    audio_context = generate_audio_context(id2voices)
    return audio_context


def streaming_process_video(video_graph, sample, output_path, intermediate_outputs_path):
    """Process video segments with lightweight prior-clip summary."""
    clips = glob.glob(sample["clip_path"] + "/*")

    # Sort clips numerically for temporal order
    try:
        clips.sort(key=lambda x: int(os.path.basename(x).split(".")[0]))
    except Exception:
        clips.sort()

    prior_summary = None  # No context for the first clip

    for clip_path in clips:
        clip_id = int(clip_path.split("/")[-1].split(".")[0])
        base64_audio = process_audio_clip(clip_path)
        if base64_audio:
            audio_context = process_segment(
                video_graph,
                base64_audio,
                clip_id,
                clip_path,
                intermediate_outputs_path,
                prior_summary,
            )

            # Generate summary of THIS clip from its raw input
            # This will be used as prior_summary for the NEXT clip
            prior_summary = generate_clip_summary(audio_context)

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
        default="_audio_dev_v5",
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
    args = parser.parse_args()

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
            output_path = _with_suffix(_with_suffix(sample["clip_path"], args.output_suffix), args.output_suffix)
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
