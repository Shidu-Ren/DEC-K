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
Audio-only memory graph generation — dev version.
Uses VideoGraphDev with voice embedding verification for equivalences.
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
from mmagent.memory_processing_qwen import process_memories, generate_memories_audio_only
from mmagent.utils.chat_api import parallel_get_embedding
from mmagent.utils.chat_qwen import generate_messages, get_response as qwen_get_response

logger = logging.getLogger(__name__)
processing_config = json.load(open("configs/processing_config.json"))
memory_config = json.load(open("configs/memory_config.json"))

def _with_suffix(path, suffix):
    if not suffix:
        return path
    root, ext = os.path.splitext(path)
    return f"{root}{suffix}{ext}"

def _time_to_ms(time_str):
    """Convert 'MM:SS' or 'HH:MM:SS' to milliseconds."""
    parts = time_str.split(':')
    if len(parts) == 2:
        m, s = parts
        return (int(m) * 60 + int(s)) * 1000
    elif len(parts) == 3:
        h, m, s = parts
        return (int(h) * 3600 + int(m) * 60 + int(s)) * 1000
    return 0

def add_temporal_dialogue_edges(video_graph, id2voices, threshold_ms=2000):
    """Add edges between voice nodes that occur within a time threshold."""
    timeline = []
    for node_id, voices in id2voices.items():
        for voice in voices:
            start_ms = _time_to_ms(voice['start_time'])
            end_ms = _time_to_ms(voice['end_time'])
            timeline.append((start_ms, end_ms, node_id))

    # Sort by start time
    timeline.sort(key=lambda x: x[0])

    edges_added = 0
    for i in range(len(timeline) - 1):
        _, end_i, node_i = timeline[i]
        start_j, _, node_j = timeline[i+1]

        # If different nodes and close in time
        if node_i != node_j and (start_j - end_i) <= threshold_ms:
            # We use a weight of 0.5 for Dialogue_Next structural edges
            if video_graph.add_edge(node_i, node_j, weight=0.5):
                edges_added += 1
                logger.debug(f"Added dialogue edge: {node_i} -> {node_j} (gap: {start_j - end_i}ms)")

    if edges_added > 0:
        logger.info(f"Added {edges_added} temporal dialogue edges in clip.")

class SceneConsolidator:
    """Buffers memories over N clips to create Scene Nodes."""
    def __init__(self, window_size=6):
        self.window_size = window_size
        self.clip_count = 0
        self.voice_node_ids = set()
        self.text_node_ids = set()
        self.memories_text = []

    def add_clip_data(self, id2voices, new_text_node_ids, episodic_memories):
        """Add data from a single clip."""
        self.clip_count += 1
        for nid in id2voices.keys():
            self.voice_node_ids.add(nid)
        for nid in new_text_node_ids:
            self.text_node_ids.add(nid)

        # episodic_memories is a flat list of strings (the raw captions from the LLM)
        for mem in episodic_memories:
            if isinstance(mem, str):
                self.memories_text.append(mem)
            elif isinstance(mem, dict):
                # Fallback if format changes
                text = mem.get('video_descriptions', [mem.get('video_description', '')])
                if text:
                    self.memories_text.append(text[0] if isinstance(text, list) else text)

    def flush(self, video_graph, clip_id):
        """Generate a scene node for the buffered data and link it using the memory model."""
        if not self.memories_text:
            self._reset()
            return

        logger.info(f"Flushing scene consolidator at clip {clip_id}. {len(self.memories_text)} memory fragments.")

        prompt = (
            "You are summarizing a short scene from a video. Based on the following dialogue events "
            "and observations, write a single cohesive paragraph describing what happened in this scene. "
            "Keep all <voice_X> tags to show who was involved. Be concise and factual.\n\n"
            "Scene fragments:\n" + "\n".join(f"- {m}" for m in self.memories_text[:20]) + "\n\n"
            "Output ONLY the summary paragraph, no extra commentary."
        )

        # Use the same Qwen memory model, NOT Gemini, to keep comparison valid
        input_msg = [{"type": "text", "content": prompt}]
        messages = generate_messages(input_msg)
        response = qwen_get_response(messages)

        if response and response[0]:
            summary = response[0].strip()
            if summary:
                embedding = parallel_get_embedding("text-embedding-3-large", [summary])[0][0]
                memory_obj = {'contents': [summary], 'embeddings': [embedding]}
                scene_node_id = video_graph.add_text_node(memory_obj, clip_id, 'semantic')
                logger.info(f"Scene Node {scene_node_id}: {summary[:120]}...")
                for vid in self.voice_node_ids:
                    video_graph.add_edge(scene_node_id, vid, weight=0.6)
                for tid in self.text_node_ids:
                    video_graph.add_edge(scene_node_id, tid, weight=0.6)
        self._reset()

    def _reset(self):
        self.clip_count = 0
        self.voice_node_ids.clear()
        self.text_node_ids.clear()
        self.memories_text.clear()


def process_segment(
    video_graph,
    base64_audio,
    clip_id,
    clip_path,
    save_path,
    scene_consolidator,
):
    os.makedirs(save_path, exist_ok=True)

    voices_path = os.path.join(save_path, f"clip_{clip_id}_voices.json")
    # id2voices maps voice_node_id -> list of dictionaries with 'asr', 'start_time', 'end_time'
    id2voices = process_voices(
        video_graph,
        base64_audio,
        base64_video=None,
        save_path=voices_path,
        preprocessing=[],
    )

    # 1. Add Dialogue Temporal Edges
    if id2voices:
        add_temporal_dialogue_edges(video_graph, id2voices, threshold_ms=2000)

    # Track new text nodes created in this clip
    prev_text_nodes_count = len(video_graph.text_nodes)

    episodic_memories, semantic_memories = generate_memories_audio_only(
        id2voices,
        clip_path,
    )

    process_memories(video_graph, episodic_memories, clip_id, type="episodic")
    process_memories(video_graph, semantic_memories, clip_id, type="semantic")

    # Collect newly added text node IDs
    new_text_nodes = set(video_graph.text_nodes[prev_text_nodes_count:])

    # 2. Add to Scene Consolidator
    scene_consolidator.add_clip_data(id2voices, new_text_nodes, episodic_memories)

    # Flush scene buffer if it reaches window size
    if scene_consolidator.clip_count >= scene_consolidator.window_size:
        scene_consolidator.flush(video_graph, clip_id)


def streaming_process_video(video_graph, sample, output_path, intermediate_outputs_path):
    """Process video segments at specified intervals with given fps."""
    clips = glob.glob(sample["clip_path"] + "/*")

    # Sort clips numerically if possible to process in temporal order
    try:
        clips.sort(key=lambda x: int(os.path.basename(x).split(".")[0]))
    except Exception:
        clips.sort()

    scene_consolidator = SceneConsolidator(window_size=6)

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
                scene_consolidator
            )

    # Flush any remaining memories into a final scene
    if scene_consolidator.clip_count > 0:
        scene_consolidator.flush(video_graph, clip_id)

    video_graph.refresh_equivalences()

    # Log rejected equivalences for debugging
    if video_graph.rejected_equivalences:
        logger.info(f"=== Rejected {len(video_graph.rejected_equivalences)} equivalences ===")
        for rej in video_graph.rejected_equivalences:
            logger.info(f"  {rej['equivalence']} (sim={rej['similarity']:.4f})")

    # Log character mappings
    logger.info(f"=== Character mappings ===")
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
        default="_audio_dev_v3",
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
            id_filter = set(line.strip() for line in f if line.strip() and not line.strip().startswith("#"))
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
