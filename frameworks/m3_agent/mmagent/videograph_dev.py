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
VideoGraphDev: Extended VideoGraph with voice embedding verification
and post-hoc voice clustering for equivalences.
"""
import logging
import re
import struct

import numpy as np
from sklearn.metrics.pairwise import cosine_similarity

from .videograph import VideoGraph
from .memory_processing import parse_video_caption

logger = logging.getLogger(__name__)

# Threshold for voice embedding similarity when verifying LLM-generated equivalences
VOICE_EQUIV_THRESHOLD = 0.5

# Patterns to extract names from semantic memory content
_NAME_PATTERNS = [
    # "Name: <voice_x> is Alice"
    re.compile(r"Name:\s*<voice_(\d+)>\s+is\s+([A-Z][a-zA-Z]+)"),
    # "<voice_x>'s name is Alice"
    re.compile(r"<voice_(\d+)>'s\s+name\s+is\s+([A-Z][a-zA-Z]+)"),
    # "<voice_x> is named Alice"
    re.compile(r"<voice_(\d+)>\s+is\s+named\s+([A-Z][a-zA-Z]+)"),
    # "<voice_x> introduces themselves as Alice"
    re.compile(r"<voice_(\d+)>\s+introduces\s+(?:themselves|herself|himself)\s+as\s+([A-Z][a-zA-Z]+)"),
]


def _to_float_array(emb):
    """Convert an embedding (bytes or ndarray) to a numpy float32 array."""
    if isinstance(emb, (bytes, bytearray)):
        n = len(emb) // 4
        return np.array(struct.unpack('f' * n, emb), dtype=np.float32)
    return np.array(emb, dtype=np.float32)


class VideoGraphDev(VideoGraph):
    """
    Extended VideoGraph with:
    - Voice embedding verification for equivalences (prevents snowball merging)
    - Post-hoc voice clustering (merges high-similarity voices LLM missed)
    - Optional structured name extraction from semantic memories
    """

    def __init__(self, *args, voice_equiv_threshold=VOICE_EQUIV_THRESHOLD,
                 voice_cluster_threshold=0.65,
                 extract_character_names=False, **kwargs):
        super().__init__(*args, **kwargs)
        self.voice_equiv_threshold = voice_equiv_threshold
        self.voice_cluster_threshold = voice_cluster_threshold
        self.extract_character_names = extract_character_names
        self.character_names = {}   # character_id -> name
        self.voice_names = {}       # voice_node_id -> name (raw extraction)
        self.rejected_equivalences = []  # track rejected equivalences
        self.clustered_merges = []       # track post-hoc merges

    def __getattr__(self, name):
        """Fallback for attributes missing in old pkl files."""
        defaults = {
            'voice_cluster_threshold': 0.65,
            'extract_character_names': False,
            'clustered_merges': [],
        }
        if name in defaults:
            return defaults[name]
        raise AttributeError(f"'{type(self).__name__}' object has no attribute '{name}'")

    # ------------------------------------------------------------------ #
    # Embedding helpers
    # ------------------------------------------------------------------ #
    def _compute_voice_similarity(self, node_a, node_b):
        """Compute average cosine similarity between two voice nodes' embeddings."""
        embs_a = self.nodes[node_a].embeddings
        embs_b = self.nodes[node_b].embeddings
        if not embs_a or not embs_b:
            return 0.0
        arr_a = np.array([_to_float_array(e) for e in embs_a])
        arr_b = np.array([_to_float_array(e) for e in embs_b])
        sims = cosine_similarity(arr_a, arr_b)
        return float(np.mean(sims))

    def _get_character_embedding(self, voice_tags):
        """Compute mean embedding for a character from its voice tags."""
        all_embs = []
        for tag in voice_tags:
            if not tag.startswith("voice_"):
                continue
            node_id = int(tag.replace("voice_", ""))
            node = self.nodes.get(node_id)
            if node and node.embeddings:
                for e in node.embeddings:
                    all_embs.append(_to_float_array(e))
        if not all_embs:
            return None
        return np.mean(all_embs, axis=0)

    # ------------------------------------------------------------------ #
    # Phase 2: Post-hoc voice clustering
    # ------------------------------------------------------------------ #
    def _posthoc_voice_clustering(self, character_mappings, threshold):
        """
        Merge characters whose voice embeddings are jointly similar.
        Uses conservative merging: only merges two characters if the AVERAGE
        pairwise similarity between their representative embeddings >= threshold.
        Processes pairs greedily from highest to lowest similarity.
        Does NOT use transitive union-find to avoid chain effects.
        """
        voice_chars = {}
        for cid, tags in character_mappings.items():
            emb = self._get_character_embedding(tags)
            if emb is not None:
                voice_chars[cid] = emb

        if len(voice_chars) < 2:
            return character_mappings, []

        char_ids = list(voice_chars.keys())
        embeddings = np.array([voice_chars[c] for c in char_ids])
        sim_matrix = cosine_similarity(embeddings)

        # Collect all candidate pairs sorted by similarity (highest first)
        candidates = []
        for i in range(len(char_ids)):
            for j in range(i + 1, len(char_ids)):
                if sim_matrix[i][j] >= threshold:
                    candidates.append((sim_matrix[i][j], char_ids[i], char_ids[j]))
        candidates.sort(reverse=True)

        # Greedy merge: merge a pair only if the merged group's average
        # embedding still has high similarity to both original groups
        merged_into = {}  # char_id -> group_id
        groups = {}       # group_id -> set of char_ids
        next_group = 0

        for _, ci, cj in candidates:
            gi = merged_into.get(ci)
            gj = merged_into.get(cj)

            if gi is not None and gj is not None:
                if gi == gj:
                    continue  # already same group
                # Check if merging these two groups is safe:
                # compute similarity between the merged group embeddings
                members_i = groups[gi]
                members_j = groups[gj]
                all_embs_i = [voice_chars[c] for c in members_i]
                all_embs_j = [voice_chars[c] for c in members_j]
                cross_sims = cosine_similarity(all_embs_i, all_embs_j)
                if np.min(cross_sims) >= threshold * 0.9:
                    # Merge groups
                    for c in members_j:
                        merged_into[c] = gi
                    groups[gi] = members_i | members_j
                    del groups[gj]
            elif gi is not None:
                # Check cj against all members of group gi
                sims_to_group = [cosine_similarity(
                    [voice_chars[cj]], [voice_chars[c]])[0][0]
                    for c in groups[gi]]
                if min(sims_to_group) >= threshold * 0.9:
                    merged_into[cj] = gi
                    groups[gi].add(cj)
            elif gj is not None:
                sims_to_group = [cosine_similarity(
                    [voice_chars[ci]], [voice_chars[c]])[0][0]
                    for c in groups[gj]]
                if min(sims_to_group) >= threshold * 0.9:
                    merged_into[ci] = gj
                    groups[gj].add(ci)
            else:
                # Both ungrouped, create new group
                groups[next_group] = {ci, cj}
                merged_into[ci] = next_group
                merged_into[cj] = next_group
                next_group += 1

        # Build merged character_mappings
        merged_pairs = []
        new_mappings = {}
        char_idx = 0

        # First add merged groups
        used = set()
        for gid, members in sorted(groups.items()):
            tags = []
            for c in sorted(members):
                tags.extend(character_mappings[c])
                used.add(c)
            new_mappings[f"character_{char_idx}"] = tags
            if len(members) > 1:
                members_list = sorted(members)
                for i in range(1, len(members_list)):
                    sim_val = cosine_similarity(
                        [voice_chars[members_list[0]]],
                        [voice_chars[members_list[i]]])[0][0]
                    merged_pairs.append((members_list[0], members_list[i], float(sim_val)))
            char_idx += 1

        # Then add unmerged characters
        for cid in sorted(character_mappings.keys()):
            if cid not in used:
                new_mappings[f"character_{char_idx}"] = character_mappings[cid]
                char_idx += 1

        if merged_pairs:
            logger.info(f"Post-hoc clustering: merged {len(merged_pairs)} pairs, "
                        f"{len(character_mappings)} -> {len(new_mappings)} characters")
            for ci, cj, sim in merged_pairs:
                logger.info(f"  Merged {ci} + {cj} (sim={sim:.4f})")

        return new_mappings, merged_pairs

    # ------------------------------------------------------------------ #
    # Name extraction
    # ------------------------------------------------------------------ #
    def extract_names(self):
        """
        Scan all semantic memory nodes for name mentions linked to voice IDs.
        Must be called AFTER refresh_equivalences().
        """
        self.voice_names = {}

        for node_id, node in self.nodes.items():
            if node.type != 'semantic':
                continue
            contents = node.metadata.get('contents', [])
            if not contents:
                continue
            content = contents[0]
            if not isinstance(content, str):
                continue

            for pattern in _NAME_PATTERNS:
                match = pattern.search(content)
                if match:
                    voice_id = int(match.group(1))
                    name = match.group(2)
                    if voice_id in self.nodes and self.nodes[voice_id].type == 'voice':
                        self.voice_names[voice_id] = name
                    break

        # Map voice names to character level
        self.character_names = {}
        if hasattr(self, 'reverse_character_mappings'):
            for voice_id, name in self.voice_names.items():
                voice_tag = f"voice_{voice_id}"
                char_id = self.reverse_character_mappings.get(voice_tag)
                if char_id and char_id not in self.character_names:
                    self.character_names[char_id] = name

        logger.info(f"Extracted {len(self.voice_names)} voice names, "
                    f"mapped to {len(self.character_names)} characters: {self.character_names}")

    # ------------------------------------------------------------------ #
    # Main override
    # ------------------------------------------------------------------ #
    def refresh_equivalences(self):
        """
        Override: same as parent but adds:
        1. Voice embedding verification (rejects low-sim LLM equivalences)
        2. Post-hoc voice clustering (merges high-sim voices LLM missed)
        3. Optional name extraction from semantic memories
        """
        parent = {}
        rank = {}

        def find(x):
            if x not in parent:
                parent[x] = x
                rank[x] = 0
                return x
            if parent[x] != x:
                parent[x] = find(parent[x])
            return parent[x]

        def union(x, y):
            px, py = find(x), find(y)
            if px == py:
                return
            if rank[px] < rank[py]:
                px, py = py, px
            parent[py] = px
            if rank[px] == rank[py]:
                rank[px] += 1

        # Collect equivalence nodes (same logic as parent)
        filtered_equivalence_nodes = []
        for node_id in self.nodes:
            if self.nodes[node_id].type == 'voice':
                filtered_semantic_nodes = self.fix_collisions(node_id, mode='eq_only')
                filtered_equivalence_nodes.extend([
                    node for node in filtered_semantic_nodes
                    if self.nodes[node].metadata['contents'][0].lower().startswith("equivalence")
                ])
            elif self.nodes[node_id].type == 'img':
                connected_semantic_nodes = self.get_connected_nodes(node_id, type=['semantic'])
                filtered_equivalence_nodes.extend([
                    node for node in connected_semantic_nodes
                    if self.nodes[node].metadata['contents'][0].lower().startswith("equivalence")
                ])
            else:
                continue

        filtered_equivalence_nodes = list(set(filtered_equivalence_nodes))
        equivalences = [self.nodes[node].metadata['contents'][0]
                        for node in filtered_equivalence_nodes]

        # Phase 1: LLM union-find WITH voice verification
        self.rejected_equivalences = []
        for equivalence in equivalences:
            entities = parse_video_caption(self, equivalence)
            if len(entities) >= 2:
                anchor_node = entities[0][1]
                for entity in entities[1:]:
                    target_node = entity[1]

                    if (self.nodes[anchor_node].type == 'voice' and
                            self.nodes[target_node].type == 'voice'):
                        sim = self._compute_voice_similarity(anchor_node, target_node)
                        if sim < self.voice_equiv_threshold:
                            self.rejected_equivalences.append({
                                'equivalence': equivalence,
                                'anchor': anchor_node,
                                'target': target_node,
                                'similarity': sim,
                            })
                            logger.info(
                                f"Rejected equivalence: voice_{anchor_node} <-> voice_{target_node} "
                                f"(sim={sim:.4f} < {self.voice_equiv_threshold})"
                            )
                            continue

                    union(anchor_node, target_node)

        # Build initial character mappings
        character_mappings = {}
        character_count = 0
        root_to_character = {}

        for x in parent:
            root = find(x)
            tag = f"face_{x}" if self.nodes[x].type == 'img' else f"voice_{x}"
            if root not in root_to_character:
                root_to_character[root] = f"character_{character_count}"
                character_count += 1
            character = root_to_character[root]
            if character not in character_mappings:
                character_mappings[character] = []
            character_mappings[character].append(tag)

        for x in self.nodes:
            if x in parent or self.nodes[x].type not in ['img', 'voice']:
                continue
            root = find(x)
            tag = f"face_{x}" if self.nodes[x].type == 'img' else f"voice_{x}"
            if root not in root_to_character:
                root_to_character[root] = f"character_{character_count}"
                character_count += 1
            character = root_to_character[root]
            if character not in character_mappings:
                character_mappings[character] = []
            character_mappings[character].append(tag)

        logger.info(f"LLM union-find: {character_count} characters "
                    f"(rejected {len(self.rejected_equivalences)} equivalences)")

        # Phase 2: Post-hoc voice clustering
        character_mappings, merged_pairs = self._posthoc_voice_clustering(
            character_mappings, threshold=self.voice_cluster_threshold
        )
        self.clustered_merges = merged_pairs

        # Create reverse mapping
        reverse_character_mappings = {}
        for character, tags in character_mappings.items():
            for tag in tags:
                reverse_character_mappings[tag] = character

        self.character_mappings = character_mappings
        self.reverse_character_mappings = reverse_character_mappings

        logger.info(f"Final: {len(character_mappings)} characters")

        # Match the original audio-only behavior unless name extraction is
        # explicitly requested for an experiment.
        self.voice_names = {}
        self.character_names = {}
        if self.extract_character_names:
            self.extract_names()
