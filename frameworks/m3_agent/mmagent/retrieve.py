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
import json
import math
import re
import logging
import random
import hashlib
import os
import urllib.error
import urllib.request
from difflib import SequenceMatcher
from collections import defaultdict
from .utils.chat_api import (
    generate_messages,
    get_response_with_retry,
    parallel_get_embedding,
    get_embedding_with_retry,
)
from .utils.usage_logger import log_api_usage
from .utils.general import validate_and_fix_python_list
from .prompts import *
from .memory_processing import parse_video_caption
from sklearn.metrics.pairwise import cosine_similarity
import numpy as np

processing_config = json.load(open("configs/processing_config.json"))
MAX_RETRIES = processing_config["max_retries"]
# Configure logging
logger = logging.getLogger(__name__)

EVIDENCE_ROLE_DEFINITIONS = {
    "entity_identity": "Who or which entity/person/object is involved.",
    "action_event": "What action, event, interaction, or activity happened.",
    "state_attribute": "An object's state, attribute, appearance, result, or condition.",
    "count_instance": "A distinct repeated instance/item needed for counting or enumeration.",
    "temporal_order": "Before/after/order/change over time, or evidence from different moments.",
    "spatial_location": "Where something happened, scene context, or layout/location evidence.",
}

_ROLE_ALIASES = {
    "entity": "entity_identity",
    "identity": "entity_identity",
    "person": "entity_identity",
    "object": "entity_identity",
    "actor": "entity_identity",
    "action": "action_event",
    "event": "action_event",
    "activity": "action_event",
    "state": "state_attribute",
    "attribute": "state_attribute",
    "appearance": "state_attribute",
    "result": "state_attribute",
    "count": "count_instance",
    "counting": "count_instance",
    "instance": "count_instance",
    "number": "count_instance",
    "temporal": "temporal_order",
    "time": "temporal_order",
    "order": "temporal_order",
    "sequence": "temporal_order",
    "location": "spatial_location",
    "spatial": "spatial_location",
    "place": "spatial_location",
    "scene": "spatial_location",
}

_VOICE_NAME_PATTERNS = [
    re.compile(r"<voice_(\d+)>'s\s+name\s+is\s+([A-Z][a-zA-Z\-\' ]{0,40})"),
    re.compile(r"<voice_(\d+)>\s+is\s+named\s+([A-Z][a-zA-Z\-\' ]{0,40})"),
    re.compile(r"<voice_(\d+)>\s+introduces\s+(?:themselves|herself|himself)\s+as\s+([A-Z][a-zA-Z\-\' ]{0,40})"),
]
_CHARACTER_NAME_PATTERNS = [
    re.compile(r"<character_(\d+)>'s\s+name\s+is\s+([A-Z][a-zA-Z\-\' ]{0,40})"),
    re.compile(r"<character_(\d+)>\s+is\s+named\s+([A-Z][a-zA-Z\-\' ]{0,40})"),
]
_CHARACTER_ID_QUERY = re.compile(
    r"what\s+is\s+the\s+character\s+id\s+of\s+(.+?)[\?\.!\s]*$",
    re.IGNORECASE,
)
_CHARACTER_NAME_QUERY = re.compile(
    r"what\s+is\s+the\s+name\s+of\s+(<character_\d+>)[\?\.!\s]*$",
    re.IGNORECASE,
)


def _normalize_person_name(name):
    cleaned = re.sub(r"[^A-Za-z0-9\-\' ]+", "", (name or "").strip().lower())
    return re.sub(r"\s+", " ", cleaned).strip()


def _name_to_voice_nodes(video_graph):
    """Build weak supervision map: spoken names -> related voice node IDs."""
    mapping = defaultdict(set)
    name_patterns = [
        ("voice", re.compile(r"<voice_(\d+)>[^\n]*?name is\s+([A-Za-z][A-Za-z\-\' ]{0,40})", re.IGNORECASE)),
        ("character", re.compile(r"<character_(\d+)>[^\n]*?name is\s+([A-Za-z][A-Za-z\-\' ]{0,40})", re.IGNORECASE)),
        ("face", re.compile(r"<face_(\d+)>[^\n]*?name is\s+([A-Za-z][A-Za-z\-\' ]{0,40})", re.IGNORECASE)),
    ]
    for node in video_graph.nodes.values():
        if node.type not in {"semantic", "episodic"}:
            continue
        contents = node.metadata.get("contents", [])
        for text in contents:
            if not isinstance(text, str):
                continue
            for ent_type, pat in name_patterns:
                for m in pat.finditer(text):
                    ent_id = int(m.group(1))
                    name = _normalize_person_name(m.group(2))
                    if not name:
                        continue
                    if ent_type == "voice":
                        if ent_id in video_graph.nodes and video_graph.nodes[ent_id].type == "voice":
                            mapping[name].add(ent_id)
                    elif ent_type == "character":
                        key = f"character_{ent_id}"
                        for tag in video_graph.character_mappings.get(key, []):
                            if tag.startswith("voice_"):
                                voice_id = int(tag.split("_", 1)[1])
                                if voice_id in video_graph.nodes and video_graph.nodes[voice_id].type == "voice":
                                    mapping[name].add(voice_id)
                    else:
                        # Face name evidence: map through current character mapping when possible,
                        # then fallback to direct graph connectivity.
                        face_tag = f"face_{ent_id}"
                        character_id = video_graph.reverse_character_mappings.get(face_tag)
                        if character_id:
                            for tag in video_graph.character_mappings.get(character_id, []):
                                if tag.startswith("voice_"):
                                    voice_id = int(tag.split("_", 1)[1])
                                    if voice_id in video_graph.nodes and video_graph.nodes[voice_id].type == "voice":
                                        mapping[name].add(voice_id)
                        if ent_id in video_graph.nodes and video_graph.nodes[ent_id].type == "img":
                            for voice_id in video_graph.get_connected_nodes(ent_id, type=["voice"]):
                                if voice_id in video_graph.nodes and video_graph.nodes[voice_id].type == "voice":
                                    mapping[name].add(voice_id)
    return mapping


def _build_identity_hint_cache(video_graph):
    cached = getattr(video_graph, "_identity_hint_cache", None)
    if cached is not None:
        return cached

    name_to_characters = defaultdict(set)
    character_to_names = defaultdict(set)

    def add_character_name(character_id, raw_name):
        normalized = _normalize_person_name(raw_name)
        if not normalized:
            return
        name_to_characters[normalized].add(character_id)
        character_to_names[character_id].add(raw_name.strip())

    for character_id, raw_name in getattr(video_graph, "character_names", {}).items():
        add_character_name(character_id, raw_name)

    for voice_id, raw_name in getattr(video_graph, "voice_names", {}).items():
        character_id = video_graph.reverse_character_mappings.get(f"voice_{voice_id}")
        if character_id:
            add_character_name(character_id, raw_name)

    for node in video_graph.nodes.values():
        if node.type not in {"semantic", "episodic"}:
            continue
        contents = node.metadata.get("contents", [])
        for text in contents:
            if not isinstance(text, str):
                continue
            for pattern in _VOICE_NAME_PATTERNS:
                for match in pattern.finditer(text):
                    voice_id = int(match.group(1))
                    raw_name = match.group(2).strip()
                    character_id = video_graph.reverse_character_mappings.get(f"voice_{voice_id}")
                    if character_id:
                        add_character_name(character_id, raw_name)
            for pattern in _CHARACTER_NAME_PATTERNS:
                for match in pattern.finditer(text):
                    character_id = f"character_{int(match.group(1))}"
                    raw_name = match.group(2).strip()
                    add_character_name(character_id, raw_name)

    cached = {
        "name_to_characters": {k: sorted(v) for k, v in name_to_characters.items()},
        "character_to_names": {k: sorted(v) for k, v in character_to_names.items()},
    }
    setattr(video_graph, "_identity_hint_cache", cached)
    return cached


def _score_name_match(query_name, candidate_name):
    query_norm = _normalize_person_name(query_name)
    candidate_norm = _normalize_person_name(candidate_name)
    if not query_norm or not candidate_norm:
        return 0.0
    if query_norm == candidate_norm:
        return 1.0
    if len(query_norm) >= 3 and query_norm in candidate_norm:
        return 0.95
    if len(candidate_norm) >= 3 and candidate_norm in query_norm:
        return 0.95
    prefix_len = 0
    for a, b in zip(query_norm, candidate_norm):
        if a != b:
            break
        prefix_len += 1
    ratio = SequenceMatcher(None, query_norm, candidate_norm).ratio()
    if prefix_len >= 2 and ratio >= 0.45:
        return ratio
    return 0.0


def get_identity_hints(video_graph, query):
    cache = _build_identity_hint_cache(video_graph)

    character_id_match = _CHARACTER_ID_QUERY.match(query.strip())
    if character_id_match:
        query_name = character_id_match.group(1).strip()
        candidates = []
        for candidate_name, character_ids in cache["name_to_characters"].items():
            score = _score_name_match(query_name, candidate_name)
            if score <= 0:
                continue
            for character_id in character_ids:
                candidates.append((score, candidate_name, character_id))
        candidates.sort(key=lambda item: (-item[0], item[2], item[1]))
        hints = []
        seen = set()
        for _, candidate_name, character_id in candidates:
            key = (candidate_name, character_id)
            if key in seen:
                continue
            seen.add(key)
            hints.append(f"{query_name} may refer to {character_id}, who may have been named {candidate_name.title()}.")
        return hints

    character_name_match = _CHARACTER_NAME_QUERY.match(query.strip())
    if character_name_match:
        character_token = character_name_match.group(1)
        character_id = character_token.strip("<>")
        names = cache["character_to_names"].get(character_id, [])
        return [f"{character_token} may have been named {name}." for name in names]

    return []


def infer_speaker_nodes_from_query(video_graph, query):
    name_map = getattr(video_graph, "_speaker_name_map_cache", None)
    if name_map is None:
        name_map = _name_to_voice_nodes(video_graph)
        setattr(video_graph, "_speaker_name_map_cache", name_map)
    query_l = f" {_normalize_person_name(query)} "
    if not query_l.strip():
        return set()
    selected = set()
    for name, voice_ids in name_map.items():
        if not name:
            continue
        if f" {name} " in query_l:
            selected.update(voice_ids)
    return selected


def _apply_speaker_bias(video_graph, nodes, speaker_nodes, speaker_bias=0.0, speaker_hard_filter=False):
    if not speaker_nodes:
        return nodes
    rescored = []
    bias = max(0.0, float(speaker_bias))
    for node_id, node_score in nodes:
        connected_voices = set(video_graph.get_connected_nodes(node_id, type=["voice"]))
        hit = bool(connected_voices & speaker_nodes)
        if speaker_hard_filter and not hit:
            continue
        if hit and bias > 0:
            node_score = float(node_score) * (1.0 + bias)
        rescored.append((node_id, float(node_score)))
    if not rescored:
        return nodes
    return sorted(rescored, key=lambda x: x[1], reverse=True)


def _normalize_vector(vec):
    arr = np.asarray(vec, dtype=np.float32)
    if arr.ndim != 1:
        arr = arr.reshape(-1)
    norm = np.linalg.norm(arr)
    if norm <= 1e-12:
        return arr
    return arr / norm


def _node_repr_vector(video_graph, node_id):
    raw = np.asarray(video_graph.nodes[node_id].embeddings, dtype=np.float32)
    if raw.size == 0:
        return None
    if raw.ndim == 1:
        vec = raw
    else:
        vec = np.mean(raw.reshape(-1, raw.shape[-1]), axis=0)
    return _normalize_vector(vec)


def _clip_cluster_decay_weight(rank_idx):
    if rank_idx == 0:
        return 1.0
    if rank_idx == 1:
        return 0.35
    if rank_idx == 2:
        return 0.15
    return 0.05


def _group_clip_nodes(video_graph, node_hits, max_nodes=8, sim_threshold=0.85):
    if not node_hits:
        return []

    sorted_hits = sorted(node_hits, key=lambda x: x[1], reverse=True)[: max(1, int(max_nodes))]
    clusters = []

    for node_id, node_score in sorted_hits:
        vec = _node_repr_vector(video_graph, node_id)
        if vec is None:
            continue

        best_idx = None
        best_sim = -1.0
        for idx, cluster in enumerate(clusters):
            sim = float(np.dot(vec, cluster["centroid"]))
            if sim >= sim_threshold and sim > best_sim:
                best_idx = idx
                best_sim = sim

        node_type = getattr(video_graph.nodes[node_id], "type", "")
        weight = max(float(node_score), 1e-3)
        if best_idx is None:
            clusters.append(
                {
                    "leader_id": node_id,
                    "leader_score": float(node_score),
                    "leader_type": node_type,
                    "members": [node_id],
                    "types": {node_type},
                    "weight_sum": weight,
                    "weighted_vec_sum": vec * weight,
                    "centroid": vec.copy(),
                }
            )
            continue

        cluster = clusters[best_idx]
        cluster["members"].append(node_id)
        cluster["types"].add(node_type)
        cluster["weight_sum"] += weight
        cluster["weighted_vec_sum"] += vec * weight
        cluster["centroid"] = _normalize_vector(cluster["weighted_vec_sum"] / cluster["weight_sum"])
        if float(node_score) > cluster["leader_score"]:
            cluster["leader_id"] = node_id
            cluster["leader_score"] = float(node_score)
            cluster["leader_type"] = node_type

    return sorted(clusters, key=lambda x: x["leader_score"], reverse=True)


def _compute_clip_base_score(clusters):
    if not clusters:
        return 0.0

    leaders = sorted(clusters, key=lambda x: x["leader_score"], reverse=True)
    score = 0.0
    for idx, cluster in enumerate(leaders[:4]):
        score += _clip_cluster_decay_weight(idx) * float(cluster["leader_score"])

    top_types = {cluster["leader_type"] for cluster in leaders[:3] if cluster.get("leader_type")}
    if "episodic" in top_types and "semantic" in top_types:
        score += 0.05
    if len(leaders) >= 2:
        score += min(0.03 * (len(leaders) - 1), 0.09)
    return float(score)


def _build_clip_representation(clusters):
    if not clusters:
        return None

    vecs = []
    weights = []
    for idx, cluster in enumerate(sorted(clusters, key=lambda x: x["leader_score"], reverse=True)[:4]):
        vec = cluster.get("centroid")
        if vec is None:
            continue
        weight = _clip_cluster_decay_weight(idx) * max(float(cluster["leader_score"]), 1e-3)
        vecs.append(np.asarray(vec, dtype=np.float32))
        weights.append(weight)

    if not vecs:
        return None

    weighted = np.average(np.stack(vecs, axis=0), axis=0, weights=np.asarray(weights, dtype=np.float32))
    return _normalize_vector(weighted)


def _mmr_select_clips(candidate_ids, clip_scores, clip_repr_map, topk=2, mmr_lambda=0.75):
    if not candidate_ids:
        return []

    topk = max(1, int(topk))
    mmr_lambda = float(np.clip(mmr_lambda, 0.0, 1.0))
    raw_scores = [float(clip_scores.get(cid, 0.0)) for cid in candidate_ids]
    s_min = min(raw_scores) if raw_scores else 0.0
    s_max = max(raw_scores) if raw_scores else 0.0
    if s_max > s_min:
        rel_scores = {cid: (float(clip_scores.get(cid, 0.0)) - s_min) / (s_max - s_min) for cid in candidate_ids}
    else:
        rel_scores = {cid: 1.0 for cid in candidate_ids}

    selected = []
    remaining = list(candidate_ids)

    while remaining and len(selected) < topk:
        best_id = None
        best_score = -1e9
        for cid in remaining:
            relevance = rel_scores.get(cid, 0.0)
            redundancy = 0.0
            curr_repr = clip_repr_map.get(cid)
            if curr_repr is not None and selected:
                redundancy = max(
                    float(np.dot(curr_repr, clip_repr_map[sel]))
                    for sel in selected
                    if clip_repr_map.get(sel) is not None
                ) if any(clip_repr_map.get(sel) is not None for sel in selected) else 0.0
            objective = relevance if not selected else mmr_lambda * relevance - (1.0 - mmr_lambda) * redundancy
            if objective > best_score:
                best_score = objective
                best_id = cid

        if best_id is None:
            break
        selected.append(best_id)
        remaining.remove(best_id)

    return selected


def _clip_selection_count(count, n_items, min_clips=1, max_clips=None):
    if n_items <= 0:
        return 0
    max_items = n_items if max_clips is None else min(int(max_clips), n_items)
    min_items = min(max(1, int(min_clips)), max_items)
    return int(max(min_items, min(max_items, int(count))))


def _raw_mmr_delta_root_selected_count(
    scores,
    *,
    min_clips=1,
    max_clips=None,
    normalize_by_g1=True,
):
    """Select K from MMR drops over the feasible retrieval-depth range."""
    n_scores = len(scores)
    if n_scores <= 0:
        return 0, {
            "reason": "no_candidates",
            "selected_count": 0,
            "raw_mmr_scores": [],
        }

    min_count = _clip_selection_count(min_clips, n_scores, min_clips=1, max_clips=max_clips)
    max_count = _clip_selection_count(
        n_scores if max_clips is None else max_clips,
        n_scores,
        min_clips=min_count,
        max_clips=max_clips,
    )

    eps = 1e-12
    g1 = float(scores[0])
    positive_g1 = g1 > 0.0
    feasible_max = min(max_count, n_scores - 1)
    min_decision_k = min_count
    candidate_scores = []
    for k in range(min_decision_k, feasible_max + 1):
        current = float(scores[k - 1])
        next_score = float(scores[k])
        raw_delta = current - next_score
        if normalize_by_g1:
            relative_drop = raw_delta / g1 if positive_g1 else None
            eligible = relative_drop is not None and relative_drop > 0.0
            root_score = float(relative_drop ** (1.0 / float(k))) if eligible else None
            delta = raw_delta
        else:
            delta = max(raw_delta, eps)
            relative_drop = delta
            eligible = True
            root_score = float(delta ** (1.0 / float(k)))
        candidate_scores.append(
            {
                "k": int(k),
                "current_score": current,
                "next_score": next_score,
                "raw_delta": float(raw_delta),
                "delta": float(delta),
                "normalized_delta": relative_drop,
                "eligible": bool(eligible),
                "length_normalized_score": root_score,
            }
        )

    eligible_scores = [item for item in candidate_scores if item["eligible"]]
    if eligible_scores:
        best = max(
            eligible_scores,
            key=lambda item: (
                float(item["length_normalized_score"]),
                float(item["normalized_delta"]),
                -int(item["k"]),
            ),
        )
        raw_selected_count = int(best["k"])
        selected_count = min(max_count, n_scores, max(min_count, raw_selected_count))
        reason = (
            "raw_mmr_relative_drop_root_map"
            if normalize_by_g1
            else "raw_mmr_adjacent_delta_root_map_legacy"
        )
    elif candidate_scores and normalize_by_g1:
        raw_selected_count = max_count
        selected_count = max_count
        best = None
        reason = (
            "nonpositive_g1_return_max"
            if not positive_g1
            else "no_positive_relative_drop_return_max"
        )
    else:
        raw_selected_count = min_count
        selected_count = min_count
        best = None
        reason = "insufficient_k_plus_one_scores"

    decision = {
        "reason": reason,
        "selected_count": int(selected_count),
        "raw_selected_count": int(raw_selected_count),
        "raw_mmr_scores": [float(score) for score in scores],
        "min_clips": int(min_count),
        "max_clips": int(max_count),
        "min_decision_k": int(min_decision_k),
        "requires_k_plus_one": True,
        "delta_normalizer": "g1" if normalize_by_g1 else "none",
        "delta_denominator": float(g1) if normalize_by_g1 else 1.0,
        "positive_g1": bool(positive_g1),
        "candidate_scores": candidate_scores,
        "selected_candidate": best,
        "min_clamp_applied": bool(raw_selected_count < selected_count),
    }
    return selected_count, decision


_OFFICIAL_ADAPTIVE_K_RETRIEVER_CLASS = None


def _load_official_adaptive_k_retriever_class():
    """Load Megagon's official Adaptive-k Retriever without initializing embeddings."""
    global _OFFICIAL_ADAPTIVE_K_RETRIEVER_CLASS
    if _OFFICIAL_ADAPTIVE_K_RETRIEVER_CLASS is not None:
        return _OFFICIAL_ADAPTIVE_K_RETRIEVER_CLASS

    import importlib.util
    import sys
    import tempfile
    import types

    repo_dir = os.getenv(
        "ADAPTIVE_K_RETRIEVAL_REPO",
        "third_party/adaptive-k-retrieval/adaptive-k-retrieval",
    )
    retriever_path = os.path.join(repo_dir, "retriever.py")
    if not os.path.exists(retriever_path):
        raise FileNotFoundError(
            f"Official Adaptive-k retriever.py not found at {retriever_path}. "
            "Set ADAPTIVE_K_RETRIEVAL_REPO to the cloned repository source directory."
        )

    class _DummyFaissIndex:
        pass

    dummy_faiss = types.ModuleType("faiss")
    dummy_faiss.Index = _DummyFaissIndex
    dummy_faiss.IndexFlatIP = _DummyFaissIndex
    dummy_faiss.IndexFlatL2 = _DummyFaissIndex
    dummy_faiss.read_index = lambda *args, **kwargs: None
    dummy_faiss.write_index = lambda *args, **kwargs: None
    dummy_faiss.index_cpu_to_gpu = lambda *args, **kwargs: None

    class _UnusedEmbedding:
        def __init__(self, *args, **kwargs):
            raise RuntimeError(
                "M3Agent only calls official Adaptive-k thresholding on precomputed clip scores."
            )

    dummy_utils = types.ModuleType("utils")
    dummy_utils.Embedding = _UnusedEmbedding

    dummy_eval = types.ModuleType("eval")
    dummy_eval.ContextType = dict

    old_cwd = os.getcwd()
    old_modules = {name: sys.modules.get(name) for name in ("faiss", "utils", "eval")}
    old_path = list(sys.path)
    module_name = "_megagon_official_adaptive_k_retriever"
    try:
        sys.modules["faiss"] = dummy_faiss
        sys.modules["utils"] = dummy_utils
        sys.modules["eval"] = dummy_eval
        sys.path.insert(0, repo_dir)
        with tempfile.TemporaryDirectory() as tmpdir:
            with open(os.path.join(tmpdir, "config.json"), "w") as f:
                json.dump({"adaptive_rag_dir": ""}, f)
            os.chdir(tmpdir)
            spec = importlib.util.spec_from_file_location(module_name, retriever_path)
            module = importlib.util.module_from_spec(spec)
            sys.modules[module_name] = module
            spec.loader.exec_module(module)
    finally:
        os.chdir(old_cwd)
        sys.path[:] = old_path
        for name, old_module in old_modules.items():
            if old_module is None:
                sys.modules.pop(name, None)
            else:
                sys.modules[name] = old_module

    _OFFICIAL_ADAPTIVE_K_RETRIEVER_CLASS = module.Retriever
    return _OFFICIAL_ADAPTIVE_K_RETRIEVER_CLASS


def _official_adaptive_k_selected_count(
    scores,
    *,
    strategy="largest_gap",
    min_clips=1,
    max_clips=None,
    window=3,
    ignore_extreme=0.0,
    ignore_extreme_tail=0.0,
    ignore_below_median=False,
    retrieve_more=0,
):
    arr = np.asarray(scores, dtype=float)
    if arr.size == 0:
        return 0, {
            "reason": "no_candidates",
            "strategy": strategy,
            "selected_count": 0,
        }
    if arr.size == 1:
        return _clip_selection_count(1, len(arr), min_clips=min_clips, max_clips=max_clips), {
            "reason": "single_candidate",
            "strategy": strategy,
            "selected_count": 1,
            "raw_relevance_scores": [float(arr[0])],
        }

    import contextlib
    import io
    import torch

    official_strategy = strategy
    adapter_note = None
    official_window = max(2, int(window))
    if official_strategy == "moving_avg" and arr.size <= official_window:
        # The official moving-average method requires at least window + 1 scores.
        official_strategy = "largest_gap"
        adapter_note = "official_moving_avg_requires_more_scores_fallback_largest_gap"

    retriever_class = _load_official_adaptive_k_retriever_class()
    retriever = retriever_class.__new__(retriever_class)
    retriever.strategy = official_strategy
    retriever.window = official_window
    retriever.retrieve_more = retrieve_more
    retriever.ignore_extreme = ignore_extreme
    retriever.ignore_extreme_tail = ignore_extreme_tail
    retriever.ignore_below_median = bool(ignore_below_median)
    retriever.adaptive_pool_size = None

    ranked_indices = np.arange(len(arr), dtype=int)
    ranked_context = [f"clip_{idx}" for idx in ranked_indices]
    with contextlib.redirect_stdout(io.StringIO()):
        retrieved = retriever.adaptive_retrieve(
            torch.tensor(arr, dtype=torch.float32),
            ranked_indices,
            ranked_context,
        )
    official_count = len(retrieved)
    selected_count = _clip_selection_count(
        official_count,
        len(arr),
        min_clips=min_clips,
        max_clips=max_clips,
    )
    decision = {
        "reason": "official_adaptive_k",
        "strategy": strategy,
        "official_strategy_used": official_strategy,
        "adapter_note": adapter_note,
        "window": int(official_window),
        "ignore_extreme": float(ignore_extreme),
        "ignore_extreme_tail": float(ignore_extreme_tail),
        "ignore_below_median": bool(ignore_below_median),
        "retrieve_more": retrieve_more,
        "official_selected_count": int(official_count),
        "selected_count": int(selected_count),
        "raw_relevance_scores": [float(value) for value in arr],
        "source_repo": "https://github.com/megagonlabs/adaptive-k-retrieval",
    }
    return selected_count, decision


def _adaptive_k_cut_range(n_gaps, ignore_extreme=0.0, ignore_extreme_tail=0.0):
    if n_gaps <= 0:
        return 0, 0
    if isinstance(ignore_extreme, float):
        cut_start = int(n_gaps * ignore_extreme)
    else:
        cut_start = int(ignore_extreme)
    if isinstance(ignore_extreme_tail, float):
        cut_tail = int(n_gaps * ignore_extreme_tail)
    else:
        cut_tail = int(ignore_extreme_tail)

    cut_start = max(0, min(n_gaps - 1, cut_start))
    cut_tail = max(0, min(n_gaps - cut_start - 1, cut_tail))
    cut_end = n_gaps - cut_tail
    if cut_end <= cut_start:
        return 0, n_gaps
    return cut_start, cut_end


def _adaptive_k_largest_gap_threshold(scores, ignore_extreme=0.0, ignore_extreme_tail=0.0):
    arr = np.asarray(scores, dtype=float)
    if arr.size <= 1:
        return 0, {
            "reason": "single_candidate",
            "raw_diffs": [],
            "positive_drops": [],
        }

    diffs = np.diff(arr)
    cut_start, cut_end = _adaptive_k_cut_range(
        len(diffs),
        ignore_extreme=ignore_extreme,
        ignore_extreme_tail=ignore_extreme_tail,
    )
    search_diffs = diffs[cut_start:cut_end]
    if search_diffs.size == 0:
        threshold = 0
    else:
        threshold = int(np.argmin(search_diffs) + cut_start)

    return threshold, {
        "reason": "adaptive_k_largest_gap",
        "cut_start": int(cut_start),
        "cut_end": int(cut_end),
        "raw_diffs": [float(value) for value in diffs],
        "positive_drops": [float(max(0.0, -value)) for value in diffs],
        "threshold_index": int(threshold),
    }


def _adaptive_k_moving_avg_threshold(
    scores,
    *,
    window=3,
    ignore_extreme=0.0,
    ignore_extreme_tail=0.0,
):
    arr = np.asarray(scores, dtype=float)
    window = max(2, int(window))
    if arr.size < window + 1:
        threshold, info = _adaptive_k_largest_gap_threshold(
            arr,
            ignore_extreme=ignore_extreme,
            ignore_extreme_tail=ignore_extreme_tail,
        )
        info["reason"] = "adaptive_k_moving_avg_fallback_largest_gap"
        info["moving_avg_window"] = int(window)
        return threshold, info

    kernel = np.ones(window, dtype=float) / float(window)
    moving_avg = np.convolve(arr, kernel, mode="valid")
    diffs = np.diff(moving_avg)
    cut_start, cut_end = _adaptive_k_cut_range(
        len(diffs),
        ignore_extreme=ignore_extreme,
        ignore_extreme_tail=ignore_extreme_tail,
    )
    search_diffs = diffs[cut_start:cut_end]
    if search_diffs.size == 0:
        threshold = 0
    else:
        threshold = int(np.argmin(search_diffs) + cut_start)

    return threshold, {
        "reason": "adaptive_k_moving_avg",
        "moving_avg_window": int(window),
        "moving_avg_scores": [float(value) for value in moving_avg],
        "cut_start": int(cut_start),
        "cut_end": int(cut_end),
        "raw_diffs": [float(value) for value in diffs],
        "positive_drops": [float(max(0.0, -value)) for value in diffs],
        "threshold_index": int(threshold),
    }


def _adaptive_k_2diff_threshold(scores, ignore_extreme=0.0, ignore_extreme_tail=0.0):
    arr = np.asarray(scores, dtype=float)
    if arr.size <= 3:
        threshold, info = _adaptive_k_largest_gap_threshold(
            arr,
            ignore_extreme=ignore_extreme,
            ignore_extreme_tail=ignore_extreme_tail,
        )
        info["reason"] = "adaptive_k_2diff_fallback_largest_gap"
        return threshold, info

    cut_start, cut_end = _adaptive_k_cut_range(
        arr.size - 1,
        ignore_extreme=ignore_extreme,
        ignore_extreme_tail=ignore_extreme_tail,
    )
    sliced = arr[cut_start : cut_end + 1]
    first_diff = np.diff(sliced)
    second_diff = np.diff(first_diff)
    threshold = None
    if second_diff.size > 1:
        cum_min = np.minimum.accumulate(second_diff)
        mask = (second_diff[1:] > 0) & (cum_min[:-1] < 0)
        transition_indices = np.nonzero(mask)[0]
        if transition_indices.size > 0:
            threshold = int(transition_indices[0] + 2 + cut_start)

    if threshold is None:
        threshold, info = _adaptive_k_largest_gap_threshold(
            arr,
            ignore_extreme=ignore_extreme,
            ignore_extreme_tail=ignore_extreme_tail,
        )
        info["reason"] = "adaptive_k_2diff_fallback_largest_gap"
        info["first_diff"] = [float(value) for value in first_diff]
        info["second_diff"] = [float(value) for value in second_diff]
        return threshold, info

    return threshold, {
        "reason": "adaptive_k_2diff_spike",
        "cut_start": int(cut_start),
        "cut_end": int(cut_end),
        "first_diff": [float(value) for value in first_diff],
        "second_diff": [float(value) for value in second_diff],
        "threshold_index": int(threshold),
    }


def _adaptive_k_raw_selected_count(
    scores,
    *,
    strategy="largest_gap",
    min_clips=1,
    max_clips=None,
    window=3,
    ignore_extreme=0.0,
    ignore_extreme_tail=0.0,
    retrieve_more=0,
):
    arr = np.asarray(scores, dtype=float)
    if arr.size == 0:
        return 0, {
            "reason": "no_candidates",
            "strategy": strategy,
            "selected_count": 0,
        }

    if strategy == "moving_avg":
        threshold, info = _adaptive_k_moving_avg_threshold(
            arr,
            window=window,
            ignore_extreme=ignore_extreme,
            ignore_extreme_tail=ignore_extreme_tail,
        )
    elif strategy == "2diff_spike":
        threshold, info = _adaptive_k_2diff_threshold(
            arr,
            ignore_extreme=ignore_extreme,
            ignore_extreme_tail=ignore_extreme_tail,
        )
    else:
        threshold, info = _adaptive_k_largest_gap_threshold(
            arr,
            ignore_extreme=ignore_extreme,
            ignore_extreme_tail=ignore_extreme_tail,
        )

    if retrieve_more:
        if isinstance(retrieve_more, float):
            threshold = int(threshold * retrieve_more)
        else:
            threshold = int(threshold + retrieve_more)

    selected_count = _clip_selection_count(
        int(threshold) + 1,
        len(arr),
        min_clips=min_clips,
        max_clips=max_clips,
    )
    info.update(
        {
            "strategy": strategy,
            "retrieve_more": retrieve_more,
            "selected_count": int(selected_count),
            "raw_relevance_scores": [float(value) for value in arr],
        }
    )
    return selected_count, info


def _full_adaptive_k_select_from_nodes(
    video_graph,
    nodes,
    *,
    before_clip=None,
    excluded_clips=None,
    strategy="largest_gap",
    ignore_extreme=0.0,
    ignore_extreme_tail=0.1,
    ignore_below_median=False,
    retrieve_more=5,
    candidate_nodes=None,
    min_nodes=1,
    max_nodes=None,
    min_clips=0,
    max_clips=None,
    extra_clips=0,
    trace=None,
):
    """Run official Adaptive-k on raw node similarities, then map selected nodes to clips."""
    excluded_clips = set(excluded_clips or [])
    all_ranked_nodes = []

    for node_id, node_score in nodes:
        try:
            clip_id = int(video_graph.nodes[node_id].metadata["timestamp"])
        except Exception:
            continue
        if before_clip is not None and clip_id > before_clip:
            continue
        if clip_id in excluded_clips:
            continue
        score = float(node_score)
        all_ranked_nodes.append((node_id, score, clip_id))

    original_node_count = len(all_ranked_nodes)
    if candidate_nodes is not None and int(candidate_nodes) > 0:
        ranked_nodes = all_ranked_nodes[: int(candidate_nodes)]
    else:
        ranked_nodes = all_ranked_nodes

    clip_node_hits = defaultdict(list)
    for node_id, score, clip_id in ranked_nodes:
        clip_node_hits[clip_id].append((node_id, score))

    if not ranked_nodes:
        decision = {
            "reason": "no_node_candidates",
            "strategy": strategy,
            "original_node_count": int(original_node_count),
            "candidate_node_limit": None if candidate_nodes is None else int(candidate_nodes),
            "selected_node_count": 0,
            "min_clips": None if min_clips is None else int(min_clips),
            "max_clips": None if max_clips is None else int(max_clips),
            "selected_count": 0,
        }
        if trace is not None:
            trace.update(decision)
            trace["selected"] = []
            trace["selected_nodes"] = []
        return [], clip_node_hits, decision

    scores = [score for _, score, _ in ranked_nodes]
    selected_node_count, decision = _official_adaptive_k_selected_count(
        scores,
        strategy=strategy,
        min_clips=max(1, int(min_nodes)),
        max_clips=max_nodes,
        ignore_extreme=ignore_extreme,
        ignore_extreme_tail=ignore_extreme_tail,
        ignore_below_median=ignore_below_median,
        retrieve_more=retrieve_more,
    )

    selected_nodes = ranked_nodes[:selected_node_count]
    selected_clips = []
    seen_clips = set()
    clip_cap = None if max_clips is None or int(max_clips) <= 0 else int(max_clips)
    min_clip_count = 0 if min_clips is None else max(0, int(min_clips))
    if clip_cap is not None:
        min_clip_count = min(min_clip_count, clip_cap)
    for node_id, score, clip_id in selected_nodes:
        if clip_id in seen_clips:
            continue
        selected_clips.append(clip_id)
        seen_clips.add(clip_id)
        if clip_cap is not None and len(selected_clips) >= clip_cap:
            break
    if len(selected_clips) < min_clip_count:
        for node_id, score, clip_id in ranked_nodes[selected_node_count:]:
            if clip_id in seen_clips:
                continue
            selected_clips.append(clip_id)
            seen_clips.add(clip_id)
            if len(selected_clips) >= min_clip_count:
                break
    extra_clip_count = max(0, int(extra_clips))
    extra_added = []
    if extra_clip_count > 0:
        for node_id, score, clip_id in all_ranked_nodes:
            if clip_cap is not None and len(selected_clips) >= clip_cap:
                break
            if clip_id in seen_clips:
                continue
            selected_clips.append(clip_id)
            seen_clips.add(clip_id)
            extra_added.append(clip_id)
            clip_node_hits[clip_id].append((node_id, score))
            if len(extra_added) >= extra_clip_count:
                break

    decision.update(
        {
            "reason": "full_official_adaptive_k_node_curve",
            "original_node_count": int(original_node_count),
            "candidate_node_limit": None if candidate_nodes is None else int(candidate_nodes),
            "candidate_node_count": int(len(ranked_nodes)),
            "candidate_clip_count": int(len(clip_node_hits)),
            "selected_node_count": int(selected_node_count),
            "min_clips": None if min_clips is None else int(min_clips),
            "max_clips": None if max_clips is None else int(max_clips),
            "extra_clips": int(extra_clip_count),
            "extra_added": [int(cid) for cid in extra_added],
            "selected_count": int(len(selected_clips)),
            "selected_clips": [int(cid) for cid in selected_clips],
            "selected_nodes": [
                {
                    "node_id": int(node_id),
                    "clip": int(clip_id),
                    "score": float(score),
                }
                for node_id, score, clip_id in selected_nodes
            ],
            "top_node_scores": [float(value) for value in scores[:200]],
            "clip_mapping": "unique_clips_from_selected_nodes_preserve_node_rank",
        }
    )

    if trace is not None:
        trace.update(decision)
        trace["selected"] = [int(cid) for cid in selected_clips]

    return selected_clips, clip_node_hits, decision


def _clip_adaptive_k_select_from_scores(
    candidate_ids,
    clip_scores,
    *,
    strategy="largest_gap",
    ignore_extreme=0.0,
    ignore_extreme_tail=0.1,
    ignore_below_median=False,
    retrieve_more=5,
    min_clips=1,
    max_clips=None,
    extra_clips=0,
    trace=None,
):
    """Run official Adaptive-k on an already-ranked clip-candidate score curve."""
    if not candidate_ids:
        decision = {
            "reason": "no_clip_candidates",
            "strategy": strategy,
            "selected_count": 0,
        }
        if trace is not None:
            trace.update(decision)
            trace["selected"] = []
        return [], decision

    scores = [float(clip_scores.get(cid, 0.0)) for cid in candidate_ids]
    selected_count, decision = _official_adaptive_k_selected_count(
        scores,
        strategy=strategy,
        min_clips=min_clips,
        max_clips=max_clips,
        ignore_extreme=ignore_extreme,
        ignore_extreme_tail=ignore_extreme_tail,
        ignore_below_median=ignore_below_median,
        retrieve_more=retrieve_more,
    )
    selected_clips = list(candidate_ids[:selected_count])
    extra_added = []
    extra_limit = None if max_clips is None else int(max_clips)
    if int(extra_clips) > 0 and (extra_limit is None or len(selected_clips) < extra_limit):
        selected_set = set(selected_clips)
        for clip_id in candidate_ids:
            if clip_id in selected_set:
                continue
            selected_clips.append(clip_id)
            selected_set.add(clip_id)
            extra_added.append(clip_id)
            if len(extra_added) >= int(extra_clips):
                break
            if extra_limit is not None and len(selected_clips) >= extra_limit:
                break
        if extra_limit is not None:
            selected_clips = selected_clips[:extra_limit]
    decision.update(
        {
            "reason": "top_candidate_clip_official_adaptive_k",
            "candidate_clip_count": int(len(candidate_ids)),
            "candidate_ids": [int(cid) for cid in candidate_ids],
            "candidate_scores": [float(score) for score in scores],
            "extra_clips": int(extra_clips),
            "extra_added": [int(cid) for cid in extra_added],
            "selected_count": int(len(selected_clips)),
            "selected_clips": [int(cid) for cid in selected_clips],
        }
    )
    if trace is not None:
        trace.update(decision)
        trace["selected"] = [int(cid) for cid in selected_clips]
    return selected_clips, decision


def _dynamicrag_completion_url(api_base):
    base = str(api_base or "").strip().rstrip("/")
    if not base:
        raise ValueError(
            "DynamicRAG API base is empty. Start a vLLM OpenAI-compatible server "
            "for gasolsun/DynamicRAG-8B and pass --dynamicrag_api_base."
        )
    if base.endswith("/chat/completions"):
        return base
    if base.endswith("/v1"):
        return f"{base}/chat/completions"
    return f"{base}/v1/chat/completions"


def _dynamicrag_get_prompt_docs(query, docs):
    """Prompt format copied from GasolSun36/DynamicRAG top_inference.py."""
    retrieved_content = "Retrieved Content:\n" + "\n".join(
        [
            f"[{i + 1}] Document {i + 1}\nContent: {doc['text']}"
            for i, doc in enumerate(docs)
        ]
    )
    return [
        {
            "role": "system",
            "content": (
                "You are an expert at dynamically generating document identifiers to answer a given query.\n"
                "I will provide you with a set of documents, each uniquely identified by a number within square brackets, e.g., [1], [2], etc.\n"
                "Your task is to identify and generate only the identifiers of the documents that contain sufficient information to answer the query.\n"
                "Stop generating identifiers as soon as the selected documents collectively provide enough information to answer the query.\n"
                "If no documents are required to answer the query, output \"None\".\n"
                "Output the identifiers as a comma-separated list, e.g., [1], [2] or \"None\" if no documents are needed.\n"
                "Focus solely on providing the identifiers. Do not include any explanations, descriptions, or additional text."
            ),
        },
        {"role": "user", "content": f"Query: {query}\n{retrieved_content}"},
    ]


def _dynamicrag_chat_select(
    messages,
    *,
    model,
    api_base,
    api_key="EMPTY",
    temperature=0.4,
    max_tokens=100,
    timeout=60,
):
    payload = {
        "model": model,
        "messages": messages,
        "temperature": float(temperature),
        "max_tokens": int(max_tokens),
    }
    body = json.dumps(payload).encode("utf-8")
    headers = {"Content-Type": "application/json"}
    if api_key:
        headers["Authorization"] = f"Bearer {api_key}"
    request = urllib.request.Request(
        _dynamicrag_completion_url(api_base),
        data=body,
        headers=headers,
        method="POST",
    )
    try:
        with urllib.request.urlopen(request, timeout=float(timeout)) as response:
            result = json.loads(response.read().decode("utf-8"))
    except urllib.error.HTTPError as exc:
        detail = exc.read().decode("utf-8", "ignore")
        raise RuntimeError(f"DynamicRAG server returned HTTP {exc.code}: {detail}") from exc
    except urllib.error.URLError as exc:
        raise RuntimeError(f"DynamicRAG server request failed: {exc}") from exc

    usage = result.get("usage") or {}
    usage_dict = {
        "input_tokens": int(usage.get("prompt_tokens") or usage.get("input_tokens") or 0),
        "output_tokens": int(usage.get("completion_tokens") or usage.get("output_tokens") or 0),
        "total_tokens": int(usage.get("total_tokens") or 0),
        "raw": usage,
    }
    log_api_usage(
        "dynamicrag_response",
        model,
        usage_dict,
        metadata={"api_base": str(api_base), "n_messages": len(messages)},
    )

    try:
        return result["choices"][0]["message"]["content"]
    except Exception as exc:
        raise RuntimeError(f"Unexpected DynamicRAG response format: {result}") from exc


def _dynamicrag_parse_doc_ids(text, n_docs):
    selected = []
    seen = set()
    for raw in re.findall(r"\[(\d+)\]", str(text or "")):
        idx = int(raw)
        if 1 <= idx <= n_docs and idx not in seen:
            selected.append(idx)
            seen.add(idx)
    if selected or "none" in str(text or "").lower():
        return selected

    # Be permissive for local model variants that output "1, 2" without brackets.
    for raw in re.findall(r"\b(\d+)\b", str(text or "")):
        idx = int(raw)
        if 1 <= idx <= n_docs and idx not in seen:
            selected.append(idx)
            seen.add(idx)
    return selected


def _dynamicrag_parse_clip_ids(text, valid_clip_ids):
    valid = {int(clip_id) for clip_id in valid_clip_ids}
    selected = []
    seen = set()
    for raw in re.findall(r"\[(\d+)\]|\b(\d+)\b", str(text or "")):
        value = next((part for part in raw if part), "")
        if not value:
            continue
        clip_id = int(value)
        if clip_id in valid and clip_id not in seen:
            selected.append(clip_id)
            seen.add(clip_id)
    return selected


def _dynamicrag_clip_doc_text(
    video_graph,
    clip_id,
    clip_node_hits,
    *,
    max_nodes_per_clip=4,
    max_doc_chars=1600,
):
    max_nodes = int(max_nodes_per_clip)
    node_hits = sorted(
        clip_node_hits.get(clip_id, []),
        key=lambda item: item[1],
        reverse=True,
    )
    if max_nodes > 0:
        node_hits = node_hits[:max_nodes]
    if not node_hits:
        fallback_node_ids = list(video_graph.text_nodes_by_clip.get(clip_id, []))
        if max_nodes > 0:
            fallback_node_ids = fallback_node_ids[:max_nodes]
        node_hits = [
            (node_id, 0.0)
            for node_id in fallback_node_ids
        ]

    lines = []
    for node_id, _ in node_hits:
        try:
            contents = video_graph.nodes[node_id].metadata.get("contents", [])
        except Exception:
            contents = []
        if not contents:
            continue
        translated = translate(video_graph, [str(contents[0])])
        if translated:
            lines.append(translated[0])

    text = "\n".join(line for line in lines if line).strip()
    if not text:
        text = f"Memory evidence from CLIP_{int(clip_id)}."
    if len(text) > int(max_doc_chars):
        text = text[: int(max_doc_chars)].rstrip() + "..."
    return text


def _dynamicrag_select_clips(
    video_graph,
    query,
    candidate_ids,
    clip_node_hits,
    clip_scores,
    *,
    model="gasolsun/DynamicRAG-8B",
    api_base=None,
    api_key="EMPTY",
    temperature=0.4,
    max_tokens=100,
    timeout=60,
    min_clips=0,
    max_clips=None,
    max_nodes_per_clip=4,
    max_doc_chars=1600,
    trace=None,
):
    if not candidate_ids:
        decision = {
            "reason": "no_clip_candidates",
            "policy": "dynamicrag_clip_selector",
            "selected_count": 0,
        }
        if trace is not None:
            trace.update(decision)
            trace["selected"] = []
        return [], decision

    docs = [
        {
            "title": f"CLIP_{int(clip_id)}",
            "text": _dynamicrag_clip_doc_text(
                video_graph,
                clip_id,
                clip_node_hits,
                max_nodes_per_clip=max_nodes_per_clip,
                max_doc_chars=max_doc_chars,
            ),
            "clip_id": int(clip_id),
        }
        for clip_id in candidate_ids
    ]
    messages = _dynamicrag_get_prompt_docs(query, docs)
    raw_output = _dynamicrag_chat_select(
        messages,
        model=model,
        api_base=api_base,
        api_key=api_key,
        temperature=temperature,
        max_tokens=max_tokens,
        timeout=timeout,
    )
    doc_indices = _dynamicrag_parse_doc_ids(raw_output, len(docs))
    if doc_indices:
        selected_clips = [candidate_ids[idx - 1] for idx in doc_indices]
        parsed_id_mode = "document_ordinal"
    else:
        selected_clips = _dynamicrag_parse_clip_ids(raw_output, candidate_ids)
        parsed_id_mode = "raw_clip_id_fallback" if selected_clips else "none"

    if max_clips is not None and int(max_clips) > 0:
        selected_clips = selected_clips[: int(max_clips)]
    if min_clips is not None and int(min_clips) > 0 and len(selected_clips) < int(min_clips):
        selected_set = set(selected_clips)
        for clip_id in candidate_ids:
            if clip_id in selected_set:
                continue
            selected_clips.append(clip_id)
            selected_set.add(clip_id)
            if len(selected_clips) >= int(min_clips):
                break

    decision = {
        "reason": "dynamicrag_clip_selector",
        "policy": "dynamicrag_clip_selector",
        "model": str(model),
        "api_base": str(api_base),
        "temperature": float(temperature),
        "max_tokens": int(max_tokens),
        "candidate_clip_count": int(len(candidate_ids)),
        "candidate_ids": [int(cid) for cid in candidate_ids],
        "candidate_scores": [float(clip_scores.get(cid, 0.0)) for cid in candidate_ids],
        "raw_output": raw_output,
        "parsed_id_mode": parsed_id_mode,
        "selected_doc_indices": [int(idx) for idx in doc_indices],
        "selected_count": int(len(selected_clips)),
        "selected_clips": [int(cid) for cid in selected_clips],
        "source_repo": "https://github.com/GasolSun36/DynamicRAG",
        "source_commit": "be08b2a93bea37c7b1ae6b589348cde5ee8b5c1e",
        "prompt_source": "DynamicRAG top_inference.py get_prompt_docs",
    }
    if trace is not None:
        trace.update(decision)
        trace["selected"] = [int(cid) for cid in selected_clips]
    return selected_clips, decision


def _parse_float_sequence(text, default_values):
    values = []
    for raw in re.split(r"[,\s]+", str(text or "")):
        raw = raw.strip()
        if not raw:
            continue
        try:
            values.append(float(raw))
        except ValueError:
            continue
    if not values:
        values = list(default_values)
    # DF-RAG excludes lambda=0 because it disregards query relevance.
    return sorted({float(np.clip(value, 1e-6, 1.0)) for value in values})


_ADAPTIVE_RAG_LABEL_CACHE = {}


def _adaptive_rag_normalize_text(text):
    return re.sub(r"\s+", " ", str(text or "")).strip().lower()


def _adaptive_rag_normalize_label(value, fallback="B"):
    if isinstance(value, dict):
        for key in ("option", "answer", "label", "complexity", "route", "prediction"):
            if key in value:
                label = _adaptive_rag_normalize_label(value[key], fallback=None)
                if label in {"A", "B", "C"}:
                    return label
        return str(fallback or "B").strip().upper()[:1] if fallback is not None else ""
    if isinstance(value, (int, float)) and not isinstance(value, bool):
        idx = int(value)
        if idx in (0, 1, 2):
            return ("A", "B", "C")[idx]
    text = str(value or "").strip()
    if not text:
        if fallback is None:
            return ""
        return str(fallback or "B").strip().upper()[:1] or "B"

    upper = text.upper()
    if upper in {"A", "B", "C"}:
        return upper
    if re.search(r"\bA\b", upper):
        return "A"
    if re.search(r"\bB\b", upper):
        return "B"
    if re.search(r"\bC\b", upper):
        return "C"

    lowered = text.lower()
    if any(token in lowered for token in ("zero", "no retrieval", "no-retrieval", "none")):
        return "A"
    if any(token in lowered for token in ("single", "one", "one-step", "1-step")):
        return "B"
    if any(token in lowered for token in ("multi", "complex", "multi-step", "multihop", "multi-hop")):
        return "C"
    if fallback is None:
        return ""
    return str(fallback or "B").strip().upper()[:1] or "B"


def _adaptive_rag_label_description(label):
    return {
        "A": "lowest-complexity route",
        "B": "single-step retrieval",
        "C": "multi-step retrieval",
    }.get(label, "single-step retrieval")


def _adaptive_rag_register_label(record, label_map):
    if not isinstance(record, dict):
        return
    label = _adaptive_rag_normalize_label(record, fallback=None)
    if label not in {"A", "B", "C"}:
        return
    for key in ("id", "qid", "question_id"):
        if record.get(key) is not None:
            label_map[f"id:{str(record[key])}"] = label
    question = record.get("question") or record.get("query")
    if question:
        label_map[f"question:{_adaptive_rag_normalize_text(question)}"] = label


def _adaptive_rag_load_label_map(path):
    if not path:
        return {}
    path = os.path.abspath(os.path.expanduser(str(path)))
    cached = _ADAPTIVE_RAG_LABEL_CACHE.get(path)
    if cached is not None:
        return cached

    label_map = {}
    if not os.path.exists(path):
        logger.warning("Adaptive-RAG classifier file not found: %s", path)
        _ADAPTIVE_RAG_LABEL_CACHE[path] = label_map
        return label_map

    try:
        if path.endswith(".jsonl"):
            with open(path, "r", encoding="utf-8") as f:
                for line in f:
                    if line.strip():
                        _adaptive_rag_register_label(json.loads(line), label_map)
        else:
            with open(path, "r", encoding="utf-8") as f:
                data = json.load(f)
            if isinstance(data, list):
                for record in data:
                    _adaptive_rag_register_label(record, label_map)
            elif isinstance(data, dict):
                for key, value in data.items():
                    label = _adaptive_rag_normalize_label(value, fallback=None)
                    if label in {"A", "B", "C"}:
                        label_map[f"id:{str(key)}"] = label
                        label_map[f"question:{_adaptive_rag_normalize_text(key)}"] = label
                    if isinstance(value, dict):
                        _adaptive_rag_register_label({"id": key, **value}, label_map)
    except Exception as exc:
        logger.warning("Failed to read Adaptive-RAG classifier file %s: %s", path, exc)

    _ADAPTIVE_RAG_LABEL_CACHE[path] = label_map
    return label_map


def _adaptive_rag_prompt(query):
    return [
        {
            "role": "system",
            "content": (
                "You are the query-complexity classifier from Adaptive-RAG. "
                "Classify the question into exactly one option: "
                "A = zero/no-retrieval is sufficient, "
                "B = single-step retrieval is sufficient, "
                "C = multi-step retrieval is needed. "
                "Return only A, B, or C."
            ),
        },
        {"role": "user", "content": f"Question: {query}"},
    ]


def _adaptive_rag_chat_label(
    query,
    *,
    model,
    api_base,
    api_key="EMPTY",
    temperature=0.0,
    max_tokens=4,
    timeout=60,
    fallback_label="B",
):
    payload = {
        "model": model,
        "messages": _adaptive_rag_prompt(query),
        "temperature": float(temperature),
        "max_tokens": int(max_tokens),
    }
    body = json.dumps(payload).encode("utf-8")
    headers = {"Content-Type": "application/json"}
    if api_key:
        headers["Authorization"] = f"Bearer {api_key}"
    request = urllib.request.Request(
        _dynamicrag_completion_url(api_base),
        data=body,
        headers=headers,
        method="POST",
    )
    try:
        with urllib.request.urlopen(request, timeout=float(timeout)) as response:
            result = json.loads(response.read().decode("utf-8"))
    except urllib.error.HTTPError as exc:
        detail = exc.read().decode("utf-8", "ignore")
        raise RuntimeError(f"Adaptive-RAG classifier server returned HTTP {exc.code}: {detail}") from exc
    except urllib.error.URLError as exc:
        raise RuntimeError(f"Adaptive-RAG classifier request failed: {exc}") from exc

    usage = result.get("usage") or {}
    usage_dict = {
        "input_tokens": int(usage.get("prompt_tokens") or usage.get("input_tokens") or 0),
        "output_tokens": int(usage.get("completion_tokens") or usage.get("output_tokens") or 0),
        "total_tokens": int(usage.get("total_tokens") or 0),
        "raw": usage,
    }
    log_api_usage(
        "adaptive_rag_classifier_response",
        model,
        usage_dict,
        metadata={"api_base": str(api_base)},
    )
    raw = result["choices"][0]["message"]["content"]
    return _adaptive_rag_normalize_label(raw, fallback=fallback_label), raw


def _adaptive_rag_heuristic_label(query):
    text = _adaptive_rag_normalize_text(query)
    complex_markers = [
        "compare",
        "difference",
        "differences",
        "relationship",
        "sequence",
        "before",
        "after",
        "why",
        "how many",
        "all ",
        "both",
        "three",
        "several",
        "multiple",
        "each",
        "first",
        "last",
        "then",
    ]
    if any(marker in text for marker in complex_markers):
        return "C"
    return "B"


def _adaptive_rag_predict_label(
    *,
    query,
    question=None,
    question_id=None,
    route_source="heuristic",
    classifier_path=None,
    classifier_model="adaptive-rag-classifier",
    classifier_api_base=None,
    classifier_api_key="EMPTY",
    classifier_temperature=0.0,
    classifier_max_tokens=4,
    classifier_timeout=60,
    fallback_label="B",
):
    source = str(route_source or "heuristic").strip().lower()
    route_query = question or query
    raw_output = ""

    if source in {"file", "classifier_file", "precomputed"}:
        label_map = _adaptive_rag_load_label_map(classifier_path)
        keys = []
        if question_id is not None:
            keys.append(f"id:{str(question_id)}")
        keys.append(f"question:{_adaptive_rag_normalize_text(route_query)}")
        keys.append(f"question:{_adaptive_rag_normalize_text(query)}")
        for key in keys:
            if key in label_map:
                return label_map[key], {"source": source, "matched_key": key, "raw_output": label_map[key]}
        return _adaptive_rag_normalize_label(fallback_label), {
            "source": source,
            "matched_key": None,
            "raw_output": "",
            "reason": "classifier_file_miss",
        }

    if source in {"api", "llm", "openai"}:
        label, raw_output = _adaptive_rag_chat_label(
            route_query,
            model=classifier_model,
            api_base=classifier_api_base,
            api_key=classifier_api_key,
            temperature=classifier_temperature,
            max_tokens=classifier_max_tokens,
            timeout=classifier_timeout,
            fallback_label=fallback_label,
        )
        return label, {"source": source, "raw_output": raw_output}

    if source in {"constant", "fixed"}:
        label = _adaptive_rag_normalize_label(fallback_label)
        return label, {"source": source, "raw_output": label}

    label = _adaptive_rag_heuristic_label(route_query)
    return label, {"source": "heuristic", "raw_output": label}


def _adaptive_rag_select_clips(
    candidate_ids,
    clip_scores,
    clip_repr_map,
    *,
    label,
    topk=2,
    zero_clips=0,
    single_clips=2,
    multi_clips=5,
    mmr_lambda=0.75,
    selector="mmr",
):
    label = _adaptive_rag_normalize_label(label)
    if label == "A":
        target = int(zero_clips)
    elif label == "B":
        target = int(single_clips) if int(single_clips) > 0 else int(topk)
    else:
        target = int(multi_clips) if int(multi_clips) > 0 else int(topk)

    target = max(0, min(target, len(candidate_ids)))
    if target <= 0:
        return []
    if str(selector or "mmr").lower() == "top" or not clip_repr_map:
        return sorted(candidate_ids, key=lambda cid: float(clip_scores.get(cid, 0.0)), reverse=True)[:target]
    return _mmr_select_clips(
        candidate_ids,
        clip_scores,
        clip_repr_map,
        topk=target,
        mmr_lambda=mmr_lambda,
    )


def _df_rag_chat(
    messages,
    *,
    model,
    api_base,
    api_key="EMPTY",
    temperature=0.0,
    max_tokens=512,
    timeout=60,
    stage="evaluator",
):
    payload = {
        "model": model,
        "messages": messages,
        "temperature": float(temperature),
        "max_tokens": int(max_tokens),
    }
    body = json.dumps(payload).encode("utf-8")
    headers = {"Content-Type": "application/json"}
    if api_key:
        headers["Authorization"] = f"Bearer {api_key}"
    request = urllib.request.Request(
        _dynamicrag_completion_url(api_base),
        data=body,
        headers=headers,
        method="POST",
    )
    try:
        with urllib.request.urlopen(request, timeout=float(timeout)) as response:
            result = json.loads(response.read().decode("utf-8"))
    except urllib.error.HTTPError as exc:
        detail = exc.read().decode("utf-8", "ignore")
        raise RuntimeError(f"DF-RAG {stage} server returned HTTP {exc.code}: {detail}") from exc
    except urllib.error.URLError as exc:
        raise RuntimeError(f"DF-RAG {stage} server request failed: {exc}") from exc

    usage = result.get("usage") or {}
    usage_dict = {
        "input_tokens": int(usage.get("prompt_tokens") or usage.get("input_tokens") or 0),
        "output_tokens": int(usage.get("completion_tokens") or usage.get("output_tokens") or 0),
        "total_tokens": int(usage.get("total_tokens") or 0),
        "raw": usage,
    }
    log_api_usage(
        f"df_rag_{stage}_response",
        model,
        usage_dict,
        metadata={"api_base": str(api_base), "n_messages": len(messages)},
    )

    try:
        return result["choices"][0]["message"]["content"]
    except Exception as exc:
        raise RuntimeError(f"Unexpected DF-RAG {stage} response format: {result}") from exc


def _df_rag_plan_prompt(query):
    return [
        {
            "role": "system",
            "content": (
                "You are the Planner module from DF-RAG. Decompose the question into "
                "the minimal ordered reasoning steps needed to answer it. Each step "
                "should ask for one concrete piece of evidence. Return only a JSON "
                "array of strings, with no markdown and no explanation."
            ),
        },
        {"role": "user", "content": f"Question: {query}"},
    ]


def _df_rag_parse_plan(raw_output, query):
    text = str(raw_output or "").strip()
    try:
        parsed = json.loads(text)
        if isinstance(parsed, list):
            steps = [str(item).strip() for item in parsed if str(item).strip()]
            if steps:
                return steps
    except Exception:
        pass

    steps = []
    for line in text.splitlines():
        line = re.sub(r"^\s*(?:[-*]|\d+[\.\):：-])\s*", "", line).strip()
        if line:
            steps.append(line)
    return steps or [str(query).strip()]


def _df_rag_evaluator_prompt(plan_steps, docs):
    plan = "\n".join(f"{idx + 1}) {step}" for idx, step in enumerate(plan_steps))
    chunks = "\n\n".join(
        f"Chunk {idx + 1} ({doc['title']}):\n{doc['text']}"
        for idx, doc in enumerate(docs)
    )
    return [
        {
            "role": "system",
            "content": (
                "You are the Evaluator module from DF-RAG. Given a reasoning plan "
                "and a candidate chunk set, score how well the chunks support each "
                "plan step. Use this scale for each step: 0 = no supporting evidence, "
                "1 = weak or indirect evidence, 3 = useful but incomplete evidence, "
                "5 = direct evidence sufficient for the step. Return a short line for "
                "each step and end with exactly: Total Score: <integer>."
            ),
        },
        {
            "role": "user",
            "content": (
                "Plan:\n"
                f"{plan}\n\n"
                "Chunks:\n"
                f"{chunks}\n\n"
                "For each plan step, judge only the evidence explicitly present in the chunks."
            ),
        },
    ]


def _df_rag_parse_total_score(raw_output):
    text = str(raw_output or "")
    match = re.search(r"total\s+score\s*[:：]\s*(-?\d+(?:\.\d+)?)", text, flags=re.IGNORECASE)
    if match:
        return float(match.group(1))
    scores = [
        float(value)
        for value in re.findall(r"\bscore\s*[:：]\s*(-?\d+(?:\.\d+)?)", text, flags=re.IGNORECASE)
    ]
    if scores:
        return float(sum(scores))
    numbers = [float(value) for value in re.findall(r"-?\d+(?:\.\d+)?", text)]
    return float(numbers[-1]) if numbers else 0.0


def _df_rag_gmmr_select_clips(candidate_ids, clip_scores, clip_repr_map, *, set_size=5, df_lambda=0.5):
    if not candidate_ids:
        return []

    set_size = max(1, min(int(set_size), len(candidate_ids)))
    df_lambda = float(np.clip(df_lambda, 0.0, 1.0))
    raw_scores = [float(clip_scores.get(cid, 0.0)) for cid in candidate_ids]
    s_min = min(raw_scores) if raw_scores else 0.0
    s_max = max(raw_scores) if raw_scores else 0.0
    if s_max > s_min:
        rel_scores = {
            cid: (float(clip_scores.get(cid, 0.0)) - s_min) / (s_max - s_min)
            for cid in candidate_ids
        }
    else:
        rel_scores = {cid: 1.0 for cid in candidate_ids}

    selected = []
    selected_vecs = []
    remaining = list(candidate_ids)
    while remaining and len(selected) < set_size:
        centroid = None
        if selected_vecs:
            centroid = _normalize_vector(np.mean(np.stack(selected_vecs, axis=0), axis=0))

        best_id = None
        best_score = -1e9
        for cid in remaining:
            relevance = rel_scores.get(cid, 0.0)
            diversity = 0.0
            curr_vec = clip_repr_map.get(cid)
            if centroid is not None and curr_vec is not None:
                # Normalized embeddings have Euclidean distance in [0, 2].
                diversity = float(np.linalg.norm(np.asarray(curr_vec) - centroid) / 2.0)
            objective = relevance if not selected else df_lambda * relevance + (1.0 - df_lambda) * diversity
            if objective > best_score:
                best_score = objective
                best_id = cid

        if best_id is None:
            break
        selected.append(best_id)
        best_vec = clip_repr_map.get(best_id)
        if best_vec is not None:
            selected_vecs.append(np.asarray(best_vec, dtype=np.float32))
        remaining.remove(best_id)
    return selected


def _df_rag_select_clips(
    video_graph,
    query,
    candidate_ids,
    clip_node_hits,
    clip_scores,
    clip_repr_map,
    *,
    model="models/DynamicRAG-8B",
    api_base=None,
    api_key="EMPTY",
    temperature=0.0,
    planner_max_tokens=256,
    evaluator_max_tokens=512,
    timeout=60,
    lambdas="0.1,0.2,0.3,0.4,0.5,0.6,0.7,0.8,0.9,1.0",
    set_size=5,
    max_nodes_per_clip=4,
    max_doc_chars=1600,
    fallback_lambda=0.5,
    trace=None,
):
    if not candidate_ids:
        decision = {
            "reason": "no_clip_candidates",
            "policy": "df_rag_query_aware_diversity",
            "selected_count": 0,
        }
        if trace is not None:
            trace.update(decision)
            trace["selected"] = []
        return [], decision

    lambda_values = _parse_float_sequence(
        lambdas,
        default_values=[idx / 10.0 for idx in range(1, 11)],
    )
    set_size = max(1, min(int(set_size), len(candidate_ids)))
    docs_by_clip = {
        int(clip_id): {
            "title": f"CLIP_{int(clip_id)}",
            "text": _dynamicrag_clip_doc_text(
                video_graph,
                clip_id,
                clip_node_hits,
                max_nodes_per_clip=max_nodes_per_clip,
                max_doc_chars=max_doc_chars,
            ),
            "clip_id": int(clip_id),
        }
        for clip_id in candidate_ids
    }

    raw_plan = _df_rag_chat(
        _df_rag_plan_prompt(query),
        model=model,
        api_base=api_base,
        api_key=api_key,
        temperature=temperature,
        max_tokens=planner_max_tokens,
        timeout=timeout,
        stage="planner",
    )
    plan_steps = _df_rag_parse_plan(raw_plan, query)

    candidate_sets = []
    seen_sets = {}
    for value in lambda_values:
        selected = _df_rag_gmmr_select_clips(
            candidate_ids,
            clip_scores,
            clip_repr_map,
            set_size=set_size,
            df_lambda=value,
        )
        key = tuple(int(cid) for cid in selected)
        if key in seen_sets:
            seen_sets[key].append(value)
            continue
        seen_sets[key] = [value]
        candidate_sets.append(
            {
                "lambda": float(value),
                "aliases": seen_sets[key],
                "clips": selected,
                "docs": [docs_by_clip[int(cid)] for cid in selected],
            }
        )

    evaluations = []
    for item in candidate_sets:
        raw_eval = _df_rag_chat(
            _df_rag_evaluator_prompt(plan_steps, item["docs"]),
            model=model,
            api_base=api_base,
            api_key=api_key,
            temperature=temperature,
            max_tokens=evaluator_max_tokens,
            timeout=timeout,
            stage="evaluator",
        )
        score = _df_rag_parse_total_score(raw_eval)
        evaluations.append(
            {
                "lambda": float(item["lambda"]),
                "aliases": [float(value) for value in item["aliases"]],
                "clips": [int(cid) for cid in item["clips"]],
                "score": float(score),
                "raw_output": raw_eval,
            }
        )

    if evaluations:
        best_score = max(item["score"] for item in evaluations)
        tied = [item for item in evaluations if abs(float(item["score"]) - float(best_score)) <= 1e-9]
        tied.sort(key=lambda item: item["lambda"])
        chosen = tied[len(tied) // 2]  # DF-RAG upper-median tie-breaking gate.
        selected_clips = [int(cid) for cid in chosen["clips"]]
        stop_reason = "highest_evaluator_score_upper_median_tie_break"
    else:
        selected_clips = _df_rag_gmmr_select_clips(
            candidate_ids,
            clip_scores,
            clip_repr_map,
            set_size=set_size,
            df_lambda=fallback_lambda,
        )
        chosen = {
            "lambda": float(fallback_lambda),
            "aliases": [float(fallback_lambda)],
            "clips": [int(cid) for cid in selected_clips],
            "score": 0.0,
            "raw_output": "",
        }
        stop_reason = "fallback_no_evaluations"

    decision = {
        "reason": "df_rag_query_aware_diversity",
        "policy": "df_rag_query_aware_diversity",
        "model": str(model),
        "api_base": str(api_base),
        "temperature": float(temperature),
        "planner_max_tokens": int(planner_max_tokens),
        "evaluator_max_tokens": int(evaluator_max_tokens),
        "candidate_clip_count": int(len(candidate_ids)),
        "candidate_ids": [int(cid) for cid in candidate_ids],
        "candidate_scores": [float(clip_scores.get(cid, 0.0)) for cid in candidate_ids],
        "lambda_grid": [float(value) for value in lambda_values],
        "set_size": int(set_size),
        "raw_plan": raw_plan,
        "plan_steps": [str(step) for step in plan_steps],
        "evaluations": evaluations,
        "chosen": chosen,
        "selected_count": int(len(selected_clips)),
        "selected_clips": [int(cid) for cid in selected_clips],
        "stop_reason": stop_reason,
        "source_paper": "https://aclanthology.org/2026.findings-eacl.150/",
        "implementation_note": "DF-RAG-style reproduction: planner + gMMR lambda sweep + evaluator + upper-median tie break.",
    }
    if trace is not None:
        trace.update(decision)
        trace["selected"] = [int(cid) for cid in selected_clips]
    return selected_clips, decision


def _dynamic_mmr_select_clips(
    candidate_ids,
    clip_scores,
    clip_repr_map,
    *,
    min_clips=2,
    max_clips=5,
    mmr_lambda=0.75,
    stop_threshold=0.05,
    policy="threshold",
    confidence_threshold=0.30,
    ambiguity_gap_threshold=0.25,
    knee_min_drop=0.25,
    knee_alpha=1.0,
    uncertainty_alpha=1.0,
    trace=None,
):
    if not candidate_ids:
        return []

    min_clips = max(1, int(min_clips))
    max_clips = max(min_clips, int(max_clips))
    max_clips = min(max_clips, len(candidate_ids))
    mmr_lambda = float(np.clip(mmr_lambda, 0.0, 1.0))
    raw_scores = [float(clip_scores.get(cid, 0.0)) for cid in candidate_ids]
    s_min = min(raw_scores) if raw_scores else 0.0
    s_max = max(raw_scores) if raw_scores else 0.0
    if s_max > s_min:
        rel_scores = {cid: (float(clip_scores.get(cid, 0.0)) - s_min) / (s_max - s_min) for cid in candidate_ids}
    else:
        rel_scores = {cid: 1.0 for cid in candidate_ids}

    policy = str(policy or "threshold").lower()
    adaptive_policies = {
        "uncertainty",
        "confidence",
        "uncertainty_gap",
        "knee",
        "elbow",
        "saturation_knee",
        "adaptive_knee",
        "self_adaptive_knee",
        "adaptive_uncertainty",
        "self_adaptive_uncertainty",
        "drop_uncertainty",
        "adaptive_gap_uncertainty",
        "self_adaptive_gap",
        "adjacent_gap_uncertainty",
        "soft_adjacent_uncertainty",
        "adaptive_soft_uncertainty",
        "soft_gap_uncertainty",
        "soft_adjacent_uncertainty_mean",
        "adaptive_soft_uncertainty_mean",
        "soft_adjacent_survival",
        "adaptive_soft_survival",
        "survival_soft_adjacent",
        "soft_adjacent_poibinomial",
        "adaptive_soft_poibinomial",
        "poibinomial_adjacent",
        "soft_adjacent_quality_gate",
        "adaptive_soft_quality_gate",
        "compact_soft_adjacent",
        "soft_adjacent_tail_gate",
        "adaptive_soft_tail_gate",
        "soft_adjacent_gain_extend",
        "adaptive_soft_gain_extend",
        "prefix_plateau",
        "adaptive_prefix_plateau",
        "largest_gap_cut",
        "adaptive_largest_gap",
        "two_segment_change",
        "two_segment_changepoint",
        "budgeted_utility",
        "budgeted_marginal_utility",
        "utility_mass",
        "evidence_utility_mass",
        "softmax_mass",
        "evidence_mass",
        "robust_gap_boundary",
        "adaptive_robust_gap",
        "mad_gap_boundary",
        "robust_z_boundary_posterior",
        "adaptive_robust_z_boundary",
        "entropy_hazard_survival",
        "adaptive_entropy_hazard",
        "entropy_boundary_survival",
        "relative_retention_survival",
        "adaptive_relative_retention",
        "retention_lengthnorm",
        "relative_retention_expected",
        "adaptive_relative_retention_expected",
        "retention_expected",
        "relative_retention_ln_expectation",
        "adaptive_relative_retention_ln_expectation",
        "retention_ln_expectation",
        "raw_mmr_delta_root",
        "raw_mmr_delta_root_g1",
        "raw_mmr_delta_root_legacy",
        "raw_delta_root",
        "mmr_delta_root",
        "adjacent_delta_root",
        "bic_boundary",
        "boundary_bic",
        "adaptive_boundary",
        "official_adaptive_k_largest_gap",
        "official_adaptive_k_moving_avg",
        "official_adaptive_k_2diff_spike",
        "rag_adaptive_k_raw",
        "rag_adaptive_k_largest_gap_raw",
        "rag_adaptive_k_moving_avg_raw",
        "rag_adaptive_k_2diff_raw",
    }
    if policy in adaptive_policies:
        official_adaptive_k_policies = {
            "official_adaptive_k_largest_gap": "largest_gap",
            "official_adaptive_k_moving_avg": "moving_avg",
            "official_adaptive_k_2diff_spike": "2diff_spike",
        }
        if policy in official_adaptive_k_policies:
            ranked_candidate_ids = sorted(
                candidate_ids,
                key=lambda cid: rel_scores.get(cid, 0.0),
                reverse=True,
            )
            ranked_scores = [float(rel_scores.get(cid, 0.0)) for cid in ranked_candidate_ids]
            selected_count, decision = _official_adaptive_k_selected_count(
                ranked_scores,
                strategy=official_adaptive_k_policies[policy],
                min_clips=min_clips,
                max_clips=max_clips,
                window=int(os.getenv("DMMR_OFFICIAL_ADAPTIVE_K_WINDOW", "3")),
                ignore_extreme=float(os.getenv("DMMR_OFFICIAL_ADAPTIVE_K_IGNORE_EXTREME", "0.0")),
                ignore_extreme_tail=float(os.getenv("DMMR_OFFICIAL_ADAPTIVE_K_IGNORE_EXTREME_TAIL", "0.0")),
                ignore_below_median=os.getenv("DMMR_OFFICIAL_ADAPTIVE_K_IGNORE_BELOW_MEDIAN", "0") == "1",
                retrieve_more=float(os.getenv("DMMR_OFFICIAL_ADAPTIVE_K_RETRIEVE_MORE", "0.0")),
            )
            selected = ranked_candidate_ids[:selected_count]
            if trace is not None:
                trace["policy"] = policy
                trace["rag_reference"] = "Official Adaptive-k Retrieval, EMNLP 2025"
                trace["rag_reference_repo"] = "https://github.com/megagonlabs/adaptive-k-retrieval"
                for rank, cid in enumerate(ranked_candidate_ids, start=1):
                    event = "select" if rank <= selected_count else "candidate"
                    trace.setdefault("steps", []).append(
                        {
                            "event": event,
                            "clip": int(cid),
                            "rank": int(rank),
                            "mmr_score": float(rel_scores.get(cid, 0.0)),
                            "relevance": float(rel_scores.get(cid, 0.0)),
                            "redundancy": 0.0,
                        }
                    )
                decision.update(
                    {
                        "event": "stop",
                        "policy": policy,
                        "selected_so_far": [int(cid) for cid in selected],
                    }
                )
                trace.setdefault("steps", []).append(decision)
            return selected

        raw_adaptive_k_policies = {
            "rag_adaptive_k_raw": "largest_gap",
            "rag_adaptive_k_largest_gap_raw": "largest_gap",
            "rag_adaptive_k_moving_avg_raw": "moving_avg",
            "rag_adaptive_k_2diff_raw": "2diff_spike",
        }
        if policy in raw_adaptive_k_policies:
            ranked_candidate_ids = sorted(
                candidate_ids,
                key=lambda cid: rel_scores.get(cid, 0.0),
                reverse=True,
            )
            ranked_scores = [float(rel_scores.get(cid, 0.0)) for cid in ranked_candidate_ids]
            selected_count, decision = _adaptive_k_raw_selected_count(
                ranked_scores,
                strategy=raw_adaptive_k_policies[policy],
                min_clips=min_clips,
                max_clips=max_clips,
                window=int(os.getenv("DMMR_RAG_ADAPTIVE_K_WINDOW", "3")),
                ignore_extreme=float(os.getenv("DMMR_RAG_ADAPTIVE_K_IGNORE_EXTREME", "0.0")),
                ignore_extreme_tail=float(os.getenv("DMMR_RAG_ADAPTIVE_K_IGNORE_EXTREME_TAIL", "0.0")),
                retrieve_more=float(os.getenv("DMMR_RAG_ADAPTIVE_K_RETRIEVE_MORE", "0.0")),
            )
            selected = ranked_candidate_ids[:selected_count]
            if trace is not None:
                trace["policy"] = policy
                trace["rag_reference"] = "Adaptive-k Retrieval, EMNLP 2025"
                for rank, cid in enumerate(ranked_candidate_ids, start=1):
                    event = "select" if rank <= selected_count else "candidate"
                    trace.setdefault("steps", []).append(
                        {
                            "event": event,
                            "clip": int(cid),
                            "rank": int(rank),
                            "mmr_score": float(rel_scores.get(cid, 0.0)),
                            "relevance": float(rel_scores.get(cid, 0.0)),
                            "redundancy": 0.0,
                        }
                    )
                decision.update(
                    {
                        "event": "stop",
                        "policy": policy,
                        "selected_so_far": [int(cid) for cid in selected],
                    }
                )
                trace.setdefault("steps", []).append(decision)
            return selected

        full_curve_policies = {
            "relative_retention_survival",
            "adaptive_relative_retention",
            "retention_lengthnorm",
            "relative_retention_expected",
            "adaptive_relative_retention_expected",
            "retention_expected",
            "relative_retention_ln_expectation",
            "adaptive_relative_retention_ln_expectation",
            "retention_ln_expectation",
        }
        k_plus_one_policies = {
            "raw_mmr_delta_root": True,
            "raw_mmr_delta_root_g1": True,
            "raw_mmr_delta_root_legacy": False,
            "raw_delta_root": False,
            "mmr_delta_root": False,
            "adjacent_delta_root": False,
        }
        ranked_steps = []
        selected_for_scoring = []
        remaining = list(candidate_ids)
        if policy in full_curve_policies:
            rank_curve_limit = len(candidate_ids)
        elif policy in k_plus_one_policies:
            rank_curve_limit = min(len(candidate_ids), max_clips + 1)
        else:
            rank_curve_limit = max_clips
        while remaining and len(ranked_steps) < rank_curve_limit:
            best_id = None
            best_objective = -1e9
            best_relevance = 0.0
            best_redundancy = 0.0
            for cid in remaining:
                relevance = rel_scores.get(cid, 0.0)
                redundancy = 0.0
                curr_repr = clip_repr_map.get(cid)
                if curr_repr is not None and selected_for_scoring:
                    sims = [
                        float(np.dot(curr_repr, clip_repr_map[sel]))
                        for sel in selected_for_scoring
                        if clip_repr_map.get(sel) is not None
                    ]
                    redundancy = max(sims) if sims else 0.0
                objective = (
                    relevance
                    if not selected_for_scoring
                    else mmr_lambda * relevance - (1.0 - mmr_lambda) * redundancy
                )
                if objective > best_objective:
                    best_objective = objective
                    best_id = cid
                    best_relevance = relevance
                    best_redundancy = redundancy
            if best_id is None:
                break
            ranked_steps.append(
                {
                    "clip": best_id,
                    "rank": len(ranked_steps) + 1,
                    "mmr_score": float(best_objective),
                    "relevance": float(best_relevance),
                    "redundancy": float(best_redundancy),
                }
            )
            selected_for_scoring.append(best_id)
            remaining.remove(best_id)

        scores = [step["mmr_score"] for step in ranked_steps]
        if policy in k_plus_one_policies:
            selected_count, decision = _raw_mmr_delta_root_selected_count(
                scores,
                min_clips=min_clips,
                max_clips=max_clips,
                normalize_by_g1=k_plus_one_policies[policy],
            )
            selected_count = min(max_clips, len(ranked_steps), max(min_clips, selected_count))
            selected = [step["clip"] for step in ranked_steps[:selected_count]]
            if trace is not None:
                trace["policy"] = policy
                for step in ranked_steps:
                    event = "select" if step["rank"] <= selected_count else "candidate"
                    trace.setdefault("steps", []).append(
                        {
                            "event": event,
                            "clip": int(step["clip"]),
                            "rank": int(step["rank"]),
                            "mmr_score": float(step["mmr_score"]),
                            "relevance": float(step["relevance"]),
                            "redundancy": float(step["redundancy"]),
                        }
                    )
                decision.update(
                    {
                        "event": "stop",
                        "policy": policy,
                        "selected_so_far": [int(cid) for cid in selected],
                        "rank_curve_limit": int(rank_curve_limit),
                    }
                )
                trace.setdefault("steps", []).append(decision)
            return selected

        if policy in {"bic_boundary", "boundary_bic", "adaptive_boundary"}:
            selected_count = min(min_clips, len(ranked_steps))
            smoothed_scores = list(scores)
            for idx in range(1, len(smoothed_scores)):
                smoothed_scores[idx] = min(smoothed_scores[idx - 1], smoothed_scores[idx])

            n_scores = len(smoothed_scores)
            eps = 1e-8
            bic_models = []
            if n_scores <= 1:
                selected_count = n_scores
                stop_reason = "single_candidate"
            else:
                y = np.array(smoothed_scores, dtype=float)
                x = np.arange(1, n_scores + 1, dtype=float)
                slope, intercept = np.polyfit(x, y, 1)
                linear_pred = slope * x + intercept
                linear_sse = float(np.sum((y - linear_pred) ** 2))
                linear_bic = float(n_scores * np.log(linear_sse / n_scores + eps) + 2 * np.log(n_scores))
                best_model = {
                    "type": "smooth_no_boundary",
                    "selected_count": int(n_scores),
                    "bic": linear_bic,
                    "sse": linear_sse,
                    "params": 2,
                    "slope": float(slope),
                    "intercept": float(intercept),
                }
                bic_models.append(best_model)

                for split_k in range(max(1, min_clips), n_scores):
                    left = y[:split_k]
                    right = y[split_k:]
                    left_mean = float(np.mean(left))
                    right_mean = float(np.mean(right))
                    step_sse = float(np.sum((left - left_mean) ** 2) + np.sum((right - right_mean) ** 2))
                    # Two segment means plus the discrete boundary location.
                    step_bic = float(n_scores * np.log(step_sse / n_scores + eps) + 3 * np.log(n_scores))
                    model_info = {
                        "type": "two_segment_boundary",
                        "selected_count": int(split_k),
                        "bic": step_bic,
                        "sse": step_sse,
                        "params": 3,
                        "left_mean": left_mean,
                        "right_mean": right_mean,
                    }
                    bic_models.append(model_info)
                    if step_bic < best_model["bic"]:
                        best_model = model_info

                selected_count = int(best_model["selected_count"])
                stop_reason = best_model["type"]

            selected_count = max(1, min(max_clips, len(ranked_steps), selected_count))
            selected = [step["clip"] for step in ranked_steps[:selected_count]]
            if trace is not None:
                trace["policy"] = policy
                for step in ranked_steps:
                    event = "select" if step["rank"] <= selected_count else "candidate"
                    trace.setdefault("steps", []).append(
                        {
                            "event": event,
                            "clip": int(step["clip"]),
                            "rank": int(step["rank"]),
                            "mmr_score": float(step["mmr_score"]),
                            "relevance": float(step["relevance"]),
                            "redundancy": float(step["redundancy"]),
                        }
                    )
                trace.setdefault("steps", []).append(
                    {
                        "event": "stop",
                        "reason": "bic_boundary_policy",
                        "selected_model": stop_reason,
                        "policy": policy,
                        "selected_count": int(selected_count),
                        "selected_so_far": [int(cid) for cid in selected],
                        "smoothed_scores": [float(score) for score in smoothed_scores],
                        "bic_models": bic_models,
                    }
                )
            return selected

        if policy in {
            "largest_gap_cut",
            "adaptive_largest_gap",
            "two_segment_change",
            "two_segment_changepoint",
            "budgeted_utility",
            "budgeted_marginal_utility",
            "utility_mass",
            "evidence_utility_mass",
            "softmax_mass",
            "evidence_mass",
        }:
            smoothed_scores = list(scores)
            for idx in range(1, len(smoothed_scores)):
                smoothed_scores[idx] = min(smoothed_scores[idx - 1], smoothed_scores[idx])

            score_min = min(smoothed_scores) if smoothed_scores else 0.0
            score_max = max(smoothed_scores) if smoothed_scores else 0.0
            score_range = max(abs(score_max - score_min), 1e-6)
            normalized_scores = [
                (float(score) - score_min) / score_range for score in smoothed_scores
            ]
            gaps = [
                {
                    "after_rank": int(idx + 1),
                    "next_rank": int(idx + 2),
                    "gap": max(
                        0.0,
                        float(normalized_scores[idx])
                        - float(normalized_scores[idx + 1]),
                    ),
                }
                for idx in range(0, max(0, len(normalized_scores) - 1))
            ]
            decision = {
                "event": "stop",
                "policy": policy,
                "smoothed_scores": [float(score) for score in smoothed_scores],
                "normalized_scores": [float(score) for score in normalized_scores],
                "gaps": gaps,
            }

            n_scores = len(normalized_scores)
            if n_scores <= 1:
                selected_count = n_scores
                decision["reason"] = "single_candidate"
            elif policy in {"largest_gap_cut", "adaptive_largest_gap"}:
                best_gap = max(gaps, key=lambda item: item["gap"])
                selected_count = int(best_gap["after_rank"])
                decision.update(
                    {
                        "reason": "largest_adjacent_gap",
                        "best_gap": best_gap,
                    }
                )
            elif policy in {"two_segment_change", "two_segment_changepoint"}:
                y = np.array(normalized_scores, dtype=float)
                candidates = []
                best = None
                for split_k in range(max(1, min_clips), n_scores):
                    left = y[:split_k]
                    right = y[split_k:]
                    left_mean = float(np.mean(left))
                    right_mean = float(np.mean(right))
                    sse = float(
                        np.sum((left - left_mean) ** 2)
                        + np.sum((right - right_mean) ** 2)
                    )
                    info = {
                        "selected_count": int(split_k),
                        "sse": sse,
                        "left_mean": left_mean,
                        "right_mean": right_mean,
                    }
                    candidates.append(info)
                    if best is None or sse < best["sse"]:
                        best = info
                selected_count = int(best["selected_count"]) if best else min_clips
                decision.update(
                    {
                        "reason": "minimum_two_segment_sse",
                        "two_segment_candidates": candidates,
                        "selected_model": best,
                    }
                )
            elif policy in {"budgeted_utility", "budgeted_marginal_utility"}:
                values = np.array(normalized_scores, dtype=float)
                utility_alpha = float(os.getenv("DMMR_BUDGET_UTILITY_ALPHA", "0.0"))
                utility_floor = float(np.mean(values) - utility_alpha * np.std(values))
                selected_count = min(min_clips, n_scores)
                decisions = []
                while selected_count < n_scores and selected_count < max_clips:
                    candidate_value = float(values[selected_count])
                    add = candidate_value >= utility_floor
                    decisions.append(
                        {
                            "rank": int(selected_count + 1),
                            "normalized_mmr": candidate_value,
                            "utility_floor": utility_floor,
                            "add": bool(add),
                        }
                    )
                    if not add:
                        break
                    selected_count += 1
                decision.update(
                    {
                        "reason": "budgeted_marginal_utility",
                        "budget_utility_alpha": utility_alpha,
                        "utility_floor": utility_floor,
                        "utility_decisions": decisions,
                    }
                )
            elif policy in {"utility_mass", "evidence_utility_mass"}:
                rho = float(os.getenv("DMMR_UTILITY_MASS_RHO", "0.80"))
                rho = float(np.clip(rho, 0.0, 1.0))
                values = np.maximum(np.array(normalized_scores, dtype=float), 0.0)
                if float(np.sum(values)) <= 1e-12:
                    weights = np.ones_like(values) / max(len(values), 1)
                else:
                    weights = values / float(np.sum(values))
                cumulative = np.cumsum(weights)
                selected_count = int(np.searchsorted(cumulative, rho, side="left") + 1)
                decision.update(
                    {
                        "reason": "evidence_utility_mass",
                        "utility_mass_rho": rho,
                        "utility_mass_weights": [float(weight) for weight in weights],
                        "utility_mass_cumulative": [
                            float(value) for value in cumulative
                        ],
                    }
                )
            else:
                rho = float(os.getenv("DMMR_SOFTMAX_MASS_RHO", "0.80"))
                rho = float(np.clip(rho, 0.0, 1.0))
                temp_env = os.getenv("DMMR_SOFTMAX_TEMPERATURE")
                if temp_env not in {None, ""}:
                    temperature = float(temp_env)
                else:
                    median_score = float(np.median(normalized_scores))
                    score_mad = float(
                        np.median(
                            np.abs(np.array(normalized_scores, dtype=float) - median_score)
                        )
                    )
                    temperature = max(0.10, 1.4826 * score_mad)
                temperature = max(1e-6, float(temperature))
                logits = np.array(normalized_scores, dtype=float) / temperature
                logits = logits - float(np.max(logits))
                weights = np.exp(logits)
                weights = weights / max(float(np.sum(weights)), 1e-12)
                cumulative = np.cumsum(weights)
                selected_count = int(np.searchsorted(cumulative, rho, side="left") + 1)
                decision.update(
                    {
                        "reason": "softmax_evidence_mass",
                        "softmax_mass_rho": rho,
                        "softmax_temperature": float(temperature),
                        "softmax_weights": [float(weight) for weight in weights],
                        "softmax_cumulative_mass": [
                            float(value) for value in cumulative
                        ],
                    }
                )

            selected_count = min(max_clips, len(ranked_steps), max(min_clips, selected_count))
            selected = [step["clip"] for step in ranked_steps[:selected_count]]
            if trace is not None:
                trace["policy"] = policy
                for step in ranked_steps:
                    event = "select" if step["rank"] <= selected_count else "candidate"
                    trace.setdefault("steps", []).append(
                        {
                            "event": event,
                            "clip": int(step["clip"]),
                            "rank": int(step["rank"]),
                            "mmr_score": float(step["mmr_score"]),
                            "relevance": float(step["relevance"]),
                            "redundancy": float(step["redundancy"]),
                        }
                    )
                decision["selected_count"] = int(selected_count)
                decision["selected_so_far"] = [int(cid) for cid in selected]
                trace.setdefault("steps", []).append(decision)
            return selected

        if policy in {"adaptive_uncertainty", "self_adaptive_uncertainty", "drop_uncertainty"}:
            selected_count = min(min_clips, len(ranked_steps))
            smoothed_scores = list(scores)
            for idx in range(1, len(smoothed_scores)):
                smoothed_scores[idx] = min(smoothed_scores[idx - 1], smoothed_scores[idx])

            drops = []
            for idx in range(0, len(smoothed_scores) - 1):
                current = float(smoothed_scores[idx])
                next_score = float(smoothed_scores[idx + 1])
                drop = max(0.0, current - next_score) / max(abs(current), 1e-6)
                drops.append(
                    {
                        "after_rank": int(idx + 1),
                        "next_rank": int(idx + 2),
                        "drop": float(drop),
                    }
                )

            drop12 = drops[0]["drop"] if drops else 0.0
            tail_drop_values = [item["drop"] for item in drops[1:]]
            tail_mean = float(np.mean(tail_drop_values)) if tail_drop_values else 0.0
            tail_std = float(np.std(tail_drop_values)) if tail_drop_values else 0.0
            low_confidence_threshold = tail_mean + float(uncertainty_alpha) * tail_std
            ambiguity_threshold = max(0.0, tail_mean - float(uncertainty_alpha) * tail_std)
            decision = {
                "event": "stop",
                "reason": "adaptive_uncertainty_policy",
                "policy": policy,
                "uncertainty_alpha": float(uncertainty_alpha),
                "tail_drop_mean": float(tail_mean),
                "tail_drop_std": float(tail_std),
                "low_confidence_threshold": float(low_confidence_threshold),
                "ambiguity_threshold": float(ambiguity_threshold),
                "initial_k": int(selected_count),
                "bonuses": [],
                "drops": drops,
                "smoothed_scores": [float(score) for score in smoothed_scores],
            }
            if tail_drop_values and drop12 > low_confidence_threshold:
                selected_count += 1
                decision["bonuses"].append(
                    {
                        "type": "isolated_top1",
                        "drop_12": float(drop12),
                        "threshold": float(low_confidence_threshold),
                    }
                )
            for idx in (1, 2):
                if idx < len(drops) and drops[idx]["drop"] < ambiguity_threshold:
                    selected_count += 1
                    decision["bonuses"].append(
                        {
                            "type": "ambiguous_tail",
                            "from_rank": int(drops[idx]["after_rank"]),
                            "to_rank": int(drops[idx]["next_rank"]),
                            "drop": float(drops[idx]["drop"]),
                            "threshold": float(ambiguity_threshold),
                        }
                    )

            selected_count = max(1, min(max_clips, len(ranked_steps), selected_count))
            selected = [step["clip"] for step in ranked_steps[:selected_count]]
            if trace is not None:
                trace["policy"] = policy
                trace["uncertainty_alpha"] = float(uncertainty_alpha)
                for step in ranked_steps:
                    event = "select" if step["rank"] <= selected_count else "candidate"
                    trace.setdefault("steps", []).append(
                        {
                            "event": event,
                            "clip": int(step["clip"]),
                            "rank": int(step["rank"]),
                            "mmr_score": float(step["mmr_score"]),
                            "relevance": float(step["relevance"]),
                            "redundancy": float(step["redundancy"]),
                        }
                    )
                decision["selected_count"] = int(selected_count)
                decision["selected_so_far"] = [int(cid) for cid in selected]
                trace.setdefault("steps", []).append(decision)
            return selected

        if policy in {
            "entropy_hazard_survival",
            "adaptive_entropy_hazard",
            "entropy_boundary_survival",
        }:
            smoothed_scores = list(scores)
            for idx in range(1, len(smoothed_scores)):
                smoothed_scores[idx] = min(smoothed_scores[idx - 1], smoothed_scores[idx])

            score_min = min(smoothed_scores) if smoothed_scores else 0.0
            score_max = max(smoothed_scores) if smoothed_scores else 0.0
            score_range = max(abs(score_max - score_min), 1e-6)
            normalized_scores = [
                (float(score) - score_min) / score_range for score in smoothed_scores
            ]

            gaps = []
            for idx in range(0, max(0, len(normalized_scores) - 1)):
                gap = max(
                    0.0,
                    float(normalized_scores[idx])
                    - float(normalized_scores[idx + 1]),
                )
                gaps.append(
                    {
                        "after_rank": int(idx + 1),
                        "next_rank": int(idx + 2),
                        "gap": float(gap),
                    }
                )

            n_scores = len(normalized_scores)
            quantile = float(os.getenv("DMMR_ENTROPY_HAZARD_QUANTILE", "0.60"))
            quantile = float(np.clip(quantile, 0.0, 1.0))
            eps = 1e-12
            continue_probs = []
            stop_hazards = []
            hazard_steps = []
            k_distribution = []
            expected_k = float(n_scores)

            if n_scores <= 1:
                selected_count = n_scores
                stop_reason = "single_candidate"
                posterior_cdf = [1.0] if n_scores == 1 else []
                k_distribution = [1.0] if n_scores == 1 else []
                expected_k = float(n_scores)
            else:
                gap_values = [float(item["gap"]) for item in gaps]
                for idx in range(0, len(gap_values)):
                    suffix_gaps = gap_values[idx:]
                    suffix_total = float(np.sum(suffix_gaps))
                    suffix_count = len(suffix_gaps)
                    if suffix_count <= 1:
                        entropy = 1.0
                        normalized_gap_mass = [1.0]
                    elif suffix_total <= eps:
                        entropy = 1.0
                        normalized_gap_mass = [1.0 / suffix_count] * suffix_count
                    else:
                        normalized_gap_mass = [
                            max(0.0, float(gap)) / suffix_total
                            for gap in suffix_gaps
                        ]
                        entropy = -sum(
                            mass * math.log(max(mass, eps))
                            for mass in normalized_gap_mass
                        ) / max(math.log(suffix_count), eps)
                        entropy = float(np.clip(entropy, 0.0, 1.0))

                    temperature = max(
                        eps,
                        suffix_total * math.sqrt(max(entropy, eps)),
                    )
                    suffix_median = float(np.median(suffix_gaps))
                    boundary_logits = [
                        (float(gap) - suffix_median) / temperature
                        for gap in suffix_gaps
                    ]
                    # The final zero logit is a virtual "no clear boundary yet" option.
                    logits = boundary_logits + [0.0]
                    logit_max = max(logits) if logits else 0.0
                    exp_logits = [math.exp(value - logit_max) for value in logits]
                    normalizer = max(sum(exp_logits), eps)
                    posterior = [value / normalizer for value in exp_logits]

                    stop_hazard = float(posterior[0])
                    continue_prob = float(1.0 - stop_hazard)
                    stop_hazards.append(stop_hazard)
                    continue_probs.append(continue_prob)
                    hazard_steps.append(
                        {
                            "after_rank": int(idx + 1),
                            "next_rank": int(idx + 2),
                            "suffix_gaps": [float(gap) for gap in suffix_gaps],
                            "suffix_total": suffix_total,
                            "suffix_gap_mass": [
                                float(mass) for mass in normalized_gap_mass
                            ],
                            "entropy": float(entropy),
                            "temperature": float(temperature),
                            "suffix_median": float(suffix_median),
                            "boundary_logits": [
                                float(value) for value in boundary_logits
                            ],
                            "posterior_current_stop": stop_hazard,
                            "posterior_no_clear_boundary": float(posterior[-1]),
                            "continue_prob": continue_prob,
                        }
                    )

                survival = 1.0
                for idx, continue_prob in enumerate(continue_probs):
                    stop_prob = survival * (1.0 - float(continue_prob))
                    k_distribution.append(float(stop_prob))
                    survival *= float(continue_prob)
                k_distribution.append(float(survival))

                posterior_total = max(float(sum(k_distribution)), eps)
                k_distribution = [
                    float(prob / posterior_total) for prob in k_distribution
                ]
                expected_k = float(
                    sum((idx + 1) * prob for idx, prob in enumerate(k_distribution))
                )
                posterior_cdf = []
                cumulative = 0.0
                selected_count = n_scores
                for idx, prob in enumerate(k_distribution):
                    cumulative += float(prob)
                    posterior_cdf.append(float(cumulative))
                    if cumulative >= quantile:
                        selected_count = idx + 1
                        break
                stop_reason = "entropy_hazard_posterior_quantile"

            selected_count = min(max_clips, len(ranked_steps), max(min_clips, selected_count))
            selected = [step["clip"] for step in ranked_steps[:selected_count]]
            if trace is not None:
                trace["policy"] = policy
                for step in ranked_steps:
                    event = "select" if step["rank"] <= selected_count else "candidate"
                    trace.setdefault("steps", []).append(
                        {
                            "event": event,
                            "clip": int(step["clip"]),
                            "rank": int(step["rank"]),
                            "mmr_score": float(step["mmr_score"]),
                            "relevance": float(step["relevance"]),
                            "redundancy": float(step["redundancy"]),
                        }
                    )
                trace.setdefault("steps", []).append(
                    {
                        "event": "stop",
                        "reason": stop_reason,
                        "policy": policy,
                        "posterior_quantile": float(quantile),
                        "expected_k": float(expected_k),
                        "k_distribution": [
                            {
                                "k": int(idx + 1),
                                "prob": float(prob),
                            }
                            for idx, prob in enumerate(k_distribution)
                        ],
                        "posterior_cdf": [float(value) for value in posterior_cdf],
                        "gaps": gaps,
                        "stop_hazards": [float(prob) for prob in stop_hazards],
                        "continue_probs": [float(prob) for prob in continue_probs],
                        "hazard_steps": hazard_steps,
                        "smoothed_scores": [float(score) for score in smoothed_scores],
                        "normalized_scores": [float(score) for score in normalized_scores],
                        "selected_count": int(selected_count),
                        "selected_so_far": [int(cid) for cid in selected],
                    }
                )
            return selected

        if policy in {
            "relative_retention_survival",
            "adaptive_relative_retention",
            "retention_lengthnorm",
            "relative_retention_expected",
            "adaptive_relative_retention_expected",
            "retention_expected",
            "relative_retention_ln_expectation",
            "adaptive_relative_retention_ln_expectation",
            "retention_ln_expectation",
        }:
            smoothed_scores = list(scores)
            for idx in range(1, len(smoothed_scores)):
                smoothed_scores[idx] = min(smoothed_scores[idx - 1], smoothed_scores[idx])

            score_min = min(smoothed_scores) if smoothed_scores else 0.0
            score_max = max(smoothed_scores) if smoothed_scores else 0.0
            score_range = max(abs(score_max - score_min), 1e-6)
            normalized_scores = [
                (float(score) - score_min) / score_range for score in smoothed_scores
            ]
            n_scores = len(normalized_scores)
            eps = 1e-12
            relative_drops = []
            continue_probs = []
            k_distribution = []
            length_normalized_scores = []
            expected_k = 0.0
            expected_decision = None
            length_normalized_distribution = []

            if n_scores <= 1:
                selected_count = n_scores
                stop_reason = "single_candidate"
            else:
                for idx in range(0, n_scores - 1):
                    current = float(normalized_scores[idx])
                    next_score = float(normalized_scores[idx + 1])
                    drop = max(0.0, current - next_score)
                    relative_drop = drop / max(abs(current), eps)
                    continue_prob = float(np.clip(next_score / max(abs(current), eps), 0.0, 1.0))
                    relative_drops.append(
                        {
                            "after_rank": int(idx + 1),
                            "next_rank": int(idx + 2),
                            "gap": float(drop),
                            "relative_drop": float(relative_drop),
                        }
                    )
                    continue_probs.append(continue_prob)

                if policy in {
                    "relative_retention_expected",
                    "adaptive_relative_retention_expected",
                    "retention_expected",
                }:
                    top_gain = max(abs(float(normalized_scores[0])), eps)
                    expected_k = float(
                        sum(
                            max(0.0, float(score)) / top_gain
                            for score in normalized_scores
                        )
                    )
                    expected_decision = os.getenv(
                        "DMMR_RELRET_EXPECTED_DECISION", "round"
                    ).lower()
                    if expected_decision == "ceil":
                        selected_count = int(math.ceil(expected_k))
                    elif expected_decision == "floor":
                        selected_count = int(math.floor(expected_k))
                    else:
                        expected_decision = "round"
                        selected_count = int(math.floor(expected_k + 0.5))
                    stop_reason = "relative_retention_expected_k"
                else:
                    survival_log = 0.0
                    for rank in range(1, n_scores + 1):
                        if rank < n_scores:
                            stop_prob = 1.0 - float(continue_probs[rank - 1])
                            raw_log_prob = survival_log + math.log(max(eps, stop_prob))
                        else:
                            raw_log_prob = survival_log
                        k_distribution.append(float(math.exp(raw_log_prob)))
                        normalized_log_score = raw_log_prob / float(rank)
                        length_normalized_scores.append(float(math.exp(normalized_log_score)))
                        if rank < n_scores:
                            survival_log += math.log(max(eps, float(continue_probs[rank - 1])))

                    if policy in {
                        "relative_retention_ln_expectation",
                        "adaptive_relative_retention_ln_expectation",
                        "retention_ln_expectation",
                    }:
                        normalizer = max(float(sum(length_normalized_scores)), eps)
                        length_normalized_distribution = [
                            float(score / normalizer)
                            for score in length_normalized_scores
                        ]
                        expected_k = float(
                            sum(
                                (idx + 1) * prob
                                for idx, prob in enumerate(length_normalized_distribution)
                            )
                        )
                        expected_decision = os.getenv(
                            "DMMR_RELRET_LN_EXPECT_DECISION", "ceil"
                        ).lower()
                        if expected_decision == "round":
                            selected_count = int(math.floor(expected_k + 0.5))
                        elif expected_decision == "floor":
                            selected_count = int(math.floor(expected_k))
                        else:
                            expected_decision = "ceil"
                            selected_count = int(math.ceil(expected_k))
                        stop_reason = "relative_retention_length_normalized_expectation"
                    else:
                        selected_count = int(np.argmax(np.array(length_normalized_scores, dtype=float)) + 1)
                        stop_reason = "relative_retention_length_normalized_map"

            selected_count = min(max_clips, len(ranked_steps), max(min_clips, selected_count))
            selected = [step["clip"] for step in ranked_steps[:selected_count]]
            if trace is not None:
                trace["policy"] = policy
                for step in ranked_steps:
                    event = "select" if step["rank"] <= selected_count else "candidate"
                    trace.setdefault("steps", []).append(
                        {
                            "event": event,
                            "clip": int(step["clip"]),
                            "rank": int(step["rank"]),
                            "mmr_score": float(step["mmr_score"]),
                            "relevance": float(step["relevance"]),
                            "redundancy": float(step["redundancy"]),
                        }
                    )
                trace.setdefault("steps", []).append(
                    {
                        "event": "stop",
                        "reason": stop_reason,
                        "policy": policy,
                        "relative_drops": relative_drops,
                        "continue_probs": [float(prob) for prob in continue_probs],
                        "k_distribution": [
                            {"k": int(idx + 1), "prob": float(prob)}
                            for idx, prob in enumerate(k_distribution)
                        ],
                        "expected_k": float(expected_k),
                        "expected_decision": expected_decision,
                        "length_normalized_distribution": [
                            {"k": int(idx + 1), "prob": float(prob)}
                            for idx, prob in enumerate(length_normalized_distribution)
                        ],
                        "length_normalized_scores": [
                            {"k": int(idx + 1), "score": float(score)}
                            for idx, score in enumerate(length_normalized_scores)
                        ],
                        "smoothed_scores": [float(score) for score in smoothed_scores],
                        "normalized_scores": [float(score) for score in normalized_scores],
                        "selected_count": int(selected_count),
                        "selected_so_far": [int(cid) for cid in selected],
                    }
                )
            return selected

        if policy in {
            "soft_adjacent_uncertainty",
            "adaptive_soft_uncertainty",
            "soft_gap_uncertainty",
            "soft_adjacent_uncertainty_mean",
            "adaptive_soft_uncertainty_mean",
            "soft_adjacent_survival",
            "adaptive_soft_survival",
            "survival_soft_adjacent",
            "soft_adjacent_poibinomial",
            "adaptive_soft_poibinomial",
            "poibinomial_adjacent",
            "soft_adjacent_quality_gate",
            "adaptive_soft_quality_gate",
            "compact_soft_adjacent",
            "soft_adjacent_tail_gate",
            "adaptive_soft_tail_gate",
            "soft_adjacent_gain_extend",
            "adaptive_soft_gain_extend",
        }:
            smoothed_scores = list(scores)
            for idx in range(1, len(smoothed_scores)):
                smoothed_scores[idx] = min(smoothed_scores[idx - 1], smoothed_scores[idx])

            score_min = min(smoothed_scores) if smoothed_scores else 0.0
            score_max = max(smoothed_scores) if smoothed_scores else 0.0
            score_range = max(abs(score_max - score_min), 1e-6)
            normalized_scores = [
                (float(score) - score_min) / score_range for score in smoothed_scores
            ]

            gaps = []
            for idx in range(0, len(normalized_scores) - 1):
                gap = max(0.0, float(normalized_scores[idx]) - float(normalized_scores[idx + 1]))
                gaps.append(
                    {
                        "after_rank": int(idx + 1),
                        "next_rank": int(idx + 2),
                        "gap": float(gap),
                    }
                )

            gap_values = [item["gap"] for item in gaps]
            if gap_values:
                gap_array = np.array(gap_values, dtype=float)
                if policy in {"soft_adjacent_uncertainty_mean", "adaptive_soft_uncertainty_mean"}:
                    location = float(np.mean(gap_array))
                    spread = float(np.std(gap_array))
                    location_type = "mean"
                    spread_type = "std"
                else:
                    soft_gap_quantile = float(os.getenv("DMMR_SOFT_GAP_QUANTILE", "0.5"))
                    soft_gap_quantile = float(np.clip(soft_gap_quantile, 0.0, 1.0))
                    location = float(np.quantile(gap_array, soft_gap_quantile))
                    mad = float(np.median(np.abs(gap_array - location)))
                    spread = 1.4826 * mad
                    if spread <= 1e-8:
                        spread = float(np.std(gap_array))
                    location_type = f"quantile_{soft_gap_quantile:.2f}"
                    spread_type = "mad"
                temperature = max(0.05, float(uncertainty_alpha) * max(spread, 1e-8))
                continue_probs = [
                    float(1.0 / (1.0 + np.exp((gap - location) / temperature)))
                    for gap in gap_values
                ]
                survival_terms = []
                poibinomial_distribution = []
                poibinomial_decision = None
                poibinomial_quantile = None
                if policy in {
                    "soft_adjacent_survival",
                    "adaptive_soft_survival",
                    "survival_soft_adjacent",
                }:
                    survival_gamma = float(os.getenv("DMMR_SURVIVAL_GAMMA", "1.0"))
                    survival_gamma = float(np.clip(survival_gamma, 0.01, 1.0))
                    survival_prob = 1.0
                    soft_k = 1.0
                    for prob in continue_probs:
                        survival_prob *= float(prob)
                        scaled_survival = float(survival_prob ** survival_gamma)
                        survival_terms.append(scaled_survival)
                        soft_k += scaled_survival
                elif policy in {
                    "soft_adjacent_poibinomial",
                    "adaptive_soft_poibinomial",
                    "poibinomial_adjacent",
                }:
                    survival_gamma = None
                    poibinomial_distribution = [1.0]
                    for prob in continue_probs:
                        next_distribution = [0.0] * (len(poibinomial_distribution) + 1)
                        for count, mass in enumerate(poibinomial_distribution):
                            next_distribution[count] += float(mass) * (1.0 - float(prob))
                            next_distribution[count + 1] += float(mass) * float(prob)
                        poibinomial_distribution = next_distribution
                    soft_k = 1.0 + float(
                        sum(
                            count * mass
                            for count, mass in enumerate(poibinomial_distribution)
                        )
                    )
                    poibinomial_decision = os.getenv(
                        "DMMR_POIBINOMIAL_DECISION", "mean"
                    ).lower()
                    poibinomial_quantile = float(
                        os.getenv("DMMR_POIBINOMIAL_QUANTILE", "0.60")
                    )
                    poibinomial_quantile = float(
                        np.clip(poibinomial_quantile, 0.0, 1.0)
                    )
                    if poibinomial_decision == "map":
                        selected_count = 1 + int(
                            np.argmax(np.array(poibinomial_distribution, dtype=float))
                        )
                    elif poibinomial_decision == "quantile":
                        cumulative = 0.0
                        selected_count = len(poibinomial_distribution)
                        for count, mass in enumerate(poibinomial_distribution):
                            cumulative += float(mass)
                            if cumulative >= poibinomial_quantile:
                                selected_count = 1 + count
                                break
                    elif poibinomial_decision == "ceil_mean":
                        selected_count = int(np.ceil(soft_k))
                    else:
                        poibinomial_decision = "mean"
                        selected_count = int(np.floor(soft_k + 0.5))
                else:
                    survival_gamma = None
                    poibinomial_distribution = []
                    poibinomial_decision = None
                    poibinomial_quantile = None
                    soft_k = 1.0 + float(np.sum(continue_probs))
                if policy == "compact_soft_adjacent":
                    selected_count = int(np.floor(soft_k))
                elif policy not in {
                    "soft_adjacent_poibinomial",
                    "adaptive_soft_poibinomial",
                    "poibinomial_adjacent",
                }:
                    selected_count = int(np.floor(soft_k + 0.5))
            else:
                location = 0.0
                spread = 0.0
                temperature = 0.0
                continue_probs = []
                survival_terms = []
                survival_gamma = None
                poibinomial_distribution = []
                poibinomial_decision = None
                poibinomial_quantile = None
                soft_k = 1.0 if ranked_steps else 0.0
                selected_count = 1 if ranked_steps else 0
                location_type = "none"
                spread_type = "none"

            quality_gate_decisions = []
            if policy in {
                "soft_adjacent_quality_gate",
                "adaptive_soft_quality_gate",
                "compact_soft_adjacent",
            } and selected_count > min_clips:
                min_quality_rank = int(os.getenv("DMMR_QUALITY_GATE_MIN_RANK", "3"))
                gate_alpha = float(os.getenv("DMMR_QUALITY_GATE_ALPHA", "1.0"))
                ranked_mmr = np.array([float(step["mmr_score"]) for step in ranked_steps], dtype=float)
                ranked_rel = np.array([float(step["relevance"]) for step in ranked_steps], dtype=float)
                ranked_red = np.array([float(step["redundancy"]) for step in ranked_steps], dtype=float)
                mmr_median = float(np.median(ranked_mmr)) if len(ranked_mmr) else 0.0
                mmr_mad = float(np.median(np.abs(ranked_mmr - mmr_median))) if len(ranked_mmr) else 0.0
                mmr_spread = 1.4826 * mmr_mad
                if mmr_spread <= 1e-8:
                    mmr_spread = float(np.std(ranked_mmr)) if len(ranked_mmr) else 0.0
                mmr_floor = mmr_median - gate_alpha * mmr_spread
                rel_median = float(np.median(ranked_rel)) if len(ranked_rel) else 0.0
                red_median = float(np.median(ranked_red)) if len(ranked_red) else 0.0
                min_mmr_env = os.getenv("DMMR_QUALITY_GATE_MIN_MMR")
                min_mmr_score = float(min_mmr_env) if min_mmr_env not in {None, ""} else None

                while selected_count > min_clips and selected_count >= min_quality_rank:
                    tail = ranked_steps[selected_count - 1]
                    low_mmr_tail = float(tail["mmr_score"]) < mmr_floor
                    negative_mmr_tail = (
                        min_mmr_score is not None
                        and float(tail["mmr_score"]) < min_mmr_score
                    )
                    low_quality_tail = (
                        float(tail["relevance"]) < rel_median
                        and float(tail["redundancy"]) > red_median
                    )
                    decision = {
                        "rank": int(selected_count),
                        "clip": int(tail["clip"]),
                        "mmr_score": float(tail["mmr_score"]),
                        "relevance": float(tail["relevance"]),
                        "redundancy": float(tail["redundancy"]),
                        "mmr_floor": float(mmr_floor),
                        "rel_median": float(rel_median),
                        "red_median": float(red_median),
                        "min_mmr_score": None if min_mmr_score is None else float(min_mmr_score),
                        "drop": bool(low_mmr_tail or low_quality_tail or negative_mmr_tail),
                        "reasons": [],
                    }
                    if low_mmr_tail:
                        decision["reasons"].append("below_adaptive_mmr_floor")
                    if negative_mmr_tail:
                        decision["reasons"].append("below_min_mmr_score")
                    if low_quality_tail:
                        decision["reasons"].append("low_relevance_high_redundancy")
                    quality_gate_decisions.append(decision)
                    if not decision["drop"]:
                        break
                    selected_count -= 1

            tail_gate_decisions = []
            if policy in {"soft_adjacent_tail_gate", "adaptive_soft_tail_gate"}:
                tail_gate_alpha = float(os.getenv("DMMR_TAIL_GATE_ALPHA", "0.5"))
                tail_gate_min_rank = int(os.getenv("DMMR_TAIL_GATE_MIN_RANK", "3"))
                tail_gate_min_keep = int(os.getenv("DMMR_TAIL_GATE_MIN_KEEP", "2"))
                tail_gate_min_keep = max(min_clips, tail_gate_min_keep)

                if gap_values:
                    gap_array = np.array(gap_values, dtype=float)
                    gap_median = float(np.median(gap_array))
                    gap_mad = float(np.median(np.abs(gap_array - gap_median)))
                    gap_spread = 1.4826 * gap_mad
                    if gap_spread <= 1e-8:
                        gap_spread = float(np.std(gap_array))
                    gap_boundary = gap_median + tail_gate_alpha * gap_spread
                else:
                    gap_median = 0.0
                    gap_spread = 0.0
                    gap_boundary = float("inf")

                while selected_count > tail_gate_min_keep and selected_count >= tail_gate_min_rank:
                    tail = ranked_steps[selected_count - 1]
                    previous_steps = ranked_steps[: selected_count - 1]
                    previous_relevance = [float(step["relevance"]) for step in previous_steps]
                    selected_mmr = [
                        float(step["mmr_score"])
                        for step in ranked_steps[:selected_count]
                    ]
                    previous_rel_median = (
                        float(np.median(previous_relevance))
                        if previous_relevance
                        else 0.0
                    )
                    selected_mmr_median = (
                        float(np.median(selected_mmr)) if selected_mmr else 0.0
                    )
                    previous_gap = (
                        float(gap_values[selected_count - 2])
                        if selected_count - 2 < len(gap_values)
                        else 0.0
                    )
                    large_saturation_gap = previous_gap > gap_boundary
                    weak_tail_relevance = float(tail["relevance"]) <= previous_rel_median
                    weak_tail_mmr = float(tail["mmr_score"]) <= selected_mmr_median
                    drop_tail = bool(large_saturation_gap and weak_tail_relevance and weak_tail_mmr)
                    decision = {
                        "rank": int(selected_count),
                        "clip": int(tail["clip"]),
                        "mmr_score": float(tail["mmr_score"]),
                        "relevance": float(tail["relevance"]),
                        "redundancy": float(tail["redundancy"]),
                        "previous_gap": float(previous_gap),
                        "gap_boundary": float(gap_boundary),
                        "gap_median": float(gap_median),
                        "gap_spread": float(gap_spread),
                        "tail_gate_alpha": float(tail_gate_alpha),
                        "previous_rel_median": float(previous_rel_median),
                        "selected_mmr_median": float(selected_mmr_median),
                        "drop": drop_tail,
                        "reasons": [],
                    }
                    if large_saturation_gap:
                        decision["reasons"].append("large_saturation_gap_before_tail")
                    if weak_tail_relevance:
                        decision["reasons"].append("tail_relevance_not_above_previous_median")
                    if weak_tail_mmr:
                        decision["reasons"].append("tail_mmr_not_above_selected_median")
                    tail_gate_decisions.append(decision)
                    if not drop_tail:
                        break
                    selected_count -= 1

            gain_extend_decisions = []
            if policy in {"soft_adjacent_gain_extend", "adaptive_soft_gain_extend"}:
                high_gain_env = os.getenv("DMMR_HIGH_GAIN_THRESHOLD")
                high_gain_threshold = (
                    float(high_gain_env)
                    if high_gain_env not in {None, ""}
                    else float(stop_threshold)
                )
                max_extra = int(os.getenv("DMMR_HIGH_GAIN_MAX_EXTRA", "2"))
                extra_added = 0
                while selected_count < min(max_clips, len(ranked_steps)) and extra_added < max_extra:
                    candidate = ranked_steps[selected_count]
                    candidate_mmr = float(candidate["mmr_score"])
                    add_candidate = candidate_mmr >= high_gain_threshold
                    decision = {
                        "rank": int(selected_count + 1),
                        "clip": int(candidate["clip"]),
                        "mmr_score": candidate_mmr,
                        "relevance": float(candidate["relevance"]),
                        "redundancy": float(candidate["redundancy"]),
                        "high_gain_threshold": float(high_gain_threshold),
                        "add": bool(add_candidate),
                    }
                    gain_extend_decisions.append(decision)
                    if not add_candidate:
                        break
                    selected_count += 1
                    extra_added += 1

            selected_count = min(max_clips, len(ranked_steps), max(min_clips, selected_count))
            selected = [step["clip"] for step in ranked_steps[:selected_count]]
            if trace is not None:
                trace["policy"] = policy
                trace["uncertainty_alpha"] = float(uncertainty_alpha)
                for step in ranked_steps:
                    event = "select" if step["rank"] <= selected_count else "candidate"
                    trace.setdefault("steps", []).append(
                        {
                            "event": event,
                            "clip": int(step["clip"]),
                            "rank": int(step["rank"]),
                            "mmr_score": float(step["mmr_score"]),
                            "relevance": float(step["relevance"]),
                            "redundancy": float(step["redundancy"]),
                        }
                    )
                trace.setdefault("steps", []).append(
                    {
                        "event": "stop",
                        "reason": "soft_adjacent_uncertainty_policy",
                        "policy": policy,
                        "uncertainty_alpha": float(uncertainty_alpha),
                        "location_type": location_type,
                        "spread_type": spread_type,
                        "gap_location": float(location),
                        "gap_spread": float(spread),
                        "temperature": float(temperature),
                        "soft_k": float(soft_k),
                        "gaps": gaps,
                        "continue_probs": [float(prob) for prob in continue_probs],
                        "survival_terms": [float(term) for term in survival_terms],
                        "survival_gamma": None if survival_gamma is None else float(survival_gamma),
                        "poibinomial_decision": poibinomial_decision,
                        "poibinomial_quantile": (
                            None
                            if poibinomial_quantile is None
                            else float(poibinomial_quantile)
                        ),
                        "poibinomial_distribution": [
                            {
                                "k": int(count + 1),
                                "prob": float(mass),
                            }
                            for count, mass in enumerate(poibinomial_distribution)
                        ],
                        "smoothed_scores": [float(score) for score in smoothed_scores],
                        "normalized_scores": [float(score) for score in normalized_scores],
                        "quality_gate_decisions": quality_gate_decisions,
                        "tail_gate_decisions": tail_gate_decisions,
                        "gain_extend_decisions": gain_extend_decisions,
                        "selected_count": int(selected_count),
                        "selected_so_far": [int(cid) for cid in selected],
                    }
                )
            return selected

        if policy in {
            "prefix_plateau",
            "adaptive_prefix_plateau",
            "robust_gap_boundary",
            "adaptive_robust_gap",
            "mad_gap_boundary",
        }:
            selected_count = min(min_clips, len(ranked_steps))
            smoothed_scores = list(scores)
            for idx in range(1, len(smoothed_scores)):
                smoothed_scores[idx] = min(smoothed_scores[idx - 1], smoothed_scores[idx])

            score_min = min(smoothed_scores) if smoothed_scores else 0.0
            score_max = max(smoothed_scores) if smoothed_scores else 0.0
            score_range = max(abs(score_max - score_min), 1e-6)
            normalized_scores = [
                (float(score) - score_min) / score_range for score in smoothed_scores
            ]

            gaps = []
            for idx in range(0, len(normalized_scores) - 1):
                gap = max(0.0, float(normalized_scores[idx]) - float(normalized_scores[idx + 1]))
                gaps.append(
                    {
                        "after_rank": int(idx + 1),
                        "next_rank": int(idx + 2),
                        "gap": float(gap),
                    }
                )

            gap_values = [item["gap"] for item in gaps]
            if gap_values:
                gap_array = np.array(gap_values, dtype=float)
                gap_median = float(np.median(gap_array))
                gap_mad = float(np.median(np.abs(gap_array - gap_median)))
                gap_spread = 1.4826 * gap_mad
                if gap_spread <= 1e-8:
                    gap_spread = float(np.std(gap_array))
                if policy in {"prefix_plateau", "adaptive_prefix_plateau"}:
                    plateau_alpha = float(os.getenv("DMMR_PREFIX_PLATEAU_ALPHA", "0.0"))
                    boundary = gap_median + plateau_alpha * gap_spread
                else:
                    plateau_alpha = float(uncertainty_alpha)
                    boundary = gap_median + plateau_alpha * gap_spread
                selected_count = 1
                stop_reason = "max_clips_or_no_boundary"
                for idx, gap in enumerate(gap_values):
                    if selected_count >= max_clips:
                        break
                    # A large adjacent gap is evidence saturation: later clips live below the boundary.
                    if selected_count >= min_clips and gap > boundary:
                        stop_reason = "robust_gap_boundary"
                        break
                    selected_count = idx + 2
            else:
                gap_median = 0.0
                gap_spread = 0.0
                boundary = 0.0
                plateau_alpha = (
                    float(os.getenv("DMMR_PREFIX_PLATEAU_ALPHA", "0.0"))
                    if policy in {"prefix_plateau", "adaptive_prefix_plateau"}
                    else float(uncertainty_alpha)
                )
                stop_reason = "single_candidate"
                selected_count = 1 if ranked_steps else 0

            selected_count = max(min_clips, min(max_clips, len(ranked_steps), selected_count))
            selected = [step["clip"] for step in ranked_steps[:selected_count]]
            if trace is not None:
                trace["policy"] = policy
                trace["uncertainty_alpha"] = float(uncertainty_alpha)
                trace["prefix_plateau_alpha"] = float(plateau_alpha)
                for step in ranked_steps:
                    event = "select" if step["rank"] <= selected_count else "candidate"
                    trace.setdefault("steps", []).append(
                        {
                            "event": event,
                            "clip": int(step["clip"]),
                            "rank": int(step["rank"]),
                            "mmr_score": float(step["mmr_score"]),
                            "relevance": float(step["relevance"]),
                            "redundancy": float(step["redundancy"]),
                        }
                    )
                trace.setdefault("steps", []).append(
                    {
                        "event": "stop",
                        "reason": stop_reason,
                        "policy": policy,
                        "uncertainty_alpha": float(uncertainty_alpha),
                        "prefix_plateau_alpha": float(plateau_alpha),
                        "gap_median": float(gap_median),
                        "gap_spread": float(gap_spread),
                        "gap_boundary": float(boundary),
                        "gaps": gaps,
                        "smoothed_scores": [float(score) for score in smoothed_scores],
                        "normalized_scores": [float(score) for score in normalized_scores],
                        "selected_count": int(selected_count),
                        "selected_so_far": [int(cid) for cid in selected],
                    }
                )
            return selected

        if policy in {"robust_z_boundary_posterior", "adaptive_robust_z_boundary"}:
            smoothed_scores = list(scores)
            for idx in range(1, len(smoothed_scores)):
                smoothed_scores[idx] = min(smoothed_scores[idx - 1], smoothed_scores[idx])

            score_min = min(smoothed_scores) if smoothed_scores else 0.0
            score_max = max(smoothed_scores) if smoothed_scores else 0.0
            score_range = max(abs(score_max - score_min), 1e-6)
            normalized_scores = [
                (float(score) - score_min) / score_range for score in smoothed_scores
            ]

            gaps = []
            for idx in range(0, len(normalized_scores) - 1):
                gap = max(
                    0.0,
                    float(normalized_scores[idx]) - float(normalized_scores[idx + 1]),
                )
                gaps.append(
                    {
                        "after_rank": int(idx + 1),
                        "next_rank": int(idx + 2),
                        "gap": float(gap),
                    }
                )

            n_scores = len(normalized_scores)
            if n_scores <= 1:
                selected_count = n_scores
                gap_median = 0.0
                gap_spread = 0.0
                expected_k = float(n_scores)
                boundary_candidates = []
                stop_reason = "single_candidate"
            else:
                gap_values = [float(item["gap"]) for item in gaps]
                gap_array = np.array(gap_values, dtype=float)
                gap_median = float(np.median(gap_array))
                gap_mad = float(np.median(np.abs(gap_array - gap_median)))
                gap_spread = 1.4826 * gap_mad
                if gap_spread <= 1e-8:
                    gap_spread = float(np.std(gap_array))
                gap_spread = max(0.05, gap_spread)

                max_considered = min(max_clips, n_scores)
                candidate_ks = list(range(min_clips, max_considered + 1))
                boundary_candidates = []
                weights = []
                for candidate_k in candidate_ks:
                    if candidate_k < max_considered and candidate_k - 1 < len(gap_values):
                        gap = float(gap_values[candidate_k - 1])
                        z_score = (gap - gap_median) / gap_spread
                        # Gaussian CDF: large adjacent gaps are stronger boundary evidence.
                        weight = 0.5 * (1.0 + math.erf(z_score / math.sqrt(2.0)))
                        reason = "boundary_after_rank"
                    else:
                        gap = None
                        z_score = 0.0
                        weight = 0.5
                        reason = "no_boundary_before_max"
                    weight = max(1e-6, float(weight))
                    weights.append(weight)
                    boundary_candidates.append(
                        {
                            "selected_count": int(candidate_k),
                            "gap": None if gap is None else float(gap),
                            "z_score": float(z_score),
                            "weight": float(weight),
                            "reason": reason,
                        }
                    )

                weight_sum = max(float(sum(weights)), 1e-8)
                expected_k = float(
                    sum(float(k) * float(w) for k, w in zip(candidate_ks, weights))
                    / weight_sum
                )
                selected_count = int(math.floor(expected_k + 0.5))
                stop_reason = "robust_z_boundary_posterior"

            selected_count = max(1, min(max_clips, len(ranked_steps), selected_count))
            selected = [step["clip"] for step in ranked_steps[:selected_count]]
            if trace is not None:
                trace["policy"] = policy
                trace["uncertainty_alpha"] = float(uncertainty_alpha)
                for step in ranked_steps:
                    event = "select" if step["rank"] <= selected_count else "candidate"
                    trace.setdefault("steps", []).append(
                        {
                            "event": event,
                            "clip": int(step["clip"]),
                            "rank": int(step["rank"]),
                            "mmr_score": float(step["mmr_score"]),
                            "relevance": float(step["relevance"]),
                            "redundancy": float(step["redundancy"]),
                        }
                    )
                trace.setdefault("steps", []).append(
                    {
                        "event": "stop",
                        "reason": stop_reason,
                        "policy": policy,
                        "gap_median": float(gap_median),
                        "gap_spread": float(gap_spread),
                        "expected_k": float(expected_k),
                        "boundary_candidates": boundary_candidates,
                        "gaps": gaps,
                        "smoothed_scores": [float(score) for score in smoothed_scores],
                        "normalized_scores": [float(score) for score in normalized_scores],
                        "selected_count": int(selected_count),
                        "selected_so_far": [int(cid) for cid in selected],
                    }
                )
            return selected

        if policy in {"adaptive_gap_uncertainty", "self_adaptive_gap", "adjacent_gap_uncertainty"}:
            selected_count = min(min_clips, len(ranked_steps))
            smoothed_scores = list(scores)
            for idx in range(1, len(smoothed_scores)):
                smoothed_scores[idx] = min(smoothed_scores[idx - 1], smoothed_scores[idx])

            drops = []
            for idx in range(0, len(smoothed_scores) - 1):
                current = float(smoothed_scores[idx])
                next_score = float(smoothed_scores[idx + 1])
                drop = max(0.0, current - next_score) / max(abs(current), 1e-6)
                drops.append(
                    {
                        "after_rank": int(idx + 1),
                        "next_rank": int(idx + 2),
                        "drop": float(drop),
                    }
                )

            drop_values = [item["drop"] for item in drops]
            drop_mean = float(np.mean(drop_values)) if drop_values else 0.0
            drop_std = float(np.std(drop_values)) if drop_values else 0.0
            close_threshold = max(0.0, drop_mean - float(uncertainty_alpha) * drop_std)
            stop_reason = "no_more_candidates"
            decisions = []
            for idx in range(max(0, selected_count - 1), len(drops)):
                item = drops[idx]
                if item["drop"] <= close_threshold:
                    selected_count = int(item["next_rank"])
                    decisions.append(
                        {
                            "type": "close_adjacent_gap_continue",
                            "from_rank": int(item["after_rank"]),
                            "to_rank": int(item["next_rank"]),
                            "drop": float(item["drop"]),
                            "threshold": float(close_threshold),
                        }
                    )
                    stop_reason = "max_or_no_more_close_gaps"
                else:
                    stop_reason = "large_adjacent_gap_stop"
                    decisions.append(
                        {
                            "type": "large_adjacent_gap_stop",
                            "from_rank": int(item["after_rank"]),
                            "to_rank": int(item["next_rank"]),
                            "drop": float(item["drop"]),
                            "threshold": float(close_threshold),
                        }
                    )
                    break

            selected_count = max(1, min(max_clips, len(ranked_steps), selected_count))
            selected = [step["clip"] for step in ranked_steps[:selected_count]]
            if trace is not None:
                trace["policy"] = policy
                trace["uncertainty_alpha"] = float(uncertainty_alpha)
                for step in ranked_steps:
                    event = "select" if step["rank"] <= selected_count else "candidate"
                    trace.setdefault("steps", []).append(
                        {
                            "event": event,
                            "clip": int(step["clip"]),
                            "rank": int(step["rank"]),
                            "mmr_score": float(step["mmr_score"]),
                            "relevance": float(step["relevance"]),
                            "redundancy": float(step["redundancy"]),
                        }
                    )
                trace.setdefault("steps", []).append(
                    {
                        "event": "stop",
                        "reason": stop_reason,
                        "policy": policy,
                        "uncertainty_alpha": float(uncertainty_alpha),
                        "drop_mean": float(drop_mean),
                        "drop_std": float(drop_std),
                        "close_threshold": float(close_threshold),
                        "drops": drops,
                        "decisions": decisions,
                        "smoothed_scores": [float(score) for score in smoothed_scores],
                        "selected_count": int(selected_count),
                        "selected_so_far": [int(cid) for cid in selected],
                    }
                )
            return selected

        if policy in {"knee", "elbow", "saturation_knee", "adaptive_knee", "self_adaptive_knee"}:
            selected_count = min(min_clips, len(ranked_steps))
            smoothed_scores = list(scores)
            for idx in range(1, len(smoothed_scores)):
                smoothed_scores[idx] = min(smoothed_scores[idx - 1], smoothed_scores[idx])

            drops = []
            for idx in range(max(0, selected_count - 1), len(smoothed_scores) - 1):
                current = float(smoothed_scores[idx])
                next_score = float(smoothed_scores[idx + 1])
                drop = max(0.0, current - next_score) / max(abs(current), 1e-6)
                drops.append(
                    {
                        "after_rank": int(idx + 1),
                        "next_rank": int(idx + 2),
                        "drop": float(drop),
                    }
                )

            best_drop = max((item["drop"] for item in drops), default=0.0)
            drop_values = [item["drop"] for item in drops]
            drop_mean = float(np.mean(drop_values)) if drop_values else 0.0
            drop_std = float(np.std(drop_values)) if drop_values else 0.0
            if policy in {"adaptive_knee", "self_adaptive_knee"}:
                adaptive_threshold = drop_mean + float(knee_alpha) * drop_std
                best_item = max(drops, key=lambda item: item["drop"], default=None)
                if best_item is not None and best_drop >= adaptive_threshold and best_drop > drop_mean:
                    selected_count = int(best_item["after_rank"])
                    stop_reason = "adaptive_knee_detected"
                else:
                    selected_count = len(ranked_steps)
                    stop_reason = "no_adaptive_knee_return_max"
            else:
                adaptive_threshold = None
                knee_candidates = [item for item in drops if item["drop"] >= float(knee_min_drop)]
                if knee_candidates:
                    best_item = max(knee_candidates, key=lambda item: item["drop"])
                    selected_count = int(best_item["after_rank"])
                    stop_reason = "knee_detected"
                else:
                    selected_count = len(ranked_steps)
                    stop_reason = "no_knee_return_max"

            selected_count = max(1, min(max_clips, len(ranked_steps), selected_count))
            selected = [step["clip"] for step in ranked_steps[:selected_count]]
            if trace is not None:
                trace["policy"] = policy
                trace["knee_min_drop"] = float(knee_min_drop)
                trace["knee_alpha"] = float(knee_alpha)
                for step in ranked_steps:
                    event = "select" if step["rank"] <= selected_count else "candidate"
                    trace.setdefault("steps", []).append(
                        {
                            "event": event,
                            "clip": int(step["clip"]),
                            "rank": int(step["rank"]),
                            "mmr_score": float(step["mmr_score"]),
                            "relevance": float(step["relevance"]),
                            "redundancy": float(step["redundancy"]),
                        }
                    )
                trace.setdefault("steps", []).append(
                    {
                        "event": "stop",
                        "reason": stop_reason,
                        "policy": policy,
                        "knee_min_drop": float(knee_min_drop),
                        "knee_alpha": float(knee_alpha),
                        "adaptive_threshold": None if adaptive_threshold is None else float(adaptive_threshold),
                        "drop_mean": float(drop_mean),
                        "drop_std": float(drop_std),
                        "best_drop": float(best_drop),
                        "drops": drops,
                        "smoothed_scores": [float(score) for score in smoothed_scores],
                        "selected_count": int(selected_count),
                        "selected_so_far": [int(cid) for cid in selected],
                    }
                )
            return selected

        selected_count = min(min_clips, len(ranked_steps))
        decision = {
            "event": "stop",
            "reason": "uncertainty_policy",
            "policy": policy,
            "confidence_threshold": float(confidence_threshold),
            "ambiguity_gap_threshold": float(ambiguity_gap_threshold),
            "initial_k": int(selected_count),
            "bonuses": [],
        }
        if len(scores) >= 2 and scores[1] < float(confidence_threshold):
            selected_count += 1
            decision["bonuses"].append(
                {
                    "type": "low_confidence",
                    "rank": 2,
                    "score": float(scores[1]),
                    "threshold": float(confidence_threshold),
                }
            )
        for idx in (1, 2):
            if idx + 1 < len(scores):
                gap = (scores[idx] - scores[idx + 1]) / (abs(scores[idx]) + 1e-6)
                if gap < float(ambiguity_gap_threshold):
                    selected_count += 1
                    decision["bonuses"].append(
                        {
                            "type": "ambiguous_gap",
                            "from_rank": int(idx + 1),
                            "to_rank": int(idx + 2),
                            "gap": float(gap),
                            "threshold": float(ambiguity_gap_threshold),
                        }
                    )
        selected_count = max(1, min(max_clips, len(ranked_steps), selected_count))
        selected = [step["clip"] for step in ranked_steps[:selected_count]]
        if trace is not None:
            trace["policy"] = policy
            trace["confidence_threshold"] = float(confidence_threshold)
            trace["ambiguity_gap_threshold"] = float(ambiguity_gap_threshold)
            trace["knee_min_drop"] = float(knee_min_drop)
            for step in ranked_steps:
                event = "select" if step["rank"] <= selected_count else "candidate"
                trace.setdefault("steps", []).append(
                    {
                        "event": event,
                        "clip": int(step["clip"]),
                        "rank": int(step["rank"]),
                        "mmr_score": float(step["mmr_score"]),
                        "relevance": float(step["relevance"]),
                        "redundancy": float(step["redundancy"]),
                    }
                )
            decision["selected_count"] = int(selected_count)
            decision["selected_so_far"] = [int(cid) for cid in selected]
            trace.setdefault("steps", []).append(decision)
        return selected

    selected = []
    remaining = list(candidate_ids)

    while remaining and len(selected) < max_clips:
        best_id = None
        best_objective = -1e9
        best_relevance = 0.0
        best_redundancy = 0.0
        for cid in remaining:
            relevance = rel_scores.get(cid, 0.0)
            redundancy = 0.0
            curr_repr = clip_repr_map.get(cid)
            if curr_repr is not None and selected:
                sims = [
                    float(np.dot(curr_repr, clip_repr_map[sel]))
                    for sel in selected
                    if clip_repr_map.get(sel) is not None
                ]
                redundancy = max(sims) if sims else 0.0
            objective = relevance if not selected else mmr_lambda * relevance - (1.0 - mmr_lambda) * redundancy
            if objective > best_objective:
                best_objective = objective
                best_id = cid
                best_relevance = relevance
                best_redundancy = redundancy

        if best_id is None:
            if trace is not None:
                trace.setdefault("steps", []).append({"event": "stop", "reason": "no_candidate"})
            break
        if selected and len(selected) >= min_clips and best_objective < float(stop_threshold):
            if trace is not None:
                trace.setdefault("steps", []).append(
                    {
                        "event": "stop",
                        "reason": "below_threshold",
                        "candidate_clip": int(best_id),
                        "mmr_score": float(best_objective),
                        "relevance": float(best_relevance),
                        "redundancy": float(best_redundancy),
                        "threshold": float(stop_threshold),
                        "selected_so_far": [int(cid) for cid in selected],
                    }
                )
            break
        if trace is not None:
            trace.setdefault("steps", []).append(
                {
                    "event": "select",
                    "clip": int(best_id),
                    "rank": len(selected) + 1,
                    "mmr_score": float(best_objective),
                    "relevance": float(best_relevance),
                    "redundancy": float(best_redundancy),
                }
            )
        selected.append(best_id)
        remaining.remove(best_id)

    return selected


_EVIDENCE_STOPWORDS = {
    "the",
    "and",
    "for",
    "with",
    "that",
    "this",
    "from",
    "are",
    "was",
    "were",
    "has",
    "have",
    "had",
    "his",
    "her",
    "their",
    "its",
    "into",
    "onto",
    "there",
    "here",
    "then",
    "than",
    "they",
    "them",
    "him",
    "she",
    "you",
    "who",
    "what",
    "when",
    "where",
    "which",
    "clip",
    "person",
    "people",
    "someone",
    "something",
}

_ACTION_STATE_HINTS = {
    "add",
    "added",
    "adjust",
    "apply",
    "arrive",
    "ask",
    "begin",
    "carry",
    "change",
    "clean",
    "close",
    "come",
    "continue",
    "cook",
    "cut",
    "drink",
    "drive",
    "eat",
    "enter",
    "finish",
    "give",
    "go",
    "grab",
    "hold",
    "leave",
    "look",
    "make",
    "move",
    "open",
    "pick",
    "place",
    "play",
    "pour",
    "put",
    "read",
    "remove",
    "ride",
    "say",
    "sit",
    "stand",
    "start",
    "stop",
    "take",
    "talk",
    "turn",
    "use",
    "walk",
    "wash",
    "wear",
    "worn",
    "empty",
    "full",
    "closed",
    "open",
    "visible",
}


def _simple_token_stem(token):
    token = token.lower().strip("'_-")
    for suffix in ("ing", "ed", "es", "s"):
        if len(token) > len(suffix) + 3 and token.endswith(suffix):
            return token[: -len(suffix)]
    return token


def _iter_node_texts(video_graph, node_id):
    node = video_graph.nodes.get(node_id)
    if node is None:
        return []
    contents = node.metadata.get("contents", [])
    if isinstance(contents, str):
        contents = [contents]
    return [text for text in contents if isinstance(text, str)]


def _extract_evidence_tokens(text):
    lowered = text.lower()
    entity_tokens = set(re.findall(r"<(?:character|face|voice|object|scene)_\d+>", lowered))
    lexical_tokens = set()
    action_state_tokens = set()
    for raw_token in re.findall(r"[a-z][a-z0-9_\-']{2,}", lowered):
        token = _simple_token_stem(raw_token)
        if token in _EVIDENCE_STOPWORDS or len(token) < 3:
            continue
        lexical_tokens.add(token)
        if token in _ACTION_STATE_HINTS or raw_token.endswith(("ing", "ed")):
            action_state_tokens.add(token)
    return entity_tokens, lexical_tokens, action_state_tokens


def _jaccard(a, b):
    if not a or not b:
        return 0.0
    return len(a & b) / max(1, len(a | b))


def _clip_evidence_profile(video_graph, clip_id, node_hits, *, max_nodes=8, intra_clip_sim_threshold=0.85):
    clusters = _group_clip_nodes(
        video_graph,
        node_hits,
        max_nodes=max_nodes,
        sim_threshold=intra_clip_sim_threshold,
    )
    entity_tokens = set()
    lexical_tokens = set()
    action_state_tokens = set()
    for node_id, _ in sorted(node_hits, key=lambda x: x[1], reverse=True)[: max(1, int(max_nodes))]:
        for text in _iter_node_texts(video_graph, node_id):
            ents, toks, acts = _extract_evidence_tokens(text)
            entity_tokens.update(ents)
            lexical_tokens.update(toks)
            action_state_tokens.update(acts)

    return {
        "clip_id": int(clip_id),
        "cluster_centroids": [
            np.asarray(cluster["centroid"], dtype=np.float32)
            for cluster in clusters
            if cluster.get("centroid") is not None
        ],
        "entity_tokens": entity_tokens,
        "lexical_tokens": lexical_tokens,
        "action_state_tokens": action_state_tokens,
    }


def _centroid_novelty_gain(centroids, covered_centroids, sim_threshold=0.86):
    if not centroids:
        return 0.0
    if not covered_centroids:
        return 1.0
    novel = 0
    for centroid in centroids:
        max_sim = max(float(np.dot(centroid, prev)) for prev in covered_centroids)
        if max_sim < sim_threshold:
            novel += 1
    return novel / max(1, len(centroids))


def _token_novelty_gain(tokens, covered_tokens):
    if not tokens:
        return 0.0
    return len(tokens - covered_tokens) / max(1, len(tokens))


def _temporal_redundancy(profile, selected_profiles, near_clip_window=2):
    if not selected_profiles:
        return 0.0
    cid = int(profile["clip_id"])
    best = 0.0
    for selected in selected_profiles:
        gap = abs(cid - int(selected["clip_id"]))
        if gap > int(near_clip_window):
            continue
        closeness = (int(near_clip_window) + 1 - gap) / max(1, int(near_clip_window) + 1)
        overlap = _jaccard(profile["lexical_tokens"], selected["lexical_tokens"])
        best = max(best, closeness * (0.5 + 0.5 * overlap))
    return float(best)


def _evidence_saturation_select_clips(
    video_graph,
    candidate_ids,
    clip_scores,
    clip_repr_map,
    clip_node_hits,
    *,
    min_clips=1,
    max_clips=5,
    stop_threshold=0.02,
    relevance_weight=0.75,
    semantic_gain_weight=0.10,
    temporal_gain_weight=0.08,
    entity_gain_weight=0.06,
    action_state_gain_weight=0.06,
    semantic_redundancy_weight=0.15,
    temporal_redundancy_weight=0.0,
    temporal_bucket_size=4,
    near_clip_window=2,
    max_nodes_per_clip=8,
    intra_clip_sim_threshold=0.85,
):
    if not candidate_ids:
        return []

    min_clips = max(1, int(min_clips))
    max_clips = max(min_clips, int(max_clips))
    max_clips = min(max_clips, len(candidate_ids))
    raw_scores = [float(clip_scores.get(cid, 0.0)) for cid in candidate_ids]
    s_min = min(raw_scores) if raw_scores else 0.0
    s_max = max(raw_scores) if raw_scores else 0.0
    if s_max > s_min:
        rel_scores = {cid: (float(clip_scores.get(cid, 0.0)) - s_min) / (s_max - s_min) for cid in candidate_ids}
    else:
        rel_scores = {cid: 1.0 for cid in candidate_ids}

    profiles = {
        cid: _clip_evidence_profile(
            video_graph,
            cid,
            clip_node_hits.get(cid, []),
            max_nodes=max_nodes_per_clip,
            intra_clip_sim_threshold=intra_clip_sim_threshold,
        )
        for cid in candidate_ids
    }

    selected = []
    selected_profiles = []
    remaining = list(candidate_ids)
    covered_centroids = []
    covered_entities = set()
    covered_actions = set()
    covered_temporal_buckets = set()

    while remaining and len(selected) < max_clips:
        best_id = None
        best_objective = -1e9
        best_saturation_gain = -1e9

        for cid in remaining:
            profile = profiles[cid]
            relevance = rel_scores.get(cid, 0.0)
            semantic_gain = _centroid_novelty_gain(profile["cluster_centroids"], covered_centroids)
            entity_gain = _token_novelty_gain(profile["entity_tokens"], covered_entities)
            action_gain = _token_novelty_gain(profile["action_state_tokens"], covered_actions)
            bucket_size = max(1, int(temporal_bucket_size))
            bucket = int(cid) // bucket_size
            temporal_gain = 0.0 if bucket in covered_temporal_buckets else 1.0
            semantic_red = _semantic_redundancy(cid, selected, clip_repr_map)
            temporal_red = _temporal_redundancy(
                profile,
                selected_profiles,
                near_clip_window=near_clip_window,
            )
            evidence_gain = (
                float(semantic_gain_weight) * semantic_gain
                + float(temporal_gain_weight) * temporal_gain
                + float(entity_gain_weight) * entity_gain
                + float(action_state_gain_weight) * action_gain
            )
            redundancy_penalty = (
                float(semantic_redundancy_weight) * semantic_red
                + float(temporal_redundancy_weight) * temporal_red
            )
            saturation_gain = evidence_gain - redundancy_penalty
            objective = float(relevance_weight) * relevance + saturation_gain
            if objective > best_objective:
                best_objective = objective
                best_saturation_gain = saturation_gain
                best_id = cid

        if best_id is None:
            break
        if selected and len(selected) >= min_clips and best_saturation_gain < float(stop_threshold):
            break

        selected.append(best_id)
        remaining.remove(best_id)
        best_profile = profiles[best_id]
        selected_profiles.append(best_profile)
        covered_centroids.extend(best_profile["cluster_centroids"])
        covered_entities.update(best_profile["entity_tokens"])
        covered_actions.update(best_profile["action_state_tokens"])
        covered_temporal_buckets.add(int(best_id) // max(1, int(temporal_bucket_size)))

    return selected


def _canonical_role(role):
    role = re.sub(r"[^a-zA-Z0-9_]+", "_", str(role or "").strip().lower()).strip("_")
    if role in EVIDENCE_ROLE_DEFINITIONS:
        return role
    return _ROLE_ALIASES.get(role)


def _clip_memory_snippets(video_graph, clip_id, clip_node_hits, max_nodes=4, max_chars_per_node=360):
    node_hits = sorted(clip_node_hits.get(clip_id, []), key=lambda x: x[1], reverse=True)
    node_ids = [node_id for node_id, _ in node_hits[: max(1, int(max_nodes))]]
    if not node_ids and clip_id in getattr(video_graph, "text_nodes_by_clip", {}):
        node_ids = list(video_graph.text_nodes_by_clip[clip_id])[: max(1, int(max_nodes))]

    snippets = []
    seen = set()
    for node_id in node_ids:
        if node_id in seen or node_id not in video_graph.nodes:
            continue
        seen.add(node_id)
        node = video_graph.nodes[node_id]
        contents = node.metadata.get("contents", [])
        if isinstance(contents, str):
            contents = [contents]
        if not contents:
            continue
        translated = translate(video_graph, contents[:1])
        if not translated:
            continue
        text = " ".join(str(x) for x in translated if x)
        text = re.sub(r"\s+", " ", text).strip()
        if not text:
            continue
        if len(text) > max_chars_per_node:
            text = text[:max_chars_per_node].rstrip() + "..."
        snippets.append(f"- [{getattr(node, 'type', 'memory')}] {text}")
    return snippets


def _extract_json_payload(text):
    text = str(text or "").strip()
    if text.startswith("```"):
        text = re.sub(r"^```(?:json)?\s*", "", text, flags=re.IGNORECASE)
        text = re.sub(r"\s*```$", "", text)
    try:
        return json.loads(text)
    except Exception:
        pass

    start = text.find("{")
    end = text.rfind("}")
    if start >= 0 and end > start:
        try:
            return json.loads(text[start:end + 1])
        except Exception:
            return None
    return None


def _role_cache_path(cache_dir, key):
    if not cache_dir:
        return None
    return os.path.join(cache_dir, f"{key}.json")


def _load_role_cache(cache_dir, key):
    path = _role_cache_path(cache_dir, key)
    if not path or not os.path.exists(path):
        return None
    try:
        with open(path, "r", encoding="utf-8") as f:
            return json.load(f)
    except Exception:
        return None


def _save_role_cache(cache_dir, key, payload):
    path = _role_cache_path(cache_dir, key)
    if not path:
        return
    try:
        os.makedirs(cache_dir, exist_ok=True)
        tmp_path = f"{path}.{os.getpid()}.tmp"
        with open(tmp_path, "w", encoding="utf-8") as f:
            json.dump(payload, f, ensure_ascii=False)
        os.replace(tmp_path, path)
    except Exception as exc:
        logger.warning("Failed to write role-aware retrieval cache: %s", exc)


def _build_role_label_prompt(query, candidate_payload):
    role_lines = "\n".join(
        f"- {role}: {definition}"
        for role, definition in EVIDENCE_ROLE_DEFINITIONS.items()
    )
    clip_blocks = []
    for item in candidate_payload:
        snippets = "\n".join(item["snippets"]) if item["snippets"] else "- No memory text available."
        clip_blocks.append(f"CLIP_{item['clip_id']}:\n{snippets}")

    return f"""You are labeling evidence for a long-term video-memory retrieval agent.
Do not answer the question. Only identify what evidence roles each candidate clip can support.

Question or search query:
{query}

Allowed evidence roles:
{role_lines}

Candidate clips:
{chr(10).join(clip_blocks)}

Return valid JSON only, with this schema:
{{
  "question_roles": [
    {{"role": "one allowed role", "weight": 0.8}}
  ],
  "clips": [
    {{
      "clip_id": 0,
      "roles": ["one allowed role"],
      "evidence_units": [
        {{
          "role": "one allowed role",
          "subject": "short subject/entity",
          "predicate": "short action/state/count/temporal relation",
          "object": "short object/location/result if any",
          "time_hint": "short moment/order hint if any"
        }}
      ]
    }}
  ]
}}

Rules:
- Use only the allowed role names.
- question_roles are the evidence roles likely needed by the query, not a predicted answer.
- Role weights must be real support scores from 0.1 to 1.0. Never output 0.0 for a role you include.
- If a role or clip has no useful evidence, omit that role/clip instead of assigning 0.0.
- Use weight 1.0 for essential evidence roles, around 0.6 for helpful roles, and around 0.3 for weak but relevant roles.
- evidence_units should distinguish genuinely different people, objects, actions, locations, counts, or moments.
- If two clips are semantically similar but contain different necessary instances, give them different evidence_units.
- Do not include explanations outside JSON."""


def _role_cache_key(query, candidate_payload):
    compact = json.dumps(
        {"query": query, "clips": candidate_payload},
        ensure_ascii=False,
        sort_keys=True,
    )
    return hashlib.sha1(compact.encode("utf-8", "ignore")).hexdigest()


def _question_role_cache_key(query):
    compact = re.sub(r"\s+", " ", str(query or "").strip())
    return hashlib.sha1(compact.encode("utf-8", "ignore")).hexdigest()


def _build_question_role_prompt(question):
    role_lines = "\n".join(
        f"- {role}: {definition}"
        for role, definition in EVIDENCE_ROLE_DEFINITIONS.items()
    )
    return f"""You are identifying what evidence roles are needed to answer a question for a long-term video-memory retrieval agent.
Do not answer the question. Only select necessary evidence roles.

Question:
{question}

Allowed evidence roles:
{role_lines}

Return valid JSON only:
{{
  "roles": ["one allowed role"]
}}

Rules:
- Use only the allowed role names.
- Select all roles that would be useful evidence for answering the question.
- Do not assign weights, scores, confidence, or explanations.
- If uncertain, include the minimal set of likely necessary roles.
- Output JSON only."""


def _normalize_question_role_payload(payload):
    if isinstance(payload, dict):
        raw_roles = payload.get("roles", []) or payload.get("question_roles", []) or []
    elif isinstance(payload, list):
        raw_roles = payload
    else:
        raw_roles = []

    roles = []
    for item in raw_roles:
        raw_role = item.get("role") if isinstance(item, dict) else item
        role = _canonical_role(raw_role)
        if role and role not in roles:
            roles.append(role)
    return {role: 1.0 for role in roles}


def _load_precomputed_question_role_weights(question_roles_dir, question):
    if not question_roles_dir:
        return None
    key = _question_role_cache_key(question)
    path = os.path.join(str(question_roles_dir), f"{key}.json")
    if not os.path.exists(path):
        return None
    try:
        with open(path, "r", encoding="utf-8") as f:
            payload = json.load(f)
        weights = _normalize_question_role_payload(payload)
        return weights or None
    except Exception as exc:
        logger.warning("Failed to read precomputed question roles %s: %s", path, exc)
        return None


def _get_question_role_weights(
    role_query,
    *,
    mode="heuristic",
    model="local-qwen3-vl",
    model_device=None,
    max_new_tokens=512,
    cache_dir=None,
    question_roles_dir=None,
):
    mode = str(mode or "heuristic").strip().lower()
    if mode in {"heuristic", "rules", "rule"}:
        return _infer_question_role_weights(role_query)

    if mode in {"qwen_binary", "qwen", "binary"}:
        cached_weights = _load_precomputed_question_role_weights(question_roles_dir, role_query)
        if cached_weights:
            return cached_weights

        key = f"question_roles_{_question_role_cache_key(role_query)}"
        cached = _load_role_cache(cache_dir, key)
        if cached is not None:
            weights = _normalize_question_role_payload(cached)
            if weights:
                return weights

        prompt = _build_question_role_prompt(role_query)
        messages = generate_messages([{"type": "text", "content": prompt}])
        try:
            response = _get_role_label_response(
                model,
                messages,
                model_device=model_device,
                max_new_tokens=max_new_tokens,
            )
            payload = _extract_json_payload(response)
            if payload is None:
                raise ValueError(f"invalid JSON response: {response[:200]}")
            weights = _normalize_question_role_payload(payload)
            if weights:
                _save_role_cache(cache_dir, key, payload)
                return weights
        except Exception as exc:
            logger.warning(
                "Question role labeling failed; falling back to heuristic roles: %s",
                exc,
            )
        return _infer_question_role_weights(role_query)

    raise ValueError(f"Unknown role-aware question role mode: {mode}")


def _normalize_role_payload(raw_payload, candidate_ids):
    candidate_set = set(candidate_ids)
    payload = raw_payload if isinstance(raw_payload, dict) else {}

    question_role_weights = {}
    for item in payload.get("question_roles", []) or []:
        if not isinstance(item, dict):
            continue
        role = _canonical_role(item.get("role"))
        if not role:
            continue
        try:
            weight = float(item.get("weight", 1.0))
        except Exception:
            weight = 1.0
        if weight <= 0:
            weight = 0.5
        question_role_weights[role] = max(question_role_weights.get(role, 0.0), min(max(weight, 0.1), 1.0))

    clip_profiles = {
        cid: {
            "roles": set(),
            "units": set(),
            "unit_roles": {},
        }
        for cid in candidate_ids
    }
    for item in payload.get("clips", []) or []:
        if not isinstance(item, dict):
            continue
        try:
            cid = int(str(item.get("clip_id", "")).replace("CLIP_", ""))
        except Exception:
            continue
        if cid not in candidate_set:
            continue

        profile = clip_profiles[cid]
        for raw_role in item.get("roles", []) or []:
            role = _canonical_role(raw_role)
            if role:
                profile["roles"].add(role)
        for unit in item.get("evidence_units", []) or []:
            if not isinstance(unit, dict):
                continue
            role = _canonical_role(unit.get("role"))
            if not role:
                continue
            profile["roles"].add(role)
            subject = re.sub(r"\s+", " ", str(unit.get("subject", "")).strip().lower())
            predicate = re.sub(r"\s+", " ", str(unit.get("predicate", "")).strip().lower())
            obj = re.sub(r"\s+", " ", str(unit.get("object", "")).strip().lower())
            time_hint = re.sub(r"\s+", " ", str(unit.get("time_hint", "")).strip().lower())
            unit_key = "|".join([role, subject, predicate, obj, time_hint]).strip("|")
            if unit_key == role:
                unit_key = f"{role}|generic"
            profile["units"].add(unit_key)
            profile["unit_roles"][unit_key] = role
    if not question_role_weights:
        role_union = set()
        for profile in clip_profiles.values():
            role_union.update(profile["roles"])
        question_role_weights = {role: 1.0 for role in role_union}

    return question_role_weights, clip_profiles


def _get_role_profiles(
    video_graph,
    role_query,
    candidate_ids,
    clip_node_hits,
    *,
    model="local-qwen3-vl",
    model_device=None,
    max_new_tokens=2048,
    cache_dir=None,
    max_nodes_per_clip=4,
):
    candidate_payload = [
        {
            "clip_id": int(cid),
            "snippets": _clip_memory_snippets(
                video_graph,
                cid,
                clip_node_hits,
                max_nodes=max_nodes_per_clip,
            ),
        }
        for cid in candidate_ids
    ]
    key = _role_cache_key(role_query, candidate_payload)
    cached = _load_role_cache(cache_dir, key)
    if cached is not None:
        return _normalize_role_payload(cached, candidate_ids)

    prompt = _build_role_label_prompt(role_query, candidate_payload)
    messages = generate_messages([{"type": "text", "content": prompt}])
    response = _get_role_label_response(
        model,
        messages,
        model_device=model_device,
        max_new_tokens=max_new_tokens,
    )
    payload = _extract_json_payload(response)
    if payload is None:
        raise ValueError(f"Role-aware retrieval model returned invalid JSON: {response[:200]}")
    _save_role_cache(cache_dir, key, payload)
    return _normalize_role_payload(payload, candidate_ids)


def _resolve_local_role_model_path(model):
    model = str(model or "").strip()
    lowered = model.lower()
    if lowered in {"local-qwen3-vl", "qwen3-vl-8b", "qwen3vl_8b", "qwen3vl-8b"}:
        return "models/Qwen3-VL-8B-Instruct"
    if lowered.startswith("local:"):
        return model.split(":", 1)[1]
    if os.path.exists(model):
        return model
    return None


def _get_role_label_response(model, messages, *, model_device=None, max_new_tokens=2048):
    local_model_path = _resolve_local_role_model_path(model)
    if local_model_path:
        from .utils.chat_qwen3_vl import (
            generate_messages as generate_local_qwen3_vl_messages,
            get_response as get_local_qwen3_vl_response,
        )

        local_messages = generate_local_qwen3_vl_messages(messages)
        response, _ = get_local_qwen3_vl_response(
            local_messages,
            model_path=local_model_path,
            model_device=model_device or os.environ.get("ROLE_AWARE_MODEL_DEVICE"),
            max_new_tokens=max_new_tokens,
            do_sample=False,
        )
        return response

    return get_response_with_retry(model, messages, timeout=90)[0]


def _infer_question_role_weights(role_query):
    text = str(role_query or "").lower()
    weights = {}

    def add(role, weight):
        weights[role] = max(weights.get(role, 0.0), weight)

    if re.search(r"\b(who|whose|person|people|character|name|wearing|holding)\b", text):
        add("entity_identity", 1.0)
    if re.search(r"\b(where|location|located|room|place|area|near|on the|in the)\b", text):
        add("spatial_location", 1.0)
    if re.search(r"\b(how many|number of|count|total|several|times|multiple)\b", text):
        add("count_instance", 1.0)
        add("action_event", 0.6)
    if re.search(r"\b(before|after|first|last|then|next|previous|later|earlier|when|order|sequence)\b", text):
        add("temporal_order", 1.0)
        add("action_event", 0.6)
    if re.search(r"\b(color|colour|state|condition|appearance|look like|status|open|closed|empty|full)\b", text):
        add("state_attribute", 1.0)
    if re.search(r"\b(what did|doing|do |does |happen|happened|action|activity|pick|put|take|move|open|close)\b", text):
        add("action_event", 1.0)

    if not weights:
        add("action_event", 0.8)
        add("entity_identity", 0.5)
        add("state_attribute", 0.5)
    return weights


def _video_id_from_graph(video_graph):
    for attr in ("_m3agent_video_id", "video_id", "id"):
        value = getattr(video_graph, attr, None)
        if value:
            return str(value)
    mem_path = getattr(video_graph, "_m3agent_mem_path", None)
    if mem_path:
        return os.path.splitext(os.path.basename(str(mem_path)))[0]
    return None


def _load_precomputed_clip_profile(precomputed_dir, video_id, clip_id):
    path = os.path.join(str(precomputed_dir), str(video_id), f"clip_{int(clip_id):06d}.json")
    if not os.path.exists(path):
        return None
    try:
        with open(path, "r", encoding="utf-8") as f:
            payload = json.load(f)
    except Exception as exc:
        logger.warning("Failed to read precomputed role profile %s: %s", path, exc)
        return None

    profile = {"roles": set(), "units": set(), "unit_roles": {}}
    for raw_role in payload.get("roles", []) or []:
        role = _canonical_role(raw_role)
        if role:
            profile["roles"].add(role)
    for unit in payload.get("evidence_units", []) or []:
        if not isinstance(unit, dict):
            continue
        role = _canonical_role(unit.get("role"))
        if not role:
            continue
        profile["roles"].add(role)
        subject = re.sub(r"\s+", " ", str(unit.get("subject", "")).strip().lower())
        predicate = re.sub(r"\s+", " ", str(unit.get("predicate", "")).strip().lower())
        obj = re.sub(r"\s+", " ", str(unit.get("object", "")).strip().lower())
        time_hint = re.sub(r"\s+", " ", str(unit.get("time_hint", "")).strip().lower())
        unit_key = "|".join([role, subject, predicate, obj, time_hint]).strip("|")
        if unit_key == role:
            unit_key = f"{role}|generic"
        profile["units"].add(unit_key)
        profile["unit_roles"][unit_key] = role
    return profile


def _get_precomputed_role_profiles(
    video_graph,
    role_query,
    candidate_ids,
    precomputed_dir,
    *,
    question_role_mode="heuristic",
    model="local-qwen3-vl",
    model_device=None,
    max_new_tokens=512,
    cache_dir=None,
    question_roles_dir=None,
):
    video_id = _video_id_from_graph(video_graph)
    if not video_id:
        return None
    clip_profiles = {}
    loaded = 0
    for cid in candidate_ids:
        profile = _load_precomputed_clip_profile(precomputed_dir, video_id, cid)
        if profile is None:
            profile = {"roles": set(), "units": set(), "unit_roles": {}}
        else:
            loaded += 1
        clip_profiles[cid] = profile
    if loaded == 0:
        return None
    return (
        _get_question_role_weights(
            role_query,
            mode=question_role_mode,
            model=model,
            model_device=model_device,
            max_new_tokens=max_new_tokens,
            cache_dir=cache_dir,
            question_roles_dir=question_roles_dir,
        ),
        clip_profiles,
    )


def _semantic_redundancy(cid, selected, clip_repr_map):
    curr_repr = clip_repr_map.get(cid)
    if curr_repr is None or not selected:
        return 0.0
    sims = [
        float(np.dot(curr_repr, clip_repr_map[sel]))
        for sel in selected
        if clip_repr_map.get(sel) is not None
    ]
    return max(sims) if sims else 0.0


def _role_aware_select_clips(
    video_graph,
    role_query,
    candidate_ids,
    clip_scores,
    clip_repr_map,
    clip_node_hits,
    *,
    topk=2,
    model="local-qwen3-vl",
    model_device=None,
    max_new_tokens=2048,
    cache_dir=None,
    precomputed_dir=None,
    max_nodes_per_clip=4,
    question_role_mode="heuristic",
    question_roles_dir=None,
    role_match_weight=0.0,
    relevance_weight=0.55,
    coverage_weight=0.30,
    instance_weight=0.25,
    role_redundancy_weight=0.0,
    semantic_redundancy_weight=0.10,
    secondary_role_weight=0.0,
    soft_query_prior=False,
    query_coverage_weight=0.0,
    gate_semantic_redundancy=False,
    fix_first_relevance=False,
):
    if not candidate_ids:
        return []
    if len(candidate_ids) <= int(topk):
        return list(candidate_ids)[: int(topk)]
    # Kept as a compatibility argument for older job scripts. We no longer
    # penalize role overlap because repeated roles can be complementary evidence.
    _ = role_redundancy_weight

    precomputed = None
    if precomputed_dir:
        precomputed = _get_precomputed_role_profiles(
            video_graph,
            role_query,
            candidate_ids,
            precomputed_dir,
            question_role_mode=question_role_mode,
            model=model,
            model_device=model_device,
            max_new_tokens=min(max_new_tokens, 512),
            cache_dir=cache_dir,
            question_roles_dir=question_roles_dir,
        )
    if precomputed is not None:
        question_role_weights, clip_profiles = precomputed
    else:
        question_role_weights, clip_profiles = _get_role_profiles(
            video_graph,
            role_query,
            candidate_ids,
            clip_node_hits,
            model=model,
            model_device=model_device,
            max_new_tokens=max_new_tokens,
            cache_dir=cache_dir,
            max_nodes_per_clip=max_nodes_per_clip,
        )
        if str(question_role_mode or "heuristic").strip().lower() not in {"heuristic", "rules", "rule"}:
            question_role_weights = _get_question_role_weights(
                role_query,
                mode=question_role_mode,
                model=model,
                model_device=model_device,
                max_new_tokens=min(max_new_tokens, 512),
                cache_dir=cache_dir,
                question_roles_dir=question_roles_dir,
            )
    if not question_role_weights:
        return _mmr_select_clips(candidate_ids, clip_scores, clip_repr_map, topk=topk)

    raw_scores = [float(clip_scores.get(cid, 0.0)) for cid in candidate_ids]
    s_min = min(raw_scores) if raw_scores else 0.0
    s_max = max(raw_scores) if raw_scores else 0.0
    if s_max > s_min:
        rel_scores = {cid: (float(clip_scores.get(cid, 0.0)) - s_min) / (s_max - s_min) for cid in candidate_ids}
    else:
        rel_scores = {cid: 1.0 for cid in candidate_ids}

    secondary_role_weight = max(0.0, float(secondary_role_weight))
    effective_role_weights = {
        role: float(weight)
        for role, weight in question_role_weights.items()
        if float(weight) > 0.0
    }
    if secondary_role_weight > 0.0:
        for role in EVIDENCE_ROLE_DEFINITIONS:
            effective_role_weights.setdefault(role, secondary_role_weight)

    soft_query_prior = bool(soft_query_prior)
    query_role_weights = {
        role: float(weight)
        for role, weight in question_role_weights.items()
        if float(weight) > 0.0 and role in EVIDENCE_ROLE_DEFINITIONS
    }
    total_query_role_weight = max(sum(query_role_weights.values()), 1e-6)
    all_roles = set(EVIDENCE_ROLE_DEFINITIONS)

    selected = []
    remaining = list(candidate_ids)
    covered_roles = set()
    covered_units = set()
    total_role_weight = max(sum(effective_role_weights.values()), 1e-6)
    max_candidate_unit_weight = max(
        [
            sum(
                effective_role_weights.get(profile.get("unit_roles", {}).get(unit), 0.0)
                for unit in profile.get("units", set())
            )
            for profile in clip_profiles.values()
        ] + [1e-6]
    )
    max_candidate_units_all_roles = max(
        [
            sum(
                1
                for unit in profile.get("units", set())
                if profile.get("unit_roles", {}).get(unit) in all_roles
            )
            for profile in clip_profiles.values()
        ] + [1]
    )

    if fix_first_relevance and remaining and int(topk) > 0:
        first_id = max(remaining, key=lambda cid: rel_scores.get(cid, 0.0))
        selected.append(first_id)
        remaining.remove(first_id)
        first_profile = clip_profiles.get(first_id, {})
        covered_roles.update(first_profile.get("roles", set()))
        covered_units.update(first_profile.get("units", set()))

    while remaining and len(selected) < int(topk):
        best_id = None
        best_score = -1e9

        for cid in remaining:
            profile = clip_profiles.get(cid, {})
            roles = set(profile.get("roles", set()))
            units = set(profile.get("units", set()))
            if soft_query_prior:
                valid_roles = {role for role in roles if role in all_roles}
                new_roles = valid_roles - covered_roles
                role_gain = len(new_roles) / max(1, len(all_roles))

                valid_units = {
                    unit
                    for unit in units
                    if profile.get("unit_roles", {}).get(unit) in all_roles
                }
                instance_gain = min(
                    1.0,
                    len(valid_units - covered_units) / max(1, max_candidate_units_all_roles),
                )

                matched_query_roles = {
                    role
                    for role in valid_roles
                    if query_role_weights.get(role, 0.0) > 0
                }
                role_match = min(
                    1.0,
                    sum(query_role_weights.get(role, 0.0) for role in matched_query_roles)
                    / total_query_role_weight,
                )
                query_new_roles = matched_query_roles - covered_roles
                query_role_gain = min(
                    1.0,
                    sum(query_role_weights.get(role, 0.0) for role in query_new_roles)
                    / total_query_role_weight,
                )
            else:
                matched_roles = {role for role in roles if effective_role_weights.get(role, 0.0) > 0}
                role_match = sum(effective_role_weights.get(role, 0.0) for role in matched_roles) / total_role_weight
                role_match = min(1.0, role_match)

                new_roles = matched_roles - covered_roles
                role_gain = sum(effective_role_weights.get(role, 0.0) for role in new_roles) / total_role_weight

                new_unit_weight = sum(
                    effective_role_weights.get(profile.get("unit_roles", {}).get(unit), 0.0)
                    for unit in units - covered_units
                )
                instance_gain = min(1.0, new_unit_weight / max_candidate_unit_weight)
                query_role_gain = 0.0

            semantic_redundancy = _semantic_redundancy(cid, selected, clip_repr_map)
            if gate_semantic_redundancy:
                semantic_redundancy *= max(0.0, 1.0 - instance_gain)
            objective = (
                relevance_weight * rel_scores.get(cid, 0.0)
                + role_match_weight * role_match
                + coverage_weight * role_gain
                + instance_weight * instance_gain
                + query_coverage_weight * query_role_gain
                - semantic_redundancy_weight * semantic_redundancy
            )
            if objective > best_score:
                best_score = objective
                best_id = cid

        if best_id is None:
            break
        selected.append(best_id)
        remaining.remove(best_id)
        profile = clip_profiles.get(best_id, {})
        covered_roles.update(profile.get("roles", set()))
        covered_units.update(profile.get("units", set()))

    return selected


def _score_clips_with_diversity(
    video_graph,
    clip_node_hits,
    *,
    max_nodes_per_clip=8,
    intra_clip_sim_threshold=0.85,
):
    clip_scores = {}
    clip_repr_map = {}
    for clip_id, node_hits in clip_node_hits.items():
        clusters = _group_clip_nodes(
            video_graph,
            node_hits,
            max_nodes=max_nodes_per_clip,
            sim_threshold=intra_clip_sim_threshold,
        )
        clip_scores[clip_id] = _compute_clip_base_score(clusters)
        clip_repr_map[clip_id] = _build_clip_representation(clusters)
    return clip_scores, clip_repr_map

def translate(video_graph, memories):
    new_memories = []
    for memory in memories:
        if memory.lower().startswith("equivalence: "):
            continue
        new_memory = memory
        entities = parse_video_caption(video_graph, memory)
        entities = list(set(entities))
        for entity in entities:
            entity_str = f"{entity[0]}_{entity[1]}"
            if entity_str in video_graph.reverse_character_mappings.keys():
                new_memory = new_memory.replace(entity_str, video_graph.reverse_character_mappings[entity_str])
        new_memories.append(new_memory)
    return new_memories

def back_translate(video_graph, queries):
    translated_queries = []
    for query in queries:
        entities = parse_video_caption(video_graph, query)
        entities = list(set(entities))
        to_be_translated = [query]
        for entity in entities:
            entity_str = f"{entity[0]}_{entity[1]}"
            if entity_str in video_graph.character_mappings.keys():
                mappings = video_graph.character_mappings[entity_str]

                # Create new queries for each mapping
                new_queries = []
                for mapping in mappings:
                    for partially_translated in to_be_translated:
                        new_query = partially_translated.replace(entity_str, mapping)
                        new_queries.append(new_query)

                # Update translated_query with all variants
                to_be_translated = new_queries

        # Add all variants of the translated query
        translated_queries.extend(to_be_translated)
    return translated_queries

# retrieve by clip
def retrieve_from_videograph(
    video_graph,
    query,
    topk=5,
    mode='max',
    threshold=float("-inf"),
    before_clip=None,
    speaker_nodes=None,
    speaker_bias=0.0,
    speaker_hard_filter=False,
    scene_nodes=None,
    scene_rerank_weight=0.14,
    diverse_clip_retrieval=False,
    diverse_clip_pool_size=12,
    diverse_clip_mmr_candidate_pool_size=200,
    clip_intra_similarity_threshold=0.85,
    clip_mmr_lambda=0.75,
    clip_max_nodes_for_diversity=8,
    dynamic_mmr_clip_retrieval=False,
    dynamic_mmr_min_clips=2,
    dynamic_mmr_max_clips=5,
    dynamic_mmr_stop_threshold=0.05,
    dynamic_mmr_extra_clips=0,
    dynamic_mmr_trace_path=None,
    dynamic_mmr_log_scores=False,
    dynamic_mmr_policy="threshold",
    dynamic_mmr_confidence_threshold=0.30,
    dynamic_mmr_ambiguity_gap_threshold=0.25,
    dynamic_mmr_knee_min_drop=0.25,
    dynamic_mmr_knee_alpha=1.0,
    dynamic_mmr_uncertainty_alpha=1.0,
    dynamic_mmr_score_source="clip_score",
    full_adaptive_k_retrieval=False,
    full_adaptive_k_strategy="largest_gap",
    full_adaptive_k_ignore_extreme=0.0,
    full_adaptive_k_ignore_extreme_tail=0.1,
    full_adaptive_k_ignore_below_median=False,
    full_adaptive_k_retrieve_more=5,
    full_adaptive_k_candidate_nodes=None,
    full_adaptive_k_min_nodes=1,
    full_adaptive_k_max_nodes=None,
    full_adaptive_k_min_clips=None,
    full_adaptive_k_max_clips=None,
    full_adaptive_k_extra_clips=0,
    clip_adaptive_k_retrieval=False,
    clip_adaptive_k_strategy="largest_gap",
    clip_adaptive_k_ignore_extreme=0.0,
    clip_adaptive_k_ignore_extreme_tail=0.1,
    clip_adaptive_k_ignore_below_median=False,
    clip_adaptive_k_retrieve_more=5,
    clip_adaptive_k_min_clips=1,
    clip_adaptive_k_max_clips=None,
    clip_adaptive_k_extra_clips=0,
    clip_adaptive_k_score_source="max_node",
    dynamicrag_clip_retrieval=False,
    dynamicrag_model="gasolsun/DynamicRAG-8B",
    dynamicrag_api_base=None,
    dynamicrag_api_key="EMPTY",
    dynamicrag_temperature=0.4,
    dynamicrag_max_tokens=100,
    dynamicrag_timeout=60,
    dynamicrag_min_clips=0,
    dynamicrag_max_clips=None,
    dynamicrag_max_nodes_per_clip=4,
    dynamicrag_max_doc_chars=1600,
    df_rag_clip_retrieval=False,
    df_rag_model="models/DynamicRAG-8B",
    df_rag_api_base=None,
    df_rag_api_key="EMPTY",
    df_rag_temperature=0.0,
    df_rag_planner_max_tokens=256,
    df_rag_evaluator_max_tokens=512,
    df_rag_timeout=60,
    df_rag_lambdas="0.1,0.2,0.3,0.4,0.5,0.6,0.7,0.8,0.9,1.0",
    df_rag_set_size=5,
    df_rag_max_nodes_per_clip=4,
    df_rag_max_doc_chars=1600,
    df_rag_fallback_lambda=0.5,
    adaptive_rag_retrieval=False,
    adaptive_rag_route_source="heuristic",
    adaptive_rag_classifier_path=None,
    adaptive_rag_classifier_model="adaptive-rag-classifier",
    adaptive_rag_classifier_api_base=None,
    adaptive_rag_classifier_api_key="EMPTY",
    adaptive_rag_classifier_temperature=0.0,
    adaptive_rag_classifier_max_tokens=4,
    adaptive_rag_classifier_timeout=60,
    adaptive_rag_fallback_label="B",
    adaptive_rag_question=None,
    adaptive_rag_question_id=None,
    adaptive_rag_zero_clips=0,
    adaptive_rag_single_clips=2,
    adaptive_rag_multi_clips=5,
    adaptive_rag_selector="top",
    adaptive_rag_score_source="max_node",
    evidence_saturation_retrieval=False,
    evidence_saturation_min_clips=1,
    evidence_saturation_max_clips=5,
    evidence_saturation_stop_threshold=0.02,
    evidence_saturation_relevance_weight=0.75,
    evidence_saturation_semantic_gain_weight=0.10,
    evidence_saturation_temporal_gain_weight=0.08,
    evidence_saturation_entity_gain_weight=0.06,
    evidence_saturation_action_state_gain_weight=0.06,
    evidence_saturation_semantic_redundancy_weight=0.15,
    evidence_saturation_temporal_redundancy_weight=0.0,
    evidence_saturation_temporal_bucket_size=4,
    evidence_saturation_near_clip_window=2,
    excluded_clips=None,
    role_aware_clip_retrieval=False,
    role_aware_question=None,
    role_aware_model="local-qwen3-vl",
    role_aware_model_device=None,
    role_aware_max_new_tokens=2048,
    role_aware_cache_dir=None,
    role_aware_precomputed_dir=None,
    role_aware_max_nodes_per_clip=4,
    role_aware_question_role_mode="heuristic",
    role_aware_question_roles_dir=None,
    role_aware_role_match_weight=0.0,
    role_aware_relevance_weight=0.55,
    role_aware_coverage_weight=0.30,
    role_aware_instance_weight=0.25,
    role_aware_role_redundancy_weight=0.0,
    role_aware_semantic_redundancy_weight=0.10,
    role_aware_secondary_role_weight=0.0,
    role_aware_soft_query_prior=False,
    role_aware_query_coverage_weight=0.0,
    role_aware_gate_semantic_redundancy=False,
    role_aware_fix_first_relevance=False,
    fixed_clip_backfill_current=False,
):
    top_clips = []
    # find all CLIP_x in query
    pattern = r"CLIP_(\d+)"
    matches = re.finditer(pattern, query)
    top_clips = []
    for match in matches:
        try:
            clip_id = int(match.group(1))
            top_clips.append(clip_id)
        except ValueError:
            continue

    queries = back_translate(video_graph, [query])
    if len(queries) > 100:
        logger.error(f"Anomaly detected from query: {query}, randomly sample 100 translatedqueries")
        queries = random.sample(queries, 100)

    related_nodes = get_related_nodes(video_graph, query)

    model = "text-embedding-3-large"
    query_embeddings = parallel_get_embedding(model, queries)[0]

    clip_node_hits = defaultdict(list)
    clip_scores = {}
    clip_repr_map = {}
    excluded_clips = set(excluded_clips or [])

    if mode not in ['sum', 'max', 'mean']:
        raise ValueError(f"Unknown mode: {mode}")

    # calculate scores for each node
    nodes = video_graph.search_text_nodes(query_embeddings, related_nodes, mode='max')
    nodes = _apply_speaker_bias(
        video_graph,
        nodes,
        speaker_nodes=speaker_nodes or set(),
        speaker_bias=speaker_bias,
        speaker_hard_filter=speaker_hard_filter,
    )

    if full_adaptive_k_retrieval:
        full_trace = None
        if dynamic_mmr_trace_path or dynamic_mmr_log_scores:
            full_trace = {
                "query": query,
                "policy": "full_official_adaptive_k",
                "strategy": str(full_adaptive_k_strategy),
                "ignore_extreme": float(full_adaptive_k_ignore_extreme),
                "ignore_extreme_tail": float(full_adaptive_k_ignore_extreme_tail),
                "ignore_below_median": bool(full_adaptive_k_ignore_below_median),
                "retrieve_more": full_adaptive_k_retrieve_more,
                "candidate_nodes": None if full_adaptive_k_candidate_nodes is None else int(full_adaptive_k_candidate_nodes),
                "min_nodes": int(full_adaptive_k_min_nodes),
                "max_nodes": None if full_adaptive_k_max_nodes is None else int(full_adaptive_k_max_nodes),
                "min_clips": None if full_adaptive_k_min_clips is None else int(full_adaptive_k_min_clips),
                "max_clips": None if full_adaptive_k_max_clips is None else int(full_adaptive_k_max_clips),
                "extra_clips": int(full_adaptive_k_extra_clips),
            }
        top_clips, clip_node_hits, full_decision = _full_adaptive_k_select_from_nodes(
            video_graph,
            nodes,
            before_clip=before_clip,
            excluded_clips=excluded_clips,
            strategy=full_adaptive_k_strategy,
            ignore_extreme=full_adaptive_k_ignore_extreme,
            ignore_extreme_tail=full_adaptive_k_ignore_extreme_tail,
            ignore_below_median=full_adaptive_k_ignore_below_median,
            retrieve_more=full_adaptive_k_retrieve_more,
            candidate_nodes=full_adaptive_k_candidate_nodes,
            min_nodes=full_adaptive_k_min_nodes,
            max_nodes=full_adaptive_k_max_nodes,
            min_clips=full_adaptive_k_min_clips,
            max_clips=full_adaptive_k_max_clips,
            extra_clips=full_adaptive_k_extra_clips,
            trace=full_trace,
        )
        for clip_id, node_hits in clip_node_hits.items():
            scores = [score for _, score in node_hits]
            if mode == 'sum':
                clip_scores[clip_id] = sum(scores)
            elif mode == 'mean':
                clip_scores[clip_id] = np.mean(scores)
            else:
                clip_scores[clip_id] = max(scores)
        if dynamic_mmr_log_scores:
            print(
                "[full_adaptive_k] "
                f"strategy={full_adaptive_k_strategy} "
                f"nodes={full_decision.get('selected_node_count', 0)}/"
                f"{full_decision.get('candidate_node_count', 0)} "
                f"clips={top_clips}",
                flush=True,
            )
        if dynamic_mmr_trace_path and full_trace is not None:
            try:
                os.makedirs(os.path.dirname(dynamic_mmr_trace_path), exist_ok=True)
                with open(dynamic_mmr_trace_path, "a", encoding="utf-8") as f:
                    f.write(json.dumps(full_trace, ensure_ascii=False) + "\n")
            except Exception as exc:
                logger.warning("Failed to write full Adaptive-k trace to %s: %s", dynamic_mmr_trace_path, exc)
        return top_clips, clip_scores, nodes


    # collect node scores for each clip
    for node_id, node_score in nodes:
        clip_id = video_graph.nodes[node_id].metadata['timestamp']
        clip_node_hits[clip_id].append((node_id, float(node_score)))

    if diverse_clip_retrieval:
        clip_scores, clip_repr_map = _score_clips_with_diversity(
            video_graph,
            clip_node_hits,
            max_nodes_per_clip=clip_max_nodes_for_diversity,
            intra_clip_sim_threshold=clip_intra_similarity_threshold,
        )
    else:
        # calculate scores for each clip
        for clip_id, node_hits in clip_node_hits.items():
            scores = [score for _, score in node_hits]
            if mode == 'sum':
                clip_score = sum(scores)
            elif mode == 'max':
                clip_score = max(scores)
            elif mode == 'mean':
                clip_score = np.mean(scores)
            else:
                raise ValueError(f"Unknown mode: {mode}")
            clip_scores[clip_id] = clip_score

    # Scene Node reranking: boost clips near relevant scene nodes
    if scene_nodes and clip_scores:
        scene_embeddings = np.array([emb for emb, _ in scene_nodes])
        scene_clips = [cid for _, cid in scene_nodes]
        query_emb_mean = np.mean(np.array(query_embeddings), axis=0, keepdims=True)
        scene_sims = cosine_similarity(query_emb_mean, scene_embeddings)[0]
        for sim, scene_clip in zip(scene_sims, scene_clips):
            if sim > 0.3:
                for offset in range(-5, 1):  # scene covers ~6 clips
                    c = scene_clip + offset
                    if c in clip_scores:
                        clip_scores[c] += scene_rerank_weight * sim

    # sort clips by score
    sorted_clips = sorted(clip_scores.items(), key=lambda x: x[1], reverse=True)

    if diverse_clip_retrieval:
        candidate_ids = []
        pool_size = max(int(diverse_clip_pool_size), int(topk))
        mmr_candidate_pool_size = int(diverse_clip_mmr_candidate_pool_size or 0)
        if mmr_candidate_pool_size <= 0:
            mmr_candidate_pool_size = pool_size
        mmr_candidate_pool_size = max(pool_size, mmr_candidate_pool_size)
        clip_adaptive_score_source = str(clip_adaptive_k_score_source or "max_node")
        adaptive_rag_score_source = str(adaptive_rag_score_source or "max_node")
        dynamic_mmr_score_source = str(dynamic_mmr_score_source or "clip_score")
        clip_adaptive_scores = clip_scores
        adaptive_rag_scores = clip_scores
        dynamic_mmr_scores = clip_scores
        use_max_node_candidates = (
            (clip_adaptive_k_retrieval and clip_adaptive_score_source == "max_node")
            or (adaptive_rag_retrieval and adaptive_rag_score_source == "max_node")
            or (dynamic_mmr_clip_retrieval and dynamic_mmr_score_source == "max_node")
        )
        if use_max_node_candidates:
            max_node_clip_scores = {}
            seen_candidate_clips = set()
            for node_id, node_score in sorted(nodes, key=lambda x: x[1], reverse=True):
                try:
                    clip_id = int(video_graph.nodes[node_id].metadata["timestamp"])
                except Exception:
                    continue
                if before_clip is not None and clip_id > before_clip:
                    continue
                if clip_id in excluded_clips:
                    continue
                if clip_id in seen_candidate_clips:
                    continue
                seen_candidate_clips.add(clip_id)
                candidate_ids.append(clip_id)
                max_node_clip_scores[clip_id] = float(node_score)
                if len(candidate_ids) >= mmr_candidate_pool_size:
                    break
            if clip_adaptive_k_retrieval and clip_adaptive_score_source == "max_node":
                clip_adaptive_scores = max_node_clip_scores
            if adaptive_rag_retrieval and adaptive_rag_score_source == "max_node":
                adaptive_rag_scores = max_node_clip_scores
            if dynamic_mmr_clip_retrieval and dynamic_mmr_score_source == "max_node":
                dynamic_mmr_scores = max_node_clip_scores
        else:
            for clip_id, score in sorted_clips:
                if not clip_adaptive_k_retrieval and score < threshold:
                    continue
                if before_clip is not None and clip_id > before_clip:
                    continue
                if clip_id in excluded_clips:
                    continue
                candidate_ids.append(clip_id)
                if len(candidate_ids) >= mmr_candidate_pool_size:
                    break
        dynamic_mmr_trace = None
        if adaptive_rag_retrieval:
            adaptive_rag_trace = None
            if dynamic_mmr_trace_path or dynamic_mmr_log_scores:
                adaptive_rag_trace = {
                    "query": query,
                    "question": adaptive_rag_question,
                    "question_id": adaptive_rag_question_id,
                    "policy": "adaptive_rag_complexity_route",
                    "candidate_ids": [int(cid) for cid in candidate_ids],
                    "clip_scores": {str(int(cid)): float(adaptive_rag_scores.get(cid, 0.0)) for cid in candidate_ids},
                    "pool_size": int(pool_size),
                    "mmr_candidate_pool_size": int(mmr_candidate_pool_size),
                    "candidate_count": int(len(candidate_ids)),
                    "score_source": str(adaptive_rag_score_source),
                    "route_source": str(adaptive_rag_route_source),
                    "zero_clips": int(adaptive_rag_zero_clips),
                    "single_clips": int(adaptive_rag_single_clips),
                    "multi_clips": int(adaptive_rag_multi_clips),
                    "selector": str(adaptive_rag_selector),
                    "source_repo": "https://github.com/starsuzi/Adaptive-RAG",
                    "source_paper": "https://arxiv.org/abs/2403.14403",
                }
            adaptive_label, adaptive_meta = _adaptive_rag_predict_label(
                query=query,
                question=adaptive_rag_question,
                question_id=adaptive_rag_question_id,
                route_source=adaptive_rag_route_source,
                classifier_path=adaptive_rag_classifier_path,
                classifier_model=adaptive_rag_classifier_model,
                classifier_api_base=adaptive_rag_classifier_api_base,
                classifier_api_key=adaptive_rag_classifier_api_key,
                classifier_temperature=adaptive_rag_classifier_temperature,
                classifier_max_tokens=adaptive_rag_classifier_max_tokens,
                classifier_timeout=adaptive_rag_classifier_timeout,
                fallback_label=adaptive_rag_fallback_label,
            )
            top_clips = _adaptive_rag_select_clips(
                candidate_ids,
                adaptive_rag_scores,
                clip_repr_map,
                label=adaptive_label,
                topk=topk,
                zero_clips=adaptive_rag_zero_clips,
                single_clips=adaptive_rag_single_clips,
                multi_clips=adaptive_rag_multi_clips,
                mmr_lambda=clip_mmr_lambda,
                selector=adaptive_rag_selector,
            )
            if adaptive_rag_trace is not None:
                adaptive_rag_trace.update(
                    {
                        "route_label": adaptive_label,
                        "route_description": _adaptive_rag_label_description(adaptive_label),
                        "route_meta": adaptive_meta,
                        "selected": [int(cid) for cid in top_clips],
                        "selected_count": int(len(top_clips)),
                    }
                )
            if dynamic_mmr_log_scores:
                score_parts = [
                    f"r{idx + 1}:clip={int(cid)},score={float(adaptive_rag_scores.get(cid, 0.0)):.3f}"
                    for idx, cid in enumerate(candidate_ids[:12])
                ]
                print(
                    "[adaptive_rag] "
                    f"route={adaptive_label}({_adaptive_rag_label_description(adaptive_label)}) "
                    f"source={adaptive_meta.get('source')} "
                    f"score_source={adaptive_rag_score_source} "
                    f"selected={top_clips} candidates=[{'; '.join(score_parts)}]",
                    flush=True,
                )
            if dynamic_mmr_trace_path and adaptive_rag_trace is not None:
                try:
                    os.makedirs(os.path.dirname(dynamic_mmr_trace_path), exist_ok=True)
                    with open(dynamic_mmr_trace_path, "a", encoding="utf-8") as f:
                        f.write(json.dumps(adaptive_rag_trace, ensure_ascii=False) + "\n")
                except Exception as exc:
                    logger.warning("Failed to write Adaptive-RAG trace to %s: %s", dynamic_mmr_trace_path, exc)
        elif dynamic_mmr_clip_retrieval:
            if dynamic_mmr_trace_path or dynamic_mmr_log_scores:
                dynamic_mmr_trace = {
                    "query": query,
                    "question": adaptive_rag_question,
                    "question_id": adaptive_rag_question_id,
                    "candidate_ids": [int(cid) for cid in candidate_ids],
                    "clip_scores": {str(int(cid)): float(dynamic_mmr_scores.get(cid, 0.0)) for cid in candidate_ids},
                    "score_source": str(dynamic_mmr_score_source),
                    "pool_size": int(pool_size),
                    "mmr_candidate_pool_size": int(mmr_candidate_pool_size),
                    "candidate_count": int(len(candidate_ids)),
                    "mmr_lambda": float(clip_mmr_lambda),
                    "min_clips": int(dynamic_mmr_min_clips),
                    "max_clips": int(dynamic_mmr_max_clips),
                    "stop_threshold": float(dynamic_mmr_stop_threshold),
                    "extra_clips": int(dynamic_mmr_extra_clips),
                    "policy": str(dynamic_mmr_policy),
                    "confidence_threshold": float(dynamic_mmr_confidence_threshold),
                    "ambiguity_gap_threshold": float(dynamic_mmr_ambiguity_gap_threshold),
                    "knee_min_drop": float(dynamic_mmr_knee_min_drop),
                    "knee_alpha": float(dynamic_mmr_knee_alpha),
                    "uncertainty_alpha": float(dynamic_mmr_uncertainty_alpha),
                }
            top_clips = _dynamic_mmr_select_clips(
                candidate_ids,
                dynamic_mmr_scores,
                clip_repr_map,
                min_clips=dynamic_mmr_min_clips,
                max_clips=dynamic_mmr_max_clips,
                mmr_lambda=clip_mmr_lambda,
                stop_threshold=dynamic_mmr_stop_threshold,
                policy=dynamic_mmr_policy,
                confidence_threshold=dynamic_mmr_confidence_threshold,
                ambiguity_gap_threshold=dynamic_mmr_ambiguity_gap_threshold,
                knee_min_drop=dynamic_mmr_knee_min_drop,
                knee_alpha=dynamic_mmr_knee_alpha,
                uncertainty_alpha=dynamic_mmr_uncertainty_alpha,
                trace=dynamic_mmr_trace,
            )
            extra_clips = max(0, int(dynamic_mmr_extra_clips))
            if extra_clips > 0 and len(top_clips) < int(dynamic_mmr_max_clips):
                selected_set = set(top_clips)
                ranked_extra_candidates = []
                if dynamic_mmr_trace is not None:
                    ranked_steps = [
                        step
                        for step in dynamic_mmr_trace.get("steps", [])
                        if step.get("event") in {"select", "candidate"} and "rank" in step
                    ]
                    ranked_steps.sort(key=lambda step: int(step.get("rank", 0)))
                    ranked_extra_candidates = [step.get("clip") for step in ranked_steps]
                if not ranked_extra_candidates:
                    ranked_extra_candidates = list(candidate_ids)
                added = []
                for clip_id in ranked_extra_candidates:
                    if clip_id in selected_set:
                        continue
                    top_clips.append(clip_id)
                    selected_set.add(clip_id)
                    added.append(clip_id)
                    if len(added) >= extra_clips or len(top_clips) >= int(dynamic_mmr_max_clips):
                        break
                if dynamic_mmr_trace is not None:
                    dynamic_mmr_trace.setdefault("steps", []).append(
                        {
                            "event": "extra_extend",
                            "reason": "dynamic_mmr_extra_clips",
                            "extra_clips": int(extra_clips),
                            "added": [int(cid) for cid in added],
                            "selected_count": int(len(top_clips)),
                            "selected_so_far": [int(cid) for cid in top_clips],
                        }
                    )
            if dynamic_mmr_log_scores and dynamic_mmr_trace is not None:
                selected_steps = [
                    step for step in dynamic_mmr_trace.get("steps", [])
                    if step.get("event") == "select"
                ]
                stop_steps = [
                    step for step in dynamic_mmr_trace.get("steps", [])
                    if step.get("event") == "stop"
                ]
                score_parts = [
                    (
                        f"r{int(step.get('rank', 0))}:clip={int(step.get('clip', -1))},"
                        f"mmr={float(step.get('mmr_score', 0.0)):.3f},"
                        f"rel={float(step.get('relevance', 0.0)):.3f},"
                        f"red={float(step.get('redundancy', 0.0)):.3f}"
                    )
                    for step in selected_steps
                ]
                stop_msg = ""
                if stop_steps:
                    stop = stop_steps[-1]
                    stop_msg = (
                        f" stop={stop.get('reason')} "
                        f"next_clip={stop.get('candidate_clip')} "
                        f"next_mmr={float(stop.get('mmr_score', 0.0)):.3f}"
                    )
                print(
                    "[dynamic_mmr] "
                    f"selected={top_clips} scores=[{'; '.join(score_parts)}]"
                    f"{stop_msg}",
                    flush=True,
                )
            if dynamic_mmr_trace_path and dynamic_mmr_trace is not None:
                dynamic_mmr_trace["selected"] = [int(cid) for cid in top_clips]
                dynamic_mmr_trace["selected_count"] = len(top_clips)
                try:
                    os.makedirs(os.path.dirname(dynamic_mmr_trace_path), exist_ok=True)
                    with open(dynamic_mmr_trace_path, "a", encoding="utf-8") as f:
                        f.write(json.dumps(dynamic_mmr_trace, ensure_ascii=False) + "\n")
                except Exception as exc:
                    logger.warning("Failed to write dynamic MMR trace to %s: %s", dynamic_mmr_trace_path, exc)
        elif clip_adaptive_k_retrieval:
            clip_adaptive_trace = None
            if dynamic_mmr_trace_path or dynamic_mmr_log_scores:
                clip_adaptive_trace = {
                    "query": query,
                    "policy": "top_candidate_clip_official_adaptive_k",
                    "candidate_ids": [int(cid) for cid in candidate_ids],
                    "clip_scores": {str(int(cid)): float(clip_adaptive_scores.get(cid, 0.0)) for cid in candidate_ids},
                    "pool_size": int(pool_size),
                    "mmr_candidate_pool_size": int(mmr_candidate_pool_size),
                    "candidate_count": int(len(candidate_ids)),
                    "score_source": clip_adaptive_score_source,
                    "strategy": str(clip_adaptive_k_strategy),
                    "ignore_extreme": float(clip_adaptive_k_ignore_extreme),
                    "ignore_extreme_tail": float(clip_adaptive_k_ignore_extreme_tail),
                    "ignore_below_median": bool(clip_adaptive_k_ignore_below_median),
                    "retrieve_more": clip_adaptive_k_retrieve_more,
                    "min_clips": None if clip_adaptive_k_min_clips is None else int(clip_adaptive_k_min_clips),
                    "max_clips": None if clip_adaptive_k_max_clips is None else int(clip_adaptive_k_max_clips),
                    "extra_clips": int(clip_adaptive_k_extra_clips),
                    "pool_size": int(pool_size),
                }
            top_clips, clip_adaptive_decision = _clip_adaptive_k_select_from_scores(
                candidate_ids,
                clip_adaptive_scores,
                strategy=clip_adaptive_k_strategy,
                ignore_extreme=clip_adaptive_k_ignore_extreme,
                ignore_extreme_tail=clip_adaptive_k_ignore_extreme_tail,
                ignore_below_median=clip_adaptive_k_ignore_below_median,
                retrieve_more=clip_adaptive_k_retrieve_more,
                min_clips=clip_adaptive_k_min_clips,
                max_clips=clip_adaptive_k_max_clips,
                extra_clips=clip_adaptive_k_extra_clips,
                trace=clip_adaptive_trace,
            )
            if dynamic_mmr_log_scores:
                score_parts = [
                    f"r{idx + 1}:clip={int(cid)},score={float(clip_adaptive_scores.get(cid, 0.0)):.3f}"
                    for idx, cid in enumerate(candidate_ids[:12])
                ]
                print(
                    "[clip_adaptive_k] "
                    f"strategy={clip_adaptive_k_strategy} "
                    f"source={clip_adaptive_score_source} "
                    f"official_count={clip_adaptive_decision.get('official_selected_count')} "
                    f"selected={top_clips} candidates=[{'; '.join(score_parts)}]",
                    flush=True,
                )
            if dynamic_mmr_trace_path and clip_adaptive_trace is not None:
                try:
                    os.makedirs(os.path.dirname(dynamic_mmr_trace_path), exist_ok=True)
                    with open(dynamic_mmr_trace_path, "a", encoding="utf-8") as f:
                        f.write(json.dumps(clip_adaptive_trace, ensure_ascii=False) + "\n")
                except Exception as exc:
                    logger.warning("Failed to write clip Adaptive-k trace to %s: %s", dynamic_mmr_trace_path, exc)
        elif dynamicrag_clip_retrieval:
            dynamicrag_trace = None
            if dynamic_mmr_trace_path or dynamic_mmr_log_scores:
                dynamicrag_trace = {
                    "query": query,
                    "policy": "dynamicrag_clip_selector",
                    "candidate_ids": [int(cid) for cid in candidate_ids],
                    "clip_scores": {str(int(cid)): float(clip_scores.get(cid, 0.0)) for cid in candidate_ids},
                    "pool_size": int(pool_size),
                    "min_clips": int(dynamicrag_min_clips),
                    "max_clips": None if dynamicrag_max_clips is None else int(dynamicrag_max_clips),
                    "model": str(dynamicrag_model),
                    "api_base": str(dynamicrag_api_base),
                    "temperature": float(dynamicrag_temperature),
                    "max_tokens": int(dynamicrag_max_tokens),
                }
            top_clips, dynamicrag_decision = _dynamicrag_select_clips(
                video_graph,
                query,
                candidate_ids,
                clip_node_hits,
                clip_scores,
                model=dynamicrag_model,
                api_base=dynamicrag_api_base,
                api_key=dynamicrag_api_key,
                temperature=dynamicrag_temperature,
                max_tokens=dynamicrag_max_tokens,
                timeout=dynamicrag_timeout,
                min_clips=dynamicrag_min_clips,
                max_clips=dynamicrag_max_clips,
                max_nodes_per_clip=dynamicrag_max_nodes_per_clip,
                max_doc_chars=dynamicrag_max_doc_chars,
                trace=dynamicrag_trace,
            )
            if dynamic_mmr_log_scores:
                score_parts = [
                    f"r{idx + 1}:clip={int(cid)},score={float(clip_scores.get(cid, 0.0)):.3f}"
                    for idx, cid in enumerate(candidate_ids[:12])
                ]
                print(
                    "[dynamicrag] "
                    f"selected={top_clips} "
                    f"doc_ids={dynamicrag_decision.get('selected_doc_indices')} "
                    f"raw={dynamicrag_decision.get('raw_output')!r} "
                    f"candidates=[{'; '.join(score_parts)}]",
                    flush=True,
                )
            if dynamic_mmr_trace_path and dynamicrag_trace is not None:
                try:
                    os.makedirs(os.path.dirname(dynamic_mmr_trace_path), exist_ok=True)
                    with open(dynamic_mmr_trace_path, "a", encoding="utf-8") as f:
                        f.write(json.dumps(dynamicrag_trace, ensure_ascii=False) + "\n")
                except Exception as exc:
                    logger.warning("Failed to write DynamicRAG trace to %s: %s", dynamic_mmr_trace_path, exc)
        elif df_rag_clip_retrieval:
            df_rag_trace = None
            if dynamic_mmr_trace_path or dynamic_mmr_log_scores:
                df_rag_trace = {
                    "query": query,
                    "policy": "df_rag_query_aware_diversity",
                    "candidate_ids": [int(cid) for cid in candidate_ids],
                    "clip_scores": {str(int(cid)): float(clip_scores.get(cid, 0.0)) for cid in candidate_ids},
                    "pool_size": int(pool_size),
                    "lambda_grid": str(df_rag_lambdas),
                    "set_size": int(df_rag_set_size),
                    "model": str(df_rag_model),
                    "api_base": str(df_rag_api_base),
                    "temperature": float(df_rag_temperature),
                }
            top_clips, df_rag_decision = _df_rag_select_clips(
                video_graph,
                query,
                candidate_ids,
                clip_node_hits,
                clip_scores,
                clip_repr_map,
                model=df_rag_model,
                api_base=df_rag_api_base,
                api_key=df_rag_api_key,
                temperature=df_rag_temperature,
                planner_max_tokens=df_rag_planner_max_tokens,
                evaluator_max_tokens=df_rag_evaluator_max_tokens,
                timeout=df_rag_timeout,
                lambdas=df_rag_lambdas,
                set_size=df_rag_set_size,
                max_nodes_per_clip=df_rag_max_nodes_per_clip,
                max_doc_chars=df_rag_max_doc_chars,
                fallback_lambda=df_rag_fallback_lambda,
                trace=df_rag_trace,
            )
            if dynamic_mmr_log_scores:
                eval_parts = [
                    f"lambda={float(item.get('lambda', 0.0)):.2f}:score={float(item.get('score', 0.0)):.1f}:clips={item.get('clips')}"
                    for item in df_rag_decision.get("evaluations", [])[:12]
                ]
                print(
                    "[df_rag] "
                    f"chosen_lambda={float(df_rag_decision.get('chosen', {}).get('lambda', 0.0)):.2f} "
                    f"selected={top_clips} evals=[{'; '.join(eval_parts)}]",
                    flush=True,
                )
            if dynamic_mmr_trace_path and df_rag_trace is not None:
                try:
                    os.makedirs(os.path.dirname(dynamic_mmr_trace_path), exist_ok=True)
                    with open(dynamic_mmr_trace_path, "a", encoding="utf-8") as f:
                        f.write(json.dumps(df_rag_trace, ensure_ascii=False) + "\n")
                except Exception as exc:
                    logger.warning("Failed to write DF-RAG trace to %s: %s", dynamic_mmr_trace_path, exc)
        elif evidence_saturation_retrieval:
            top_clips = _evidence_saturation_select_clips(
                video_graph,
                candidate_ids,
                clip_scores,
                clip_repr_map,
                clip_node_hits,
                min_clips=evidence_saturation_min_clips,
                max_clips=evidence_saturation_max_clips,
                stop_threshold=evidence_saturation_stop_threshold,
                relevance_weight=evidence_saturation_relevance_weight,
                semantic_gain_weight=evidence_saturation_semantic_gain_weight,
                temporal_gain_weight=evidence_saturation_temporal_gain_weight,
                entity_gain_weight=evidence_saturation_entity_gain_weight,
                action_state_gain_weight=evidence_saturation_action_state_gain_weight,
                semantic_redundancy_weight=evidence_saturation_semantic_redundancy_weight,
                temporal_redundancy_weight=evidence_saturation_temporal_redundancy_weight,
                temporal_bucket_size=evidence_saturation_temporal_bucket_size,
                near_clip_window=evidence_saturation_near_clip_window,
                max_nodes_per_clip=clip_max_nodes_for_diversity,
                intra_clip_sim_threshold=clip_intra_similarity_threshold,
            )
        elif role_aware_clip_retrieval:
            try:
                top_clips = _role_aware_select_clips(
                    video_graph,
                    role_aware_question or query,
                    candidate_ids,
                    clip_scores,
                    clip_repr_map,
                    clip_node_hits,
                    topk=topk,
                    model=role_aware_model,
                    model_device=role_aware_model_device,
                    max_new_tokens=role_aware_max_new_tokens,
                    cache_dir=role_aware_cache_dir,
                    precomputed_dir=role_aware_precomputed_dir,
                    max_nodes_per_clip=role_aware_max_nodes_per_clip,
                    question_role_mode=role_aware_question_role_mode,
                    question_roles_dir=role_aware_question_roles_dir,
                    role_match_weight=role_aware_role_match_weight,
                    relevance_weight=role_aware_relevance_weight,
                    coverage_weight=role_aware_coverage_weight,
                    instance_weight=role_aware_instance_weight,
                    role_redundancy_weight=role_aware_role_redundancy_weight,
                    semantic_redundancy_weight=role_aware_semantic_redundancy_weight,
                    secondary_role_weight=role_aware_secondary_role_weight,
                    soft_query_prior=role_aware_soft_query_prior,
                    query_coverage_weight=role_aware_query_coverage_weight,
                    gate_semantic_redundancy=role_aware_gate_semantic_redundancy,
                    fix_first_relevance=role_aware_fix_first_relevance,
                )
            except Exception as exc:
                logger.warning("Role-aware clip retrieval failed; falling back to MMR: %s", exc)
                top_clips = _mmr_select_clips(
                    candidate_ids,
                    clip_scores,
                    clip_repr_map,
                    topk=topk,
                    mmr_lambda=clip_mmr_lambda,
                )
        else:
            top_clips = _mmr_select_clips(
                candidate_ids,
                clip_scores,
                clip_repr_map,
                topk=topk,
                mmr_lambda=clip_mmr_lambda,
            )
    else:
        # filter out clips that have 0 score and get top k clips
        if before_clip is not None:
            top_clips = [
                clip_id
                for clip_id, score in sorted_clips
                if score >= threshold and clip_id <= before_clip and clip_id not in excluded_clips
            ][:topk]
        else:
            top_clips = [
                clip_id
                for clip_id, score in sorted_clips
                if score >= threshold and clip_id not in excluded_clips
            ][:topk]
    return top_clips, clip_scores, nodes

def get_related_nodes(video_graph, query):
    related_nodes = []
    entities = parse_video_caption(video_graph, query)
    for entity in entities:
        type = entity[0]
        node_id = entity[1]
        if not (f"{type}_{node_id}" in video_graph.character_mappings.keys() or f"{type}_{node_id}" in video_graph.reverse_character_mappings.keys()):
            continue
        if type == "character":
            related_nodes.extend([int(node.split("_")[1]) for node in video_graph.character_mappings[f"{type}_{node_id}"]])
        else:
            related_nodes.append(node_id)
    return list(set(related_nodes))

def generate_action(question, knowledge, retrieval_plan=None, multiple_queries=False, responses=[], switch=False, model="models/gemini-2.5-pro"):
    # select prompt
    if not switch:
        if multiple_queries:
            prompt = prompt_generate_action_with_plan_multiple_queries
        else:
            prompt = prompt_generate_action_with_plan
            # prompt = prompt_generate_action_with_plan_multiple_queries
    else:
        logger.info(f"Route switch triggered.")
        if multiple_queries:
            prompt = prompt_generate_action_with_plan_multiple_queries_new_direction
        else:
            prompt = prompt_generate_action_with_plan_new_direction
            # prompt = prompt_generate_action_with_plan_multiple_queries_new_direction

    input = [
        {
            "type": "text",
            "content": prompt.format(
                question=question,
                knowledge=knowledge,
                retrieval_plan=retrieval_plan,
            )
        }
    ]
    messages = generate_messages(input)
    action_type = None
    action_content = None
    for i in range(MAX_RETRIES):
        action = get_response_with_retry(model, messages)[0]
        if "[ANSWER]" in action:
            action_type = "answer"
            reasoning = action.split("[ANSWER]")[0].strip()
            action_content = action.split("[ANSWER]")[1].strip()
        elif "[SEARCH]" in action:
            if not multiple_queries:
                action_type = "search"
                reasoning = action.split("[SEARCH]")[0].strip()
                action_content = action.split("[SEARCH]")[1].strip()
            else:
                action_type = "search"
                reasoning = action.split("[SEARCH]")[0].strip()
                action_content = select_queries(validate_and_fix_python_list(action.split("[SEARCH]")[1].strip()), responses)
        else:
            raise ValueError(f"Unknown action type: {action}")
        if action_content is not None:
            break
    if action_content is None:
        raise Exception("Failed to generate action")
    return reasoning, action_type, action_content

def select_queries(action_content, responses):
    if not action_content:
        return None

    history_queries = [response["action_content"] for response in responses]
    history_embeddings = parallel_get_embedding("text-embedding-3-large", history_queries)[0]

    queries = action_content
    embeddings = parallel_get_embedding("text-embedding-3-large", queries)[0]

    # If there are no history queries, return the first query
    if not history_queries:
        return queries[0]

    # Calculate cosine similarity between each query and all history queries
    avg_similarities = []
    for query_embedding in embeddings:
        similarities = []
        for history_embedding in history_embeddings:
            # Compute cosine similarity
            dot_product = sum(a*b for a,b in zip(query_embedding, history_embedding))
            query_norm = sum(a*a for a in query_embedding) ** 0.5
            history_norm = sum(b*b for b in history_embedding) ** 0.5
            cos_sim = dot_product / (query_norm * history_norm)
            similarities.append(cos_sim)
        # Calculate average similarity for this query
        avg_similarity = sum(similarities) / len(similarities)
        avg_similarities.append(avg_similarity)

    # Return query with lowest average similarity
    min_similarity_idx = avg_similarities.index(min(avg_similarities))
    return queries[min_similarity_idx]

def search(
    video_graph,
    query,
    current_clips,
    topk=5,
    mode='max',
    threshold=float("-inf"),
    mem_wise=False,
    before_clip=None,
    episodic_only=False,
    speaker_aware=False,
    speaker_bias=0.0,
    speaker_hard_filter=False,
    scene_nodes=None,
    scene_rerank_weight=0.3,
    diverse_clip_retrieval=False,
    diverse_clip_pool_size=12,
    diverse_clip_mmr_candidate_pool_size=200,
    clip_intra_similarity_threshold=0.85,
    clip_mmr_lambda=0.75,
    clip_max_nodes_for_diversity=8,
    dynamic_mmr_clip_retrieval=False,
    dynamic_mmr_min_clips=2,
    dynamic_mmr_max_clips=5,
    dynamic_mmr_stop_threshold=0.05,
    dynamic_mmr_extra_clips=0,
    dynamic_mmr_trace_path=None,
    dynamic_mmr_log_scores=False,
    dynamic_mmr_policy="threshold",
    dynamic_mmr_confidence_threshold=0.30,
    dynamic_mmr_ambiguity_gap_threshold=0.25,
    dynamic_mmr_knee_min_drop=0.25,
    dynamic_mmr_knee_alpha=1.0,
    dynamic_mmr_uncertainty_alpha=1.0,
    dynamic_mmr_score_source="clip_score",
    full_adaptive_k_retrieval=False,
    full_adaptive_k_strategy="largest_gap",
    full_adaptive_k_ignore_extreme=0.0,
    full_adaptive_k_ignore_extreme_tail=0.1,
    full_adaptive_k_ignore_below_median=False,
    full_adaptive_k_retrieve_more=5,
    full_adaptive_k_candidate_nodes=None,
    full_adaptive_k_min_nodes=1,
    full_adaptive_k_max_nodes=None,
    full_adaptive_k_min_clips=None,
    full_adaptive_k_max_clips=None,
    full_adaptive_k_extra_clips=0,
    clip_adaptive_k_retrieval=False,
    clip_adaptive_k_strategy="largest_gap",
    clip_adaptive_k_ignore_extreme=0.0,
    clip_adaptive_k_ignore_extreme_tail=0.1,
    clip_adaptive_k_ignore_below_median=False,
    clip_adaptive_k_retrieve_more=5,
    clip_adaptive_k_min_clips=1,
    clip_adaptive_k_max_clips=None,
    clip_adaptive_k_extra_clips=0,
    clip_adaptive_k_score_source="max_node",
    dynamicrag_clip_retrieval=False,
    dynamicrag_model="gasolsun/DynamicRAG-8B",
    dynamicrag_api_base=None,
    dynamicrag_api_key="EMPTY",
    dynamicrag_temperature=0.4,
    dynamicrag_max_tokens=100,
    dynamicrag_timeout=60,
    dynamicrag_min_clips=0,
    dynamicrag_max_clips=None,
    dynamicrag_max_nodes_per_clip=4,
    dynamicrag_max_doc_chars=1600,
    df_rag_clip_retrieval=False,
    df_rag_model="models/DynamicRAG-8B",
    df_rag_api_base=None,
    df_rag_api_key="EMPTY",
    df_rag_temperature=0.0,
    df_rag_planner_max_tokens=256,
    df_rag_evaluator_max_tokens=512,
    df_rag_timeout=60,
    df_rag_lambdas="0.1,0.2,0.3,0.4,0.5,0.6,0.7,0.8,0.9,1.0",
    df_rag_set_size=5,
    df_rag_max_nodes_per_clip=4,
    df_rag_max_doc_chars=1600,
    df_rag_fallback_lambda=0.5,
    adaptive_rag_retrieval=False,
    adaptive_rag_route_source="heuristic",
    adaptive_rag_classifier_path=None,
    adaptive_rag_classifier_model="adaptive-rag-classifier",
    adaptive_rag_classifier_api_base=None,
    adaptive_rag_classifier_api_key="EMPTY",
    adaptive_rag_classifier_temperature=0.0,
    adaptive_rag_classifier_max_tokens=4,
    adaptive_rag_classifier_timeout=60,
    adaptive_rag_fallback_label="B",
    adaptive_rag_question=None,
    adaptive_rag_question_id=None,
    adaptive_rag_zero_clips=0,
    adaptive_rag_single_clips=2,
    adaptive_rag_multi_clips=5,
    adaptive_rag_selector="top",
    adaptive_rag_score_source="max_node",
    evidence_saturation_retrieval=False,
    evidence_saturation_min_clips=1,
    evidence_saturation_max_clips=5,
    evidence_saturation_stop_threshold=0.02,
    evidence_saturation_relevance_weight=0.75,
    evidence_saturation_semantic_gain_weight=0.10,
    evidence_saturation_temporal_gain_weight=0.08,
    evidence_saturation_entity_gain_weight=0.06,
    evidence_saturation_action_state_gain_weight=0.06,
    evidence_saturation_semantic_redundancy_weight=0.15,
    evidence_saturation_temporal_redundancy_weight=0.0,
    evidence_saturation_temporal_bucket_size=4,
    evidence_saturation_near_clip_window=2,
    role_aware_clip_retrieval=False,
    role_aware_question=None,
    role_aware_model="local-qwen3-vl",
    role_aware_model_device=None,
    role_aware_max_new_tokens=2048,
    role_aware_cache_dir=None,
    role_aware_precomputed_dir=None,
    role_aware_max_nodes_per_clip=4,
    role_aware_question_role_mode="heuristic",
    role_aware_question_roles_dir=None,
    role_aware_role_match_weight=0.0,
    role_aware_relevance_weight=0.55,
    role_aware_coverage_weight=0.30,
    role_aware_instance_weight=0.25,
    role_aware_role_redundancy_weight=0.0,
    role_aware_semantic_redundancy_weight=0.10,
    role_aware_secondary_role_weight=0.0,
    role_aware_soft_query_prior=False,
    role_aware_query_coverage_weight=0.0,
    role_aware_gate_semantic_redundancy=False,
    role_aware_fix_first_relevance=False,
    fixed_clip_backfill_current=False,
):
    speaker_nodes = set()
    if speaker_aware:
        speaker_nodes = infer_speaker_nodes_from_query(video_graph, query)
    top_clips, clip_scores, nodes = retrieve_from_videograph(
        video_graph,
        query,
        topk,
        mode,
        threshold,
        before_clip,
        speaker_nodes=speaker_nodes,
        speaker_bias=speaker_bias,
        speaker_hard_filter=speaker_hard_filter,
        scene_nodes=scene_nodes,
        scene_rerank_weight=scene_rerank_weight,
        diverse_clip_retrieval=diverse_clip_retrieval,
        diverse_clip_pool_size=diverse_clip_pool_size,
        diverse_clip_mmr_candidate_pool_size=diverse_clip_mmr_candidate_pool_size,
        clip_intra_similarity_threshold=clip_intra_similarity_threshold,
        clip_mmr_lambda=clip_mmr_lambda,
        clip_max_nodes_for_diversity=clip_max_nodes_for_diversity,
        excluded_clips=(
            current_clips
            if (diverse_clip_retrieval or fixed_clip_backfill_current) and not mem_wise
            else None
        ),
        dynamic_mmr_clip_retrieval=dynamic_mmr_clip_retrieval and not mem_wise,
        dynamic_mmr_min_clips=dynamic_mmr_min_clips,
        dynamic_mmr_max_clips=dynamic_mmr_max_clips,
        dynamic_mmr_stop_threshold=dynamic_mmr_stop_threshold,
        dynamic_mmr_extra_clips=dynamic_mmr_extra_clips,
        dynamic_mmr_trace_path=dynamic_mmr_trace_path,
        dynamic_mmr_log_scores=dynamic_mmr_log_scores,
        dynamic_mmr_policy=dynamic_mmr_policy,
        dynamic_mmr_confidence_threshold=dynamic_mmr_confidence_threshold,
        dynamic_mmr_ambiguity_gap_threshold=dynamic_mmr_ambiguity_gap_threshold,
        dynamic_mmr_knee_min_drop=dynamic_mmr_knee_min_drop,
        dynamic_mmr_knee_alpha=dynamic_mmr_knee_alpha,
        dynamic_mmr_uncertainty_alpha=dynamic_mmr_uncertainty_alpha,
        dynamic_mmr_score_source=dynamic_mmr_score_source,
        full_adaptive_k_retrieval=full_adaptive_k_retrieval and not mem_wise,
        full_adaptive_k_strategy=full_adaptive_k_strategy,
        full_adaptive_k_ignore_extreme=full_adaptive_k_ignore_extreme,
        full_adaptive_k_ignore_extreme_tail=full_adaptive_k_ignore_extreme_tail,
        full_adaptive_k_ignore_below_median=full_adaptive_k_ignore_below_median,
        full_adaptive_k_retrieve_more=full_adaptive_k_retrieve_more,
        full_adaptive_k_candidate_nodes=full_adaptive_k_candidate_nodes,
        full_adaptive_k_min_nodes=full_adaptive_k_min_nodes,
        full_adaptive_k_max_nodes=full_adaptive_k_max_nodes,
        full_adaptive_k_min_clips=full_adaptive_k_min_clips,
        full_adaptive_k_max_clips=full_adaptive_k_max_clips,
        full_adaptive_k_extra_clips=full_adaptive_k_extra_clips,
        clip_adaptive_k_retrieval=clip_adaptive_k_retrieval and not mem_wise,
        clip_adaptive_k_strategy=clip_adaptive_k_strategy,
        clip_adaptive_k_ignore_extreme=clip_adaptive_k_ignore_extreme,
        clip_adaptive_k_ignore_extreme_tail=clip_adaptive_k_ignore_extreme_tail,
        clip_adaptive_k_ignore_below_median=clip_adaptive_k_ignore_below_median,
        clip_adaptive_k_retrieve_more=clip_adaptive_k_retrieve_more,
        clip_adaptive_k_min_clips=clip_adaptive_k_min_clips,
        clip_adaptive_k_max_clips=clip_adaptive_k_max_clips,
        clip_adaptive_k_extra_clips=clip_adaptive_k_extra_clips,
        clip_adaptive_k_score_source=clip_adaptive_k_score_source,
        dynamicrag_clip_retrieval=dynamicrag_clip_retrieval and not mem_wise,
        dynamicrag_model=dynamicrag_model,
        dynamicrag_api_base=dynamicrag_api_base,
        dynamicrag_api_key=dynamicrag_api_key,
        dynamicrag_temperature=dynamicrag_temperature,
        dynamicrag_max_tokens=dynamicrag_max_tokens,
        dynamicrag_timeout=dynamicrag_timeout,
        dynamicrag_min_clips=dynamicrag_min_clips,
        dynamicrag_max_clips=dynamicrag_max_clips,
        dynamicrag_max_nodes_per_clip=dynamicrag_max_nodes_per_clip,
        dynamicrag_max_doc_chars=dynamicrag_max_doc_chars,
        df_rag_clip_retrieval=df_rag_clip_retrieval and not mem_wise,
        df_rag_model=df_rag_model,
        df_rag_api_base=df_rag_api_base,
        df_rag_api_key=df_rag_api_key,
        df_rag_temperature=df_rag_temperature,
        df_rag_planner_max_tokens=df_rag_planner_max_tokens,
        df_rag_evaluator_max_tokens=df_rag_evaluator_max_tokens,
        df_rag_timeout=df_rag_timeout,
        df_rag_lambdas=df_rag_lambdas,
        df_rag_set_size=df_rag_set_size,
        df_rag_max_nodes_per_clip=df_rag_max_nodes_per_clip,
        df_rag_max_doc_chars=df_rag_max_doc_chars,
        df_rag_fallback_lambda=df_rag_fallback_lambda,
        adaptive_rag_retrieval=adaptive_rag_retrieval and not mem_wise,
        adaptive_rag_route_source=adaptive_rag_route_source,
        adaptive_rag_classifier_path=adaptive_rag_classifier_path,
        adaptive_rag_classifier_model=adaptive_rag_classifier_model,
        adaptive_rag_classifier_api_base=adaptive_rag_classifier_api_base,
        adaptive_rag_classifier_api_key=adaptive_rag_classifier_api_key,
        adaptive_rag_classifier_temperature=adaptive_rag_classifier_temperature,
        adaptive_rag_classifier_max_tokens=adaptive_rag_classifier_max_tokens,
        adaptive_rag_classifier_timeout=adaptive_rag_classifier_timeout,
        adaptive_rag_fallback_label=adaptive_rag_fallback_label,
        adaptive_rag_question=adaptive_rag_question,
        adaptive_rag_question_id=adaptive_rag_question_id,
        adaptive_rag_zero_clips=adaptive_rag_zero_clips,
        adaptive_rag_single_clips=adaptive_rag_single_clips,
        adaptive_rag_multi_clips=adaptive_rag_multi_clips,
        adaptive_rag_selector=adaptive_rag_selector,
        adaptive_rag_score_source=adaptive_rag_score_source,
        evidence_saturation_retrieval=evidence_saturation_retrieval and not mem_wise,
        evidence_saturation_min_clips=evidence_saturation_min_clips,
        evidence_saturation_max_clips=evidence_saturation_max_clips,
        evidence_saturation_stop_threshold=evidence_saturation_stop_threshold,
        evidence_saturation_relevance_weight=evidence_saturation_relevance_weight,
        evidence_saturation_semantic_gain_weight=evidence_saturation_semantic_gain_weight,
        evidence_saturation_temporal_gain_weight=evidence_saturation_temporal_gain_weight,
        evidence_saturation_entity_gain_weight=evidence_saturation_entity_gain_weight,
        evidence_saturation_action_state_gain_weight=evidence_saturation_action_state_gain_weight,
        evidence_saturation_semantic_redundancy_weight=evidence_saturation_semantic_redundancy_weight,
        evidence_saturation_temporal_redundancy_weight=evidence_saturation_temporal_redundancy_weight,
        evidence_saturation_temporal_bucket_size=evidence_saturation_temporal_bucket_size,
        evidence_saturation_near_clip_window=evidence_saturation_near_clip_window,
        role_aware_clip_retrieval=role_aware_clip_retrieval and not mem_wise,
        role_aware_question=role_aware_question,
        role_aware_model=role_aware_model,
        role_aware_model_device=role_aware_model_device,
        role_aware_max_new_tokens=role_aware_max_new_tokens,
        role_aware_cache_dir=role_aware_cache_dir,
        role_aware_precomputed_dir=role_aware_precomputed_dir,
        role_aware_max_nodes_per_clip=role_aware_max_nodes_per_clip,
        role_aware_question_role_mode=role_aware_question_role_mode,
        role_aware_question_roles_dir=role_aware_question_roles_dir,
        role_aware_role_match_weight=role_aware_role_match_weight,
        role_aware_relevance_weight=role_aware_relevance_weight,
        role_aware_coverage_weight=role_aware_coverage_weight,
        role_aware_instance_weight=role_aware_instance_weight,
        role_aware_role_redundancy_weight=role_aware_role_redundancy_weight,
        role_aware_semantic_redundancy_weight=role_aware_semantic_redundancy_weight,
        role_aware_secondary_role_weight=role_aware_secondary_role_weight,
        role_aware_soft_query_prior=role_aware_soft_query_prior,
        role_aware_query_coverage_weight=role_aware_query_coverage_weight,
        role_aware_gate_semantic_redundancy=role_aware_gate_semantic_redundancy,
        role_aware_fix_first_relevance=role_aware_fix_first_relevance,
        fixed_clip_backfill_current=fixed_clip_backfill_current,
    )

    if mem_wise:
        new_memories = {}
        top_nodes_num = 0
        # fetch top nodes
        for top_node, _ in nodes:
            clip_id = video_graph.nodes[top_node].metadata['timestamp']
            if before_clip is not None and clip_id > before_clip:
                continue
            if clip_id not in new_memories:
                new_memories[clip_id] = []
            new_ = translate(video_graph, video_graph.nodes[top_node].metadata['contents'])
            new_memories[clip_id].extend(new_)
            top_nodes_num += len(new_)
            if top_nodes_num >= topk:
                break
        # sort related_memories by timestamp
        new_memories = dict(sorted(new_memories.items(), key=lambda x: x[0]))
        new_memories = {f"CLIP_{k}": v for k, v in new_memories.items() if len(v) > 0}
        return new_memories, current_clips, clip_scores

    new_clips = [top_clip for top_clip in top_clips if top_clip not in current_clips]
    new_memories = {}
    current_clips.extend(new_clips)

    for new_clip in new_clips:
        if new_clip not in video_graph.text_nodes_by_clip:
            new_memories[new_clip] = [f"CLIP_{new_clip} not found in memory bank, please search for other information"]
        else:
            related_nodes = video_graph.text_nodes_by_clip[new_clip]
            new_memories[new_clip] = translate(video_graph, [video_graph.nodes[node_id].metadata['contents'][0] for node_id in related_nodes if (not episodic_only or video_graph.nodes[node_id].type != "semantic")])

    # sort related_memories by timestamp
    new_memories = dict(sorted(new_memories.items(), key=lambda x: x[0]))
    new_memories = {f"CLIP_{k}": v for k, v in new_memories.items()}

    return new_memories, current_clips, clip_scores

def answer_with_retrieval(video_graph, question, video_clip_base64=None, topk=5, auto_refresh=False, mode='max', multiple_queries=False, max_retrieval_steps=10, route_switch=True, threshold=0, model="models/gemini-2.5-pro", before_clip=None):
    if before_clip is not None:
        video_graph.truncate_memory_by_clip(before_clip)

    if auto_refresh:
        video_graph.refresh_equivalences()

    related_clips = []
    context = []

    final_answer = None

    memories = [[]]
    responses = []

    if video_clip_base64 is not None:
        input = [
            {
                "type": "video_base64/mp4",
                "content": video_clip_base64,
            },
            {
                "type": "text",
                "content": prompt_generate_plan.format(question=question),
            }
        ]

        messages = generate_messages(input)
        plan_model = "models/gemini-3-pro-preview"
        retrieval_plan = get_response_with_retry(plan_model, messages)[0]
        logger.info(f"Retrieval plan: {retrieval_plan}")
    else:
        retrieval_plan = None

    switch = False
    for i in range(max_retrieval_steps):
        # reasoning, action_type, action_content = generate_action(question, context, retrieval_plan)
        reasoning, action_type, action_content = generate_action(question, context, retrieval_plan, multiple_queries=multiple_queries, responses=responses, switch=switch, model=model)
        reasoning = reasoning.strip("### Reasoning:").strip("### Answer or Search:").strip("Reasoning:").strip()
        if action_type == "answer":
            final_answer = action_content
            responses.append({
                "reasoning": reasoning,
                "action_type": action_type,
                "action_content": action_content
            })
            logger.info(f"Answer: {final_answer}")
            break
        elif action_type == "search":
            if i == max_retrieval_steps - 1:
                input = [
                    {
                        "type": "text",
                        "content": prompt_answer_with_retrieval_final.format(
                            question=question,
                            information=context,
                        ),
                    }
                ]
                messages = generate_messages(input)
                resp = get_response_with_retry(model, messages)[0]
                reasoning = resp.split("[ANSWER]")[0].strip()
                final_answer = resp.split("[ANSWER]")[1].strip()
                responses.append({
                    "reasoning": reasoning,
                    "action_type": "answer",
                    "action_content": final_answer
                })
                logger.info(f"Forced answer: {final_answer}")
                break

            new_memories, related_clips, _ = search(video_graph, action_content, related_clips, topk, mode, threshold=threshold, before_clip=before_clip)

            if len(new_memories.items()) == 0 and route_switch:
                switch = True
            else:
                switch = False

            context.append({
                "reasoning": reasoning,
                "query": action_content,
                "retrieved memories": new_memories
            })

            new_response_item = {
                "reasoning": reasoning,
                "action_type": action_type,
                "action_content": action_content
            }
            responses.append(new_response_item)

            new_memory_items = [{
                "clip_id": k,
                "memory": v
            } for k, v in new_memories.items()]
            memories.append(new_memory_items)

            if processing_config["logging"] == "DETAIL":
                logger.debug("=" * 10 + "Retrieval Step " + str(i+1) + "=" * 10)
                logger.debug(new_response_item)
                logger.debug(new_memory_items)

    return final_answer, (memories, responses)

def verify_qa(question, gt, pred, model="gpt-4o"):
    try:
        input = [
            {
                "type": "text",
                "content": prompt_agent_verify_answer_referencing.format(
                    question=question,
                    ground_truth_answer=gt,
                    agent_answer=pred,
                ),
            }
        ]
        messages = generate_messages(input)
        response = get_response_with_retry(model, messages)
        result = response[0]
    except Exception as e:
        logger.error(f"Error verifying qa: {question}")
        logger.error(str(e))
        return None
    return result

def calculate_similarity(mem, query, related_nodes):
    related_nodes_embeddings = np.array([mem.nodes[node_id].embeddings[0] for node_id in related_nodes])
    query_embedding = np.array(get_embedding_with_retry("text-embedding-3-large", query)[0]).reshape(1, -1)
    similarities = cosine_similarity(query_embedding, related_nodes_embeddings)[0]
    return similarities.tolist()

def retrieve_all_episodic_memories(video_graph):
    episodic_memories = {}
    for node_id in video_graph.text_nodes:
        if video_graph.nodes[node_id].type == "episodic":
            clips_id = f"CLIP_{video_graph.nodes[node_id].metadata['timestamp']}"
            if clips_id not in episodic_memories:
                episodic_memories[clips_id] = []
            episodic_memories[clips_id].extend(video_graph.nodes[node_id].metadata["contents"])
    return episodic_memories

def retrieve_all_semantic_memories(video_graph):
    semantic_memories = {}
    for node_id in video_graph.text_nodes:
        if video_graph.nodes[node_id].type == "semantic":
            clips_id = f"CLIP_{video_graph.nodes[node_id].metadata['timestamp']}"
            if clips_id not in semantic_memories:
                semantic_memories[clips_id] = []
            semantic_memories[clips_id].extend(video_graph.nodes[node_id].metadata["contents"])
    return semantic_memories


if __name__ == "__main__":
    from utils.general import load_video_graph
    import base64
    processing_config["logging"] = "DETAIL"
    processing_config["topk"] = 30

    def video_to_base64(video_path):
        with open(video_path, 'rb') as video_file:
            video_bytes = video_file.read()
            base64_encoded = base64.b64encode(video_bytes).decode('utf-8')
            return base64_encoded

    video_graph_path = "/mnt/hdfs/foundation/longlin.kylin/mmagent/data/mems/CZ_1/Efk3K4epEzg_30_5_-1_10_20_0.3_0.6.pkl"
    video_graph = load_video_graph(video_graph_path)

    question = "Which collection has the highest starting price?"
    answer = answer_with_retrieval(video_graph, question, video_to_base64("/mnt/hdfs/foundation/longlin.kylin/mmagent/data/video_clips/CZ_1/Efk3K4epEzg/39.mp4"), topk=processing_config["topk"], multiple_queries=processing_config["multiple_queries"], max_retrieval_steps=processing_config["max_retrieval_steps"])
