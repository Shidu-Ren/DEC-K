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
import re
import os
import sys
import json
import time
import argparse
import multiprocessing
import mmagent.videograph
from mmagent.retrieve import get_identity_hints, search
from mmagent.robot_dev import augment_robot_dev_memories
from vllm import LLM, SamplingParams
from transformers import AutoTokenizer
from mmagent.videograph_dev import VideoGraphDev
from mmagent.utils.general import load_video_graph
from mmagent.utils.chat_api import generate_messages, get_response_with_retry, get_response_with_usage_retry
from mmagent.prompts import prompt_agent_verify_answer_referencing

sys.modules["videograph"] = mmagent.videograph
processing_config = json.load(open("configs/processing_config.json"))
model_name = "models/M3-Agent-Control"
eval_model = os.getenv("M3AGENT_EVAL_MODEL", "gpt-4o")


def _model_price_per_1m(model):
    normalized = model[len("models/"):] if model.startswith("models/") else model
    input_price = float(os.getenv("M3AGENT_JUDGE_INPUT_PRICE_PER_1M", "2.50"))
    output_price = float(os.getenv("M3AGENT_JUDGE_OUTPUT_PRICE_PER_1M", "10.00"))
    if normalized.startswith("gpt-4o-mini"):
        input_price = float(os.getenv("M3AGENT_JUDGE_INPUT_PRICE_PER_1M", "0.15"))
        output_price = float(os.getenv("M3AGENT_JUDGE_OUTPUT_PRICE_PER_1M", "0.60"))
    return input_price, output_price


def estimate_usage_cost(model, usage):
    input_price, output_price = _model_price_per_1m(model)
    input_tokens = int(usage.get("input_tokens", 0) or 0)
    output_tokens = int(usage.get("output_tokens", 0) or 0)
    total = (input_tokens * input_price + output_tokens * output_price) / 1_000_000.0
    return {
        "model": model,
        "input_price_per_1m": input_price,
        "output_price_per_1m": output_price,
        "usd": total,
    }


def parse_int_or_float(value):
    text = str(value).strip()
    if re.fullmatch(r"[+-]?\d+", text):
        return int(text)
    return float(text)


def normalize_choices(choices):
    if not isinstance(choices, dict):
        return {}
    normalized = {}
    for key, value in choices.items():
        option = str(key).strip().upper()
        if len(option) == 1 and option in {"A", "B", "C", "D"}:
            normalized[option] = str(value).strip()
    return {key: normalized[key] for key in sorted(normalized)}


def format_question_for_prompt(question, choices=None):
    choices = normalize_choices(choices)
    if not choices:
        return question

    lines = [question, "", "Choices:"]
    for option, text in choices.items():
        lines.append(f"{option}. {text}")
    lines.append("")
    lines.append("If you choose Action: [Answer], the Content must be only the option letter (A/B/C/D).")
    return "\n".join(lines)


def extract_option_letter(response, choices):
    choices = normalize_choices(choices)
    if not choices or not response:
        return None

    text = str(response).strip()
    compact = text.replace("*", " ").replace("`", " ")
    patterns = [
        r"\b(?:answer|option|choice)\s*[:：]?\s*\**\s*([ABCD])\b",
        r"^\s*\**\s*([ABCD])\s*[\.\):：-]?\s*$",
        r"\b([ABCD])\b",
    ]
    for pattern in patterns:
        match = re.search(pattern, compact, flags=re.IGNORECASE)
        if match:
            option = match.group(1).upper()
            if option in choices:
                return option

    lowered = " ".join(text.lower().split())
    for option, option_text in choices.items():
        normalized_text = " ".join(option_text.lower().split())
        if lowered == normalized_text:
            return option
        if lowered.startswith(normalized_text):
            return option

    return None

def eval_answer(question, predict, ground_truth):
    empty_usage = {"input_tokens": 0, "output_tokens": 0, "total_tokens": 0}
    if predict == "":
        return False, empty_usage, estimate_usage_cost(eval_model, empty_usage)
    try:
        input = [
            {
                "type": "text",
                "content": prompt_agent_verify_answer_referencing.format(
                    question=question,
                    ground_truth_answer=ground_truth,
                    agent_answer=predict,
                ),
            }
        ]
        messages = generate_messages(input)
        response, usage = get_response_with_usage_retry(eval_model, messages, timeout=60)
        result = response.lower() if response else ""
    except Exception as e:
        print(f"Error verifying qa: {question} | {str(e)}")
        return False, empty_usage, estimate_usage_cost(eval_model, empty_usage)
    return True if "yes" in result else False, usage, estimate_usage_cost(eval_model, usage)

system_prompt = "You are given a question and some relevant knowledge. Your task is to reason about whether the provided knowledge is sufficient to answer the question. If it is sufficient, output [Answer] followed by the answer. If it is not sufficient, output [Search] and generate a query that will be encoded into embeddings for a vector similarity search. The query will help retrieve additional information from a memory bank.\n\nQuestion: {question}"
instruction = f"""

Output the answer in the format:
Action: [Answer] or [Search]
Content: {{content}}

If the answer cannot be derived yet, the {{content}} should be a single search query that would help retrieve the missing information. The search {{content}} needs to be different from the previous.
You can get the mapping relationship between character ID and name by using search query such as: "What is the name of <character_{{i}}>" or "What is the character id of {{name}}".
After obtaining the mapping, it is best to use character ID instead of name for searching.
If the answer can be derived from the provided knowledge, the {{content}} is the specific answer to the question. Only name can appear in the answer, not character ID like <character_{{i}}>.
If the question includes answer choices, then the {{content}} for Action: [Answer] must be only the option letter, such as A, B, C, or D."""

tokenizer = AutoTokenizer.from_pretrained(model_name)
sampling_params = SamplingParams(
    temperature=float(os.getenv("M3AGENT_CONTROL_TEMPERATURE", "0.6")),
    top_p=float(os.getenv("M3AGENT_CONTROL_TOP_P", "0.95")),
    top_k=int(os.getenv("M3AGENT_CONTROL_TOP_K", "20")),
    max_tokens=int(os.getenv("M3AGENT_CONTROL_MAX_TOKENS", "1024")),
)
pattern = r"Action: \[(.*)\].*Content: (.*)"
EQUIVALENCE_MODE = "default"
SPEAKER_AWARE_RETRIEVAL = False
SPEAKER_RETRIEVAL_BIAS = 0.0
SPEAKER_RETRIEVAL_HARD_FILTER = False
STRIP_TEMPORAL_EDGES = False
STRIP_SCENE_NODES = False
SCENE_RERANK = False
SCENE_RERANK_WEIGHT = 0.14
SCENE_BACKGROUND = False
DIVERSE_CLIP_RETRIEVAL = False
FIXED_CLIP_BACKFILL_CURRENT = False
DIVERSE_CLIP_POOL_SIZE = 12
DIVERSE_CLIP_MMR_CANDIDATE_POOL_SIZE = 200
CLIP_INTRA_SIMILARITY_THRESHOLD = 0.85
CLIP_MMR_LAMBDA = 0.75
CLIP_MAX_NODES_FOR_DIVERSITY = 8
DYNAMIC_MMR_SCORE_SOURCE = "clip_score"
DYNAMIC_MMR_TYPE_SWITCH = False
DYNAMIC_MMR_TYPE_SWITCH_TYPES = set()
DYNAMIC_MMR_TYPE_SWITCH_POLICY = "threshold"
DYNAMIC_MMR_TYPE_SWITCH_STOP_THRESHOLD = 0.20
DYNAMIC_MMR_TYPE_SWITCH_MIN_CLIPS = 2
DYNAMIC_MMR_TYPE_SWITCH_MAX_CLIPS = 5
DYNAMICRAG_CLIP_RETRIEVAL = False
DYNAMICRAG_MODEL = "gasolsun/DynamicRAG-8B"
DYNAMICRAG_API_BASE = None
DYNAMICRAG_API_KEY = "EMPTY"
DYNAMICRAG_TEMPERATURE = 0.4
DYNAMICRAG_MAX_TOKENS = 100
DYNAMICRAG_TIMEOUT = 60
DYNAMICRAG_MIN_CLIPS = 0
DYNAMICRAG_MAX_CLIPS = None
DYNAMICRAG_MAX_NODES_PER_CLIP = 4
DYNAMICRAG_MAX_DOC_CHARS = 1600
DF_RAG_CLIP_RETRIEVAL = False
DF_RAG_MODEL = "models/DynamicRAG-8B"
DF_RAG_API_BASE = None
DF_RAG_API_KEY = "EMPTY"
DF_RAG_TEMPERATURE = 0.0
DF_RAG_PLANNER_MAX_TOKENS = 256
DF_RAG_EVALUATOR_MAX_TOKENS = 512
DF_RAG_TIMEOUT = 60
DF_RAG_LAMBDAS = "0.1,0.2,0.3,0.4,0.5,0.6,0.7,0.8,0.9,1.0"
DF_RAG_SET_SIZE = 5
DF_RAG_MAX_NODES_PER_CLIP = 4
DF_RAG_MAX_DOC_CHARS = 1600
DF_RAG_FALLBACK_LAMBDA = 0.5
ADAPTIVE_RAG_RETRIEVAL = False
ADAPTIVE_RAG_ROUTE_SOURCE = "heuristic"
ADAPTIVE_RAG_CLASSIFIER_PATH = None
ADAPTIVE_RAG_CLASSIFIER_MODEL = "adaptive-rag-classifier"
ADAPTIVE_RAG_CLASSIFIER_API_BASE = None
ADAPTIVE_RAG_CLASSIFIER_API_KEY = "EMPTY"
ADAPTIVE_RAG_CLASSIFIER_TEMPERATURE = 0.0
ADAPTIVE_RAG_CLASSIFIER_MAX_TOKENS = 4
ADAPTIVE_RAG_CLASSIFIER_TIMEOUT = 60
ADAPTIVE_RAG_FALLBACK_LABEL = "B"
ADAPTIVE_RAG_ZERO_CLIPS = 0
ADAPTIVE_RAG_SINGLE_CLIPS = 2
ADAPTIVE_RAG_MULTI_CLIPS = 5
ADAPTIVE_RAG_SELECTOR = "top"
ADAPTIVE_RAG_SCORE_SOURCE = "max_node"
ROLE_AWARE_CLIP_RETRIEVAL = False
ROLE_AWARE_MODEL = "local-qwen3-vl"
ROLE_AWARE_MODEL_DEVICE = None
ROLE_AWARE_MAX_NEW_TOKENS = 2048
ROLE_AWARE_CACHE_DIR = None
ROLE_AWARE_PRECOMPUTED_DIR = None
ROLE_AWARE_MAX_NODES_PER_CLIP = 4
ROLE_AWARE_QUESTION_ROLE_MODE = "heuristic"
ROLE_AWARE_QUESTION_ROLES_DIR = None
ROLE_AWARE_ROLE_MATCH_WEIGHT = 0.0
ROLE_AWARE_RELEVANCE_WEIGHT = 0.55
ROLE_AWARE_COVERAGE_WEIGHT = 0.30
ROLE_AWARE_INSTANCE_WEIGHT = 0.25
ROLE_AWARE_ROLE_REDUNDANCY_WEIGHT = 0.0
ROLE_AWARE_SEMANTIC_REDUNDANCY_WEIGHT = 0.10
ROLE_AWARE_SECONDARY_ROLE_WEIGHT = 0.0
ROLE_AWARE_SOFT_QUERY_PRIOR = False
ROLE_AWARE_QUERY_COVERAGE_WEIGHT = 0.0
ROLE_AWARE_GATE_SEMANTIC_REDUNDANCY = False
ROLE_AWARE_FIX_FIRST_RELEVANCE = False
DEV_VOICE_EQUIV_THRESHOLD = 0.5
DEV_VOICE_CLUSTER_THRESHOLD = 0.65
IDENTITY_HINT_MODE = "off"
ROBOT_DEV_MODE = False
ROBOT_DEV_CONTEXT_ROOT = "data/vl_contexts/robot"
ROBOT_DEV_CLIP_ROOT = "data/clips/robot"
ROBOT_DEV_INTERMEDIATE_ROOT = "data/intermediate_outputs/robot"
ROBOT_DEV_DETAIL_ROOT = "data/robot_dev_descriptions/robot"
ROBOT_DEV_VL_MODEL_PATH = "models/Qwen3-VL-8B-Instruct"
ROBOT_DEV_VL_DEVICE = None
ROBOT_DEV_FORCE_REGEN = False
ROBOT_DEV_FACES_INPUT = "face_only"
ROBOT_DEV_MAX_DETAIL_ITEMS = 12
ROBOT_DEV_MAX_NEW_TOKENS = 1024


def promote_graph_to_dev(mem_node):
    """Upgrade a loaded VideoGraph instance to VideoGraphDev in-place.

    This lets us apply the V2 dev union logic on top of an existing non-dev
    memory graph (e.g. *_audio_only.pkl) without regenerating memories.
    """
    if mem_node is None:
        return None
    if not isinstance(mem_node, VideoGraphDev):
        mem_node.__class__ = VideoGraphDev
    if not hasattr(mem_node, "voice_equiv_threshold"):
        mem_node.voice_equiv_threshold = DEV_VOICE_EQUIV_THRESHOLD
    else:
        mem_node.voice_equiv_threshold = DEV_VOICE_EQUIV_THRESHOLD
    if not hasattr(mem_node, "voice_cluster_threshold"):
        mem_node.voice_cluster_threshold = DEV_VOICE_CLUSTER_THRESHOLD
    else:
        mem_node.voice_cluster_threshold = DEV_VOICE_CLUSTER_THRESHOLD
    if not hasattr(mem_node, "character_names"):
        mem_node.character_names = {}
    if not hasattr(mem_node, "voice_names"):
        mem_node.voice_names = {}
    if not hasattr(mem_node, "rejected_equivalences"):
        mem_node.rejected_equivalences = []
    if not hasattr(mem_node, "clustered_merges"):
        mem_node.clustered_merges = []
    return mem_node


def dynamic_mmr_params_for_sample(data):
    policy = DYNAMIC_MMR_POLICY
    min_clips = DYNAMIC_MMR_MIN_CLIPS
    max_clips = DYNAMIC_MMR_MAX_CLIPS
    stop_threshold = DYNAMIC_MMR_STOP_THRESHOLD
    question_types = set(data.get("question_types") or [])
    if DYNAMIC_MMR_TYPE_SWITCH and question_types & DYNAMIC_MMR_TYPE_SWITCH_TYPES:
        policy = DYNAMIC_MMR_TYPE_SWITCH_POLICY
        min_clips = DYNAMIC_MMR_TYPE_SWITCH_MIN_CLIPS
        max_clips = DYNAMIC_MMR_TYPE_SWITCH_MAX_CLIPS
        stop_threshold = DYNAMIC_MMR_TYPE_SWITCH_STOP_THRESHOLD
    return policy, min_clips, max_clips, stop_threshold


def consumer(data):
    if not data["finish"]:
        before_clip = data.get("before_clip", None)
        response = data["conversations"][-1]["content"]
        match_result = re.search(pattern, response.split("</think>")[-1], re.DOTALL)
        if match_result:
            action = match_result.group(1)
            content = match_result.group(2)
        else:
            action = "Search"
            content = None
        if action == "Answer":
            data["response"] = content
            data["finish"] = True
        else:
            new_memories = {}
            scene_nodes_text_dict = {}
            clip_scores = {}
            if content:
                mem_node = load_video_graph(data["mem_path"])
                if STRIP_TEMPORAL_EDGES and mem_node is not None:
                    to_del = [(a, b) for (a, b), w in list(mem_node.edges.items())
                              if a in mem_node.nodes and b in mem_node.nodes
                              and mem_node.nodes[a].type == 'voice'
                              and mem_node.nodes[b].type == 'voice'
                              and abs(w - 0.5) < 0.01]
                    for edge in to_del:
                        del mem_node.edges[edge]
                if STRIP_SCENE_NODES and mem_node is not None:
                    to_del = [(a, b) for (a, b), w in list(mem_node.edges.items())
                              if abs(w - 0.6) < 0.01]
                    for edge in to_del:
                        del mem_node.edges[edge]
                # Scene Node logic: extract scene nodes
                scene_nodes_info = None
                scene_nodes_text_dict = {}
                if (SCENE_RERANK or SCENE_BACKGROUND) and mem_node is not None:
                    scene_node_ids = set()
                    for nid in mem_node.text_nodes:
                        if mem_node.nodes[nid].type != 'semantic':
                            continue
                        node_edges = [(a, b, w) for (a, b), w in mem_node.edges.items()
                                      if a == nid or b == nid]
                        if node_edges and all(abs(w - 0.6) < 0.01 for _, _, w in node_edges):
                            scene_node_ids.add(nid)
                    if scene_node_ids:
                        scene_nodes_info = []
                        for nid in scene_node_ids:
                            clip_id = mem_node.nodes[nid].metadata.get('timestamp')
                            emb = mem_node.nodes[nid].embeddings[0]
                            scene_nodes_info.append((emb, clip_id))
                            scene_nodes_text_dict[clip_id] = mem_node.nodes[nid].metadata.get('contents', [''])[0]
                        # Strip scene edges so they don't interfere with normal retrieval
                        to_del = [(a, b) for (a, b), w in list(mem_node.edges.items())
                                  if abs(w - 0.6) < 0.01]
                        for edge in to_del:
                            del mem_node.edges[edge]
                if mem_node is None:
                    new_memories = {}
                    search_result = (
                        "Searched knowledge: {}"
                        "\n(The memory graph is missing for this sample.)"
                    )
                    data["conversations"].append({"role": "user", "content": search_result})
                    return data
                mem_node._m3agent_mem_path = data["mem_path"]
                mem_node._m3agent_video_id = os.path.splitext(os.path.basename(data["mem_path"]))[0]
                if before_clip is not None:
                    mem_node.truncate_memory_by_clip(before_clip, False)
                if EQUIVALENCE_MODE == "dev":
                    mem_node = promote_graph_to_dev(mem_node)
                    mem_node.refresh_equivalences()
                elif EQUIVALENCE_MODE == "dev_one_to_one":
                    mem_node.refresh_equivalences_dev_one_to_one()
                elif EQUIVALENCE_MODE == "no_equiv":
                    mem_node.order_character()
                elif EQUIVALENCE_MODE == "none":
                    pass
                else:
                    mem_node.refresh_equivalences()
                scene_nodes_to_pass = scene_nodes_info if SCENE_RERANK else None
                identity_hints = []
                if IDENTITY_HINT_MODE == "basic":
                    identity_hints = get_identity_hints(mem_node, content)

                if identity_hints:
                    new_memories["IDENTITY_HINTS"] = identity_hints
                elif "character id" in content:
                    dynamic_policy, dynamic_min_clips, dynamic_max_clips, dynamic_stop_threshold = dynamic_mmr_params_for_sample(data)
                    memories, _, clip_scores = search(
                        mem_node,
                        content,
                        [],
                        mem_wise=True,
                        topk=20,
                        before_clip=before_clip,
                        speaker_aware=SPEAKER_AWARE_RETRIEVAL,
                        speaker_bias=SPEAKER_RETRIEVAL_BIAS,
                        speaker_hard_filter=SPEAKER_RETRIEVAL_HARD_FILTER,
                        scene_nodes=scene_nodes_to_pass,
                        scene_rerank_weight=SCENE_RERANK_WEIGHT,
                        diverse_clip_retrieval=DIVERSE_CLIP_RETRIEVAL,
                        diverse_clip_pool_size=DIVERSE_CLIP_POOL_SIZE,
                        diverse_clip_mmr_candidate_pool_size=DIVERSE_CLIP_MMR_CANDIDATE_POOL_SIZE,
                        clip_intra_similarity_threshold=CLIP_INTRA_SIMILARITY_THRESHOLD,
                        clip_mmr_lambda=CLIP_MMR_LAMBDA,
                        clip_max_nodes_for_diversity=CLIP_MAX_NODES_FOR_DIVERSITY,
                        dynamic_mmr_clip_retrieval=DYNAMIC_MMR_CLIP_RETRIEVAL,
                        dynamic_mmr_min_clips=dynamic_min_clips,
                        dynamic_mmr_max_clips=dynamic_max_clips,
                        dynamic_mmr_stop_threshold=dynamic_stop_threshold,
                        dynamic_mmr_extra_clips=DYNAMIC_MMR_EXTRA_CLIPS,
                        dynamic_mmr_trace_path=DYNAMIC_MMR_TRACE_PATH,
                        dynamic_mmr_log_scores=DYNAMIC_MMR_LOG_SCORES,
                        dynamic_mmr_policy=dynamic_policy,
                        dynamic_mmr_confidence_threshold=DYNAMIC_MMR_CONFIDENCE_THRESHOLD,
                        dynamic_mmr_ambiguity_gap_threshold=DYNAMIC_MMR_AMBIGUITY_GAP_THRESHOLD,
                        dynamic_mmr_knee_min_drop=DYNAMIC_MMR_KNEE_MIN_DROP,
                        dynamic_mmr_knee_alpha=DYNAMIC_MMR_KNEE_ALPHA,
                        dynamic_mmr_uncertainty_alpha=DYNAMIC_MMR_UNCERTAINTY_ALPHA,
                        dynamic_mmr_score_source=DYNAMIC_MMR_SCORE_SOURCE,
                        full_adaptive_k_retrieval=FULL_ADAPTIVE_K_RETRIEVAL,
                        full_adaptive_k_strategy=FULL_ADAPTIVE_K_STRATEGY,
                        full_adaptive_k_ignore_extreme=FULL_ADAPTIVE_K_IGNORE_EXTREME,
                        full_adaptive_k_ignore_extreme_tail=FULL_ADAPTIVE_K_IGNORE_EXTREME_TAIL,
                        full_adaptive_k_ignore_below_median=FULL_ADAPTIVE_K_IGNORE_BELOW_MEDIAN,
                        full_adaptive_k_retrieve_more=FULL_ADAPTIVE_K_RETRIEVE_MORE,
                        full_adaptive_k_candidate_nodes=FULL_ADAPTIVE_K_CANDIDATE_NODES,
                        full_adaptive_k_min_nodes=FULL_ADAPTIVE_K_MIN_NODES,
                        full_adaptive_k_max_nodes=FULL_ADAPTIVE_K_MAX_NODES,
                        full_adaptive_k_min_clips=FULL_ADAPTIVE_K_MIN_CLIPS,
                        full_adaptive_k_max_clips=FULL_ADAPTIVE_K_MAX_CLIPS,
                        full_adaptive_k_extra_clips=FULL_ADAPTIVE_K_EXTRA_CLIPS,
                        clip_adaptive_k_retrieval=CLIP_ADAPTIVE_K_RETRIEVAL,
                        clip_adaptive_k_strategy=CLIP_ADAPTIVE_K_STRATEGY,
                        clip_adaptive_k_ignore_extreme=CLIP_ADAPTIVE_K_IGNORE_EXTREME,
                        clip_adaptive_k_ignore_extreme_tail=CLIP_ADAPTIVE_K_IGNORE_EXTREME_TAIL,
                        clip_adaptive_k_ignore_below_median=CLIP_ADAPTIVE_K_IGNORE_BELOW_MEDIAN,
                        clip_adaptive_k_retrieve_more=CLIP_ADAPTIVE_K_RETRIEVE_MORE,
                        clip_adaptive_k_min_clips=CLIP_ADAPTIVE_K_MIN_CLIPS,
                        clip_adaptive_k_max_clips=CLIP_ADAPTIVE_K_MAX_CLIPS,
                        clip_adaptive_k_extra_clips=CLIP_ADAPTIVE_K_EXTRA_CLIPS,
                        clip_adaptive_k_score_source=CLIP_ADAPTIVE_K_SCORE_SOURCE,
                        dynamicrag_clip_retrieval=DYNAMICRAG_CLIP_RETRIEVAL,
                        dynamicrag_model=DYNAMICRAG_MODEL,
                        dynamicrag_api_base=DYNAMICRAG_API_BASE,
                        dynamicrag_api_key=DYNAMICRAG_API_KEY,
                        dynamicrag_temperature=DYNAMICRAG_TEMPERATURE,
                        dynamicrag_max_tokens=DYNAMICRAG_MAX_TOKENS,
                        dynamicrag_timeout=DYNAMICRAG_TIMEOUT,
                        dynamicrag_min_clips=DYNAMICRAG_MIN_CLIPS,
                        dynamicrag_max_clips=DYNAMICRAG_MAX_CLIPS,
                        dynamicrag_max_nodes_per_clip=DYNAMICRAG_MAX_NODES_PER_CLIP,
                        dynamicrag_max_doc_chars=DYNAMICRAG_MAX_DOC_CHARS,
                        df_rag_clip_retrieval=DF_RAG_CLIP_RETRIEVAL,
                        df_rag_model=DF_RAG_MODEL,
                        df_rag_api_base=DF_RAG_API_BASE,
                        df_rag_api_key=DF_RAG_API_KEY,
                        df_rag_temperature=DF_RAG_TEMPERATURE,
                        df_rag_planner_max_tokens=DF_RAG_PLANNER_MAX_TOKENS,
                        df_rag_evaluator_max_tokens=DF_RAG_EVALUATOR_MAX_TOKENS,
                        df_rag_timeout=DF_RAG_TIMEOUT,
                        df_rag_lambdas=DF_RAG_LAMBDAS,
                        df_rag_set_size=DF_RAG_SET_SIZE,
                        df_rag_max_nodes_per_clip=DF_RAG_MAX_NODES_PER_CLIP,
                        df_rag_max_doc_chars=DF_RAG_MAX_DOC_CHARS,
                        df_rag_fallback_lambda=DF_RAG_FALLBACK_LAMBDA,
                        adaptive_rag_retrieval=ADAPTIVE_RAG_RETRIEVAL,
                        adaptive_rag_route_source=ADAPTIVE_RAG_ROUTE_SOURCE,
                        adaptive_rag_classifier_path=ADAPTIVE_RAG_CLASSIFIER_PATH,
                        adaptive_rag_classifier_model=ADAPTIVE_RAG_CLASSIFIER_MODEL,
                        adaptive_rag_classifier_api_base=ADAPTIVE_RAG_CLASSIFIER_API_BASE,
                        adaptive_rag_classifier_api_key=ADAPTIVE_RAG_CLASSIFIER_API_KEY,
                        adaptive_rag_classifier_temperature=ADAPTIVE_RAG_CLASSIFIER_TEMPERATURE,
                        adaptive_rag_classifier_max_tokens=ADAPTIVE_RAG_CLASSIFIER_MAX_TOKENS,
                        adaptive_rag_classifier_timeout=ADAPTIVE_RAG_CLASSIFIER_TIMEOUT,
                        adaptive_rag_fallback_label=ADAPTIVE_RAG_FALLBACK_LABEL,
                        adaptive_rag_question=data["question"],
                        adaptive_rag_question_id=data["id"],
                        adaptive_rag_zero_clips=ADAPTIVE_RAG_ZERO_CLIPS,
                        adaptive_rag_single_clips=ADAPTIVE_RAG_SINGLE_CLIPS,
                        adaptive_rag_multi_clips=ADAPTIVE_RAG_MULTI_CLIPS,
                        adaptive_rag_selector=ADAPTIVE_RAG_SELECTOR,
                        adaptive_rag_score_source=ADAPTIVE_RAG_SCORE_SOURCE,
                        evidence_saturation_retrieval=EVIDENCE_SATURATION_RETRIEVAL,
                        evidence_saturation_min_clips=EVIDENCE_SATURATION_MIN_CLIPS,
                        evidence_saturation_max_clips=EVIDENCE_SATURATION_MAX_CLIPS,
                        evidence_saturation_stop_threshold=EVIDENCE_SATURATION_STOP_THRESHOLD,
                        evidence_saturation_relevance_weight=EVIDENCE_SATURATION_RELEVANCE_WEIGHT,
                        evidence_saturation_semantic_gain_weight=EVIDENCE_SATURATION_SEMANTIC_GAIN_WEIGHT,
                        evidence_saturation_temporal_gain_weight=EVIDENCE_SATURATION_TEMPORAL_GAIN_WEIGHT,
                        evidence_saturation_entity_gain_weight=EVIDENCE_SATURATION_ENTITY_GAIN_WEIGHT,
                        evidence_saturation_action_state_gain_weight=EVIDENCE_SATURATION_ACTION_STATE_GAIN_WEIGHT,
                        evidence_saturation_semantic_redundancy_weight=EVIDENCE_SATURATION_SEMANTIC_REDUNDANCY_WEIGHT,
                        evidence_saturation_temporal_redundancy_weight=EVIDENCE_SATURATION_TEMPORAL_REDUNDANCY_WEIGHT,
                        evidence_saturation_temporal_bucket_size=EVIDENCE_SATURATION_TEMPORAL_BUCKET_SIZE,
                        evidence_saturation_near_clip_window=EVIDENCE_SATURATION_NEAR_CLIP_WINDOW,
                    )
                    new_memories.update(memories)
                else:
                    dynamic_policy, dynamic_min_clips, dynamic_max_clips, dynamic_stop_threshold = dynamic_mmr_params_for_sample(data)
                    memories, currenr_clips, clip_scores = search(
                        mem_node,
                        content,
                        data["currenr_clips"],
                        threshold=float(processing_config.get("retrieval_threshold", 0.5)),
                        topk=processing_config["topk"],
                        before_clip=before_clip,
                        speaker_aware=SPEAKER_AWARE_RETRIEVAL,
                        speaker_bias=SPEAKER_RETRIEVAL_BIAS,
                        speaker_hard_filter=SPEAKER_RETRIEVAL_HARD_FILTER,
                        scene_nodes=scene_nodes_to_pass,
                        scene_rerank_weight=SCENE_RERANK_WEIGHT,
                        diverse_clip_retrieval=DIVERSE_CLIP_RETRIEVAL,
                        fixed_clip_backfill_current=FIXED_CLIP_BACKFILL_CURRENT,
                        diverse_clip_pool_size=DIVERSE_CLIP_POOL_SIZE,
                        diverse_clip_mmr_candidate_pool_size=DIVERSE_CLIP_MMR_CANDIDATE_POOL_SIZE,
                        clip_intra_similarity_threshold=CLIP_INTRA_SIMILARITY_THRESHOLD,
                        clip_mmr_lambda=CLIP_MMR_LAMBDA,
                        clip_max_nodes_for_diversity=CLIP_MAX_NODES_FOR_DIVERSITY,
                        dynamic_mmr_clip_retrieval=DYNAMIC_MMR_CLIP_RETRIEVAL,
                        dynamic_mmr_min_clips=dynamic_min_clips,
                        dynamic_mmr_max_clips=dynamic_max_clips,
                        dynamic_mmr_stop_threshold=dynamic_stop_threshold,
                        dynamic_mmr_extra_clips=DYNAMIC_MMR_EXTRA_CLIPS,
                        dynamic_mmr_trace_path=DYNAMIC_MMR_TRACE_PATH,
                        dynamic_mmr_log_scores=DYNAMIC_MMR_LOG_SCORES,
                        dynamic_mmr_policy=dynamic_policy,
                        dynamic_mmr_confidence_threshold=DYNAMIC_MMR_CONFIDENCE_THRESHOLD,
                        dynamic_mmr_ambiguity_gap_threshold=DYNAMIC_MMR_AMBIGUITY_GAP_THRESHOLD,
                        dynamic_mmr_knee_min_drop=DYNAMIC_MMR_KNEE_MIN_DROP,
                        dynamic_mmr_knee_alpha=DYNAMIC_MMR_KNEE_ALPHA,
                        dynamic_mmr_uncertainty_alpha=DYNAMIC_MMR_UNCERTAINTY_ALPHA,
                        dynamic_mmr_score_source=DYNAMIC_MMR_SCORE_SOURCE,
                        full_adaptive_k_retrieval=FULL_ADAPTIVE_K_RETRIEVAL,
                        full_adaptive_k_strategy=FULL_ADAPTIVE_K_STRATEGY,
                        full_adaptive_k_ignore_extreme=FULL_ADAPTIVE_K_IGNORE_EXTREME,
                        full_adaptive_k_ignore_extreme_tail=FULL_ADAPTIVE_K_IGNORE_EXTREME_TAIL,
                        full_adaptive_k_ignore_below_median=FULL_ADAPTIVE_K_IGNORE_BELOW_MEDIAN,
                        full_adaptive_k_retrieve_more=FULL_ADAPTIVE_K_RETRIEVE_MORE,
                        full_adaptive_k_candidate_nodes=FULL_ADAPTIVE_K_CANDIDATE_NODES,
                        full_adaptive_k_min_nodes=FULL_ADAPTIVE_K_MIN_NODES,
                        full_adaptive_k_max_nodes=FULL_ADAPTIVE_K_MAX_NODES,
                        full_adaptive_k_min_clips=FULL_ADAPTIVE_K_MIN_CLIPS,
                        full_adaptive_k_max_clips=FULL_ADAPTIVE_K_MAX_CLIPS,
                        full_adaptive_k_extra_clips=FULL_ADAPTIVE_K_EXTRA_CLIPS,
                        clip_adaptive_k_retrieval=CLIP_ADAPTIVE_K_RETRIEVAL,
                        clip_adaptive_k_strategy=CLIP_ADAPTIVE_K_STRATEGY,
                        clip_adaptive_k_ignore_extreme=CLIP_ADAPTIVE_K_IGNORE_EXTREME,
                        clip_adaptive_k_ignore_extreme_tail=CLIP_ADAPTIVE_K_IGNORE_EXTREME_TAIL,
                        clip_adaptive_k_ignore_below_median=CLIP_ADAPTIVE_K_IGNORE_BELOW_MEDIAN,
                        clip_adaptive_k_retrieve_more=CLIP_ADAPTIVE_K_RETRIEVE_MORE,
                        clip_adaptive_k_min_clips=CLIP_ADAPTIVE_K_MIN_CLIPS,
                        clip_adaptive_k_max_clips=CLIP_ADAPTIVE_K_MAX_CLIPS,
                        clip_adaptive_k_extra_clips=CLIP_ADAPTIVE_K_EXTRA_CLIPS,
                        clip_adaptive_k_score_source=CLIP_ADAPTIVE_K_SCORE_SOURCE,
                        dynamicrag_clip_retrieval=DYNAMICRAG_CLIP_RETRIEVAL,
                        dynamicrag_model=DYNAMICRAG_MODEL,
                        dynamicrag_api_base=DYNAMICRAG_API_BASE,
                        dynamicrag_api_key=DYNAMICRAG_API_KEY,
                        dynamicrag_temperature=DYNAMICRAG_TEMPERATURE,
                        dynamicrag_max_tokens=DYNAMICRAG_MAX_TOKENS,
                        dynamicrag_timeout=DYNAMICRAG_TIMEOUT,
                        dynamicrag_min_clips=DYNAMICRAG_MIN_CLIPS,
                        dynamicrag_max_clips=DYNAMICRAG_MAX_CLIPS,
                        dynamicrag_max_nodes_per_clip=DYNAMICRAG_MAX_NODES_PER_CLIP,
                        dynamicrag_max_doc_chars=DYNAMICRAG_MAX_DOC_CHARS,
                        df_rag_clip_retrieval=DF_RAG_CLIP_RETRIEVAL,
                        df_rag_model=DF_RAG_MODEL,
                        df_rag_api_base=DF_RAG_API_BASE,
                        df_rag_api_key=DF_RAG_API_KEY,
                        df_rag_temperature=DF_RAG_TEMPERATURE,
                        df_rag_planner_max_tokens=DF_RAG_PLANNER_MAX_TOKENS,
                        df_rag_evaluator_max_tokens=DF_RAG_EVALUATOR_MAX_TOKENS,
                        df_rag_timeout=DF_RAG_TIMEOUT,
                        df_rag_lambdas=DF_RAG_LAMBDAS,
                        df_rag_set_size=DF_RAG_SET_SIZE,
                        df_rag_max_nodes_per_clip=DF_RAG_MAX_NODES_PER_CLIP,
                        df_rag_max_doc_chars=DF_RAG_MAX_DOC_CHARS,
                        df_rag_fallback_lambda=DF_RAG_FALLBACK_LAMBDA,
                        adaptive_rag_retrieval=ADAPTIVE_RAG_RETRIEVAL,
                        adaptive_rag_route_source=ADAPTIVE_RAG_ROUTE_SOURCE,
                        adaptive_rag_classifier_path=ADAPTIVE_RAG_CLASSIFIER_PATH,
                        adaptive_rag_classifier_model=ADAPTIVE_RAG_CLASSIFIER_MODEL,
                        adaptive_rag_classifier_api_base=ADAPTIVE_RAG_CLASSIFIER_API_BASE,
                        adaptive_rag_classifier_api_key=ADAPTIVE_RAG_CLASSIFIER_API_KEY,
                        adaptive_rag_classifier_temperature=ADAPTIVE_RAG_CLASSIFIER_TEMPERATURE,
                        adaptive_rag_classifier_max_tokens=ADAPTIVE_RAG_CLASSIFIER_MAX_TOKENS,
                        adaptive_rag_classifier_timeout=ADAPTIVE_RAG_CLASSIFIER_TIMEOUT,
                        adaptive_rag_fallback_label=ADAPTIVE_RAG_FALLBACK_LABEL,
                        adaptive_rag_question=data["question"],
                        adaptive_rag_question_id=data["id"],
                        adaptive_rag_zero_clips=ADAPTIVE_RAG_ZERO_CLIPS,
                        adaptive_rag_single_clips=ADAPTIVE_RAG_SINGLE_CLIPS,
                        adaptive_rag_multi_clips=ADAPTIVE_RAG_MULTI_CLIPS,
                        adaptive_rag_selector=ADAPTIVE_RAG_SELECTOR,
                        adaptive_rag_score_source=ADAPTIVE_RAG_SCORE_SOURCE,
                        evidence_saturation_retrieval=EVIDENCE_SATURATION_RETRIEVAL,
                        evidence_saturation_min_clips=EVIDENCE_SATURATION_MIN_CLIPS,
                        evidence_saturation_max_clips=EVIDENCE_SATURATION_MAX_CLIPS,
                        evidence_saturation_stop_threshold=EVIDENCE_SATURATION_STOP_THRESHOLD,
                        evidence_saturation_relevance_weight=EVIDENCE_SATURATION_RELEVANCE_WEIGHT,
                        evidence_saturation_semantic_gain_weight=EVIDENCE_SATURATION_SEMANTIC_GAIN_WEIGHT,
                        evidence_saturation_temporal_gain_weight=EVIDENCE_SATURATION_TEMPORAL_GAIN_WEIGHT,
                        evidence_saturation_entity_gain_weight=EVIDENCE_SATURATION_ENTITY_GAIN_WEIGHT,
                        evidence_saturation_action_state_gain_weight=EVIDENCE_SATURATION_ACTION_STATE_GAIN_WEIGHT,
                        evidence_saturation_semantic_redundancy_weight=EVIDENCE_SATURATION_SEMANTIC_REDUNDANCY_WEIGHT,
                        evidence_saturation_temporal_redundancy_weight=EVIDENCE_SATURATION_TEMPORAL_REDUNDANCY_WEIGHT,
                        evidence_saturation_temporal_bucket_size=EVIDENCE_SATURATION_TEMPORAL_BUCKET_SIZE,
                        evidence_saturation_near_clip_window=EVIDENCE_SATURATION_NEAR_CLIP_WINDOW,
                        role_aware_clip_retrieval=ROLE_AWARE_CLIP_RETRIEVAL,
                        role_aware_question=data["question"],
                        role_aware_model=ROLE_AWARE_MODEL,
                        role_aware_model_device=ROLE_AWARE_MODEL_DEVICE,
                        role_aware_max_new_tokens=ROLE_AWARE_MAX_NEW_TOKENS,
                        role_aware_cache_dir=ROLE_AWARE_CACHE_DIR,
                        role_aware_precomputed_dir=ROLE_AWARE_PRECOMPUTED_DIR,
                        role_aware_max_nodes_per_clip=ROLE_AWARE_MAX_NODES_PER_CLIP,
                        role_aware_question_role_mode=ROLE_AWARE_QUESTION_ROLE_MODE,
                        role_aware_question_roles_dir=ROLE_AWARE_QUESTION_ROLES_DIR,
                        role_aware_role_match_weight=ROLE_AWARE_ROLE_MATCH_WEIGHT,
                        role_aware_relevance_weight=ROLE_AWARE_RELEVANCE_WEIGHT,
                        role_aware_coverage_weight=ROLE_AWARE_COVERAGE_WEIGHT,
                        role_aware_instance_weight=ROLE_AWARE_INSTANCE_WEIGHT,
                        role_aware_role_redundancy_weight=ROLE_AWARE_ROLE_REDUNDANCY_WEIGHT,
                        role_aware_semantic_redundancy_weight=ROLE_AWARE_SEMANTIC_REDUNDANCY_WEIGHT,
                        role_aware_secondary_role_weight=ROLE_AWARE_SECONDARY_ROLE_WEIGHT,
                        role_aware_soft_query_prior=ROLE_AWARE_SOFT_QUERY_PRIOR,
                        role_aware_query_coverage_weight=ROLE_AWARE_QUERY_COVERAGE_WEIGHT,
                        role_aware_gate_semantic_redundancy=ROLE_AWARE_GATE_SEMANTIC_REDUNDANCY,
                        role_aware_fix_first_relevance=ROLE_AWARE_FIX_FIRST_RELEVANCE,
                    )
                    data["currenr_clips"] = currenr_clips
                    if ROBOT_DEV_MODE and memories:
                        memories = augment_robot_dev_memories(
                            mem_node,
                            data["mem_path"],
                            memories,
                            context_root=ROBOT_DEV_CONTEXT_ROOT,
                            detail_root=ROBOT_DEV_DETAIL_ROOT,
                            vl_model_path=ROBOT_DEV_VL_MODEL_PATH,
                            vl_model_device=ROBOT_DEV_VL_DEVICE,
                            force_regen=ROBOT_DEV_FORCE_REGEN,
                            faces_input=ROBOT_DEV_FACES_INPUT,
                            merge_mode=ROBOT_DEV_MERGE_MODE,
                            max_detail_items=ROBOT_DEV_MAX_DETAIL_ITEMS,
                            max_new_tokens=ROBOT_DEV_MAX_NEW_TOKENS,
                        )
                    new_memories.update(memories)

            search_result = ""
            if SCENE_BACKGROUND and scene_nodes_text_dict and clip_scores:
                best_clip = max(clip_scores, key=clip_scores.get)
                # Find the scene node that covers this clip (scene node covers backward 6 clips)
                # We want a scene_clip such that scene_clip >= best_clip and scene_clip <= best_clip + 5
                valid_scene_clips = [t for t in scene_nodes_text_dict.keys() if t >= best_clip and t <= best_clip + 5]
                if not valid_scene_clips:
                    valid_scene_clips = [t for t in scene_nodes_text_dict.keys() if t >= best_clip]
                if valid_scene_clips:
                    best_scene_t = min(valid_scene_clips)
                    background_text = scene_nodes_text_dict[best_scene_t]
                    search_result += f"[Background Context about clip {best_clip}]\n{background_text}\n\n[Detailed Retrieval Results]\n"

            search_result += "Searched knowledge: " + json.dumps(new_memories, ensure_ascii=False).encode("utf-8", "ignore").decode("utf-8")
            if len(new_memories) == 0:
                search_result += "\n(The search result is empty. Please try searching from another perspective.)"
            data["conversations"].append({"role": "user", "content": search_result})
    return data


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--data_file", type=str, default="data/annotations/robot.json")
    parser.add_argument("--list_file", type=str, default=None, help="Optional file with video IDs (one per line) to evaluate")
    parser.add_argument(
        "--question_ids_file",
        type=str,
        default=None,
        help="Optional file with question IDs (one per line) to evaluate within the selected videos.",
    )
    parser.add_argument("--tensor_parallel_size", type=int, default=2, help="Number of GPUs for vLLM (default: 2)")
    parser.add_argument(
        "--gpu_memory_utilization",
        type=float,
        default=float(os.getenv("M3AGENT_GPU_MEMORY_UTILIZATION", "0.90")),
        help="vLLM GPU memory utilization for the M3-Agent control model.",
    )
    parser.add_argument(
        "--max_model_len",
        type=int,
        default=int(os.getenv("M3AGENT_MAX_MODEL_LEN", "0")),
        help="Optional vLLM max model length for M3-Agent control. 0 keeps the model default.",
    )
    parser.add_argument(
        "--disable_custom_all_reduce",
        action="store_true",
        help="Disable vLLM custom all-reduce kernels; useful on GPU nodes where the custom kernel fails during TP startup.",
    )
    parser.add_argument("--output_name", type=str, default=None, help="Optional output filename (without extension) for results")
    parser.add_argument(
        "--equivalence_mode",
        type=str,
        default="default",
        choices=["default", "dev", "dev_one_to_one", "none", "no_equiv"],
        help="Equivalence refresh mode before each retrieval round.",
    )
    parser.add_argument(
        "--dev_voice_equiv_threshold",
        type=float,
        default=0.5,
        help="Voice similarity threshold used when equivalence_mode=dev.",
    )
    parser.add_argument(
        "--dev_voice_cluster_threshold",
        type=float,
        default=0.65,
        help="Post-hoc voice clustering threshold used when equivalence_mode=dev.",
    )
    parser.add_argument(
        "--speaker_aware_retrieval",
        action="store_true",
        help="Enable speaker-aware retrieval bias (audio-only experiments).",
    )
    parser.add_argument(
        "--identity_hint_mode",
        type=str,
        default="off",
        choices=["off", "basic"],
        help="Optional pre-search name/character hint injection. Default keeps original control behavior.",
    )
    parser.add_argument(
        "--topk",
        type=int,
        default=None,
        help="Override clip retrieval top-k for normal search.",
    )
    parser.add_argument(
        "--retrieval_threshold",
        type=float,
        default=float("-inf"),
        help=(
            "Minimum clip score kept by retrieval. Default disables the hard score "
            "threshold; set 0.5 to reproduce the old filtered behavior."
        ),
    )
    parser.add_argument(
        "--batch_size",
        type=int,
        default=None,
        help="Override evaluation batch size from configs/processing_config.json.",
    )
    parser.add_argument(
        "--consumer_workers",
        type=int,
        default=None,
        help="Override retrieval consumer worker count. Use 0 for serial execution.",
    )
    parser.add_argument(
        "--speaker_retrieval_bias",
        type=float,
        default=0.35,
        help="Score multiplier bias for nodes connected to inferred speaker nodes.",
    )
    parser.add_argument(
        "--speaker_retrieval_hard_filter",
        action="store_true",
        help="Filter out nodes not connected to inferred speaker nodes when available.",
    )
    parser.add_argument(
        "--strip_temporal_edges",
        action="store_true",
        help="Remove voice-voice temporal dialogue edges (weight=0.5) from loaded graphs.",
    )
    parser.add_argument(
        "--strip_scene_nodes",
        action="store_true",
        help="Remove Scene Node edges (weight=0.6) from loaded graphs.",
    )
    parser.add_argument(
        "--scene_rerank",
        action="store_true",
        help="Use Scene Nodes as weak reranking signal (strip edges but boost clip scores).",
    )
    parser.add_argument(
        "--scene_rerank_weight",
        type=float,
        default=0.14,
        help="Weight for scene node reranking bonus (default: 0.14).",
    )
    parser.add_argument(
        "--scene_background",
        action="store_true",
        help="Prepend scene node summary for the best-matched clip as background context.",
    )
    parser.add_argument(
        "--diverse_clip_retrieval",
        action="store_true",
        help="Use clip-internal deduplication plus MMR-style diverse clip selection.",
    )
    parser.add_argument(
        "--fixed_clip_backfill_current",
        action="store_true",
        help=(
            "For the original non-diverse fixed top-k flow, skip clips already in "
            "current_clips before slicing top-k so retrieval can backfill later candidates."
        ),
    )
    parser.add_argument(
        "--diverse_clip_pool_size",
        type=int,
        default=12,
        help="Final selector pool size for diverse clip retrieval.",
    )
    parser.add_argument(
        "--diverse_clip_mmr_candidate_pool_size",
        type=int,
        default=200,
        help=(
            "Number of relevance-ranked clips to feed into MMR before the final selector "
            "pool is produced. 0 reproduces the old behavior and uses "
            "--diverse_clip_pool_size only."
        ),
    )
    parser.add_argument(
        "--clip_intra_similarity_threshold",
        type=float,
        default=0.85,
        help="Similarity threshold for grouping near-duplicate text nodes within a clip.",
    )
    parser.add_argument(
        "--clip_mmr_lambda",
        type=float,
        default=0.75,
        help="Relevance/diversity tradeoff for final clip MMR selection.",
    )
    parser.add_argument(
        "--clip_max_nodes_for_diversity",
        type=int,
        default=8,
        help="Maximum scored nodes per clip used to build diversity-aware clip representations.",
    )
    parser.add_argument(
        "--dynamic_mmr_clip_retrieval",
        action="store_true",
        help="Use the old diverse MMR selector with dynamic top-k early stopping.",
    )
    parser.add_argument(
        "--dynamic_mmr_min_clips",
        type=int,
        default=2,
        help="Minimum clips returned by dynamic MMR before early stopping may trigger.",
    )
    parser.add_argument(
        "--dynamic_mmr_max_clips",
        type=int,
        default=5,
        help="Maximum clips returned by dynamic MMR.",
    )
    parser.add_argument(
        "--dynamic_mmr_stop_threshold",
        type=float,
        default=0.05,
        help="Stop dynamic MMR once the best next MMR objective falls below this threshold.",
    )
    parser.add_argument(
        "--dynamic_mmr_extra_clips",
        type=int,
        default=0,
        help="Append this many additional MMR-ranked clips after the dynamic-MMR stopping decision.",
    )
    parser.add_argument(
        "--dynamic_mmr_trace_path",
        type=str,
        default=None,
        help="Optional JSONL path recording dynamic MMR internal scores for threshold analysis.",
    )
    parser.add_argument(
        "--dynamic_mmr_log_scores",
        action="store_true",
        help="Print compact per-retrieval dynamic MMR scores to the runtime log.",
    )
    parser.add_argument(
        "--dynamic_mmr_score_source",
        type=str,
        default="clip_score",
        choices=["clip_score", "max_node"],
        help="Clip relevance source for dynamic MMR. clip_score uses our clip aggregation; max_node uses the original best node relevance per clip.",
    )
    parser.add_argument(
        "--dynamic_mmr_policy",
        type=str,
        default="threshold",
        choices=[
            "threshold",
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
        ],
        help="Dynamic MMR K policy. 'threshold' uses a fixed score threshold; 'uncertainty' expands K when marginal gains are low or close; 'adaptive_gap_uncertainty' increases K while adjacent MMR gains are unusually close under the current query's gap distribution; 'soft_adjacent_uncertainty' converts adjacent-gap closeness into a soft adaptive K; 'prefix_plateau' keeps the adjacent-MMR prefix while gaps stay within the query's gap median; 'largest_gap_cut' and 'two_segment_change' estimate an evidence/tail boundary from the MMR curve; 'budgeted_utility' keeps clips above the current query's average marginal utility; 'utility_mass' and 'softmax_mass' return the smallest prefix covering a target evidence utility mass; 'robust_gap_boundary' stops at a median+MAD adjacent-gap boundary; 'knee' stops at a marginal-gain drop; 'adaptive_knee' derives that drop threshold from the current candidate pool; 'adaptive_uncertainty' derives uncertainty from adjacent drop statistics; 'bic_boundary' selects K by BIC model selection over the MMR gain curve; 'official_adaptive_k_*' calls the cloned official Adaptive-k Retrieval repository for dynamic context selection.",
    )
    parser.add_argument(
        "--dynamic_mmr_confidence_threshold",
        type=float,
        default=0.30,
        help="For uncertainty policy, add evidence when the second MMR gain is below this confidence threshold.",
    )
    parser.add_argument(
        "--dynamic_mmr_ambiguity_gap_threshold",
        type=float,
        default=0.25,
        help="For uncertainty policy, add evidence when adjacent MMR gains have a relative gap below this threshold.",
    )
    parser.add_argument(
        "--dynamic_mmr_knee_min_drop",
        type=float,
        default=0.25,
        help="For knee policy, stop at the largest adjacent marginal-gain drop if it is at least this relative size.",
    )
    parser.add_argument(
        "--dynamic_mmr_knee_alpha",
        type=float,
        default=1.0,
        help="For adaptive_knee policy, require max drop >= mean(drop) + alpha * std(drop).",
    )
    parser.add_argument(
        "--dynamic_mmr_uncertainty_alpha",
        type=float,
        default=1.0,
        help="For adaptive_uncertainty policy, scale the per-query drop distribution thresholds.",
    )
    parser.add_argument(
        "--full_adaptive_k_retrieval",
        action="store_true",
        help="Use the official Adaptive-k threshold on the full node-level similarity curve before mapping selected nodes to clips.",
    )
    parser.add_argument(
        "--full_adaptive_k_strategy",
        type=str,
        default="largest_gap",
        choices=["largest_gap", "moving_avg", "2diff_spike"],
        help="Official Adaptive-k threshold strategy for full node-level retrieval.",
    )
    parser.add_argument(
        "--full_adaptive_k_ignore_extreme",
        type=float,
        default=0.0,
        help="Fraction/int head range ignored by official Adaptive-k when searching for the boundary.",
    )
    parser.add_argument(
        "--full_adaptive_k_ignore_extreme_tail",
        type=float,
        default=0.1,
        help="Fraction/int tail range ignored by official Adaptive-k when searching for the boundary.",
    )
    parser.add_argument(
        "--full_adaptive_k_ignore_below_median",
        action="store_true",
        help="Use official Adaptive-k's median-tail ignore mode.",
    )
    parser.add_argument(
        "--full_adaptive_k_retrieve_more",
        type=str,
        default="5",
        help="Official Adaptive-k buffer: add this many nodes after the detected boundary; integer-like values add nodes.",
    )
    parser.add_argument(
        "--full_adaptive_k_candidate_nodes",
        type=int,
        default=0,
        help="Optional top-N node candidates passed into official Adaptive-k; 0 means use the full node curve.",
    )
    parser.add_argument(
        "--full_adaptive_k_min_nodes",
        type=int,
        default=1,
        help="Minimum node chunks selected by full Adaptive-k before clip mapping.",
    )
    parser.add_argument(
        "--full_adaptive_k_max_nodes",
        type=int,
        default=0,
        help="Optional maximum node chunks selected by full Adaptive-k; 0 means no node cap.",
    )
    parser.add_argument(
        "--full_adaptive_k_min_clips",
        type=int,
        default=0,
        help="Optional minimum unique clips after selected nodes are mapped to clips; 0 means no clip minimum.",
    )
    parser.add_argument(
        "--full_adaptive_k_max_clips",
        type=int,
        default=0,
        help="Optional safety cap after selected nodes are mapped to unique clips; 0 means no clip cap.",
    )
    parser.add_argument(
        "--full_adaptive_k_extra_clips",
        type=int,
        default=0,
        help="Append this many additional node-ranked unique clips after full Adaptive-k clip mapping.",
    )
    parser.add_argument(
        "--clip_adaptive_k_retrieval",
        action="store_true",
        help="Run official Adaptive-k cutoff on the top clip-candidate score curve instead of MMR/dynamic MMR.",
    )
    parser.add_argument(
        "--clip_adaptive_k_strategy",
        type=str,
        default="largest_gap",
        choices=["largest_gap", "moving_avg", "2diff_spike"],
        help="Official Adaptive-k threshold strategy for top clip candidates.",
    )
    parser.add_argument(
        "--clip_adaptive_k_ignore_extreme",
        type=float,
        default=0.0,
        help="Fraction/int head range ignored by clip-level Adaptive-k when searching for the boundary.",
    )
    parser.add_argument(
        "--clip_adaptive_k_ignore_extreme_tail",
        type=float,
        default=0.1,
        help="Fraction/int tail range ignored by clip-level Adaptive-k when searching for the boundary.",
    )
    parser.add_argument(
        "--clip_adaptive_k_ignore_below_median",
        action="store_true",
        help="Use official Adaptive-k's median-tail ignore mode for top clip candidates.",
    )
    parser.add_argument(
        "--clip_adaptive_k_retrieve_more",
        type=str,
        default="5",
        help="Official Adaptive-k buffer for top clip candidates; integer-like values add clips.",
    )
    parser.add_argument(
        "--clip_adaptive_k_min_clips",
        type=int,
        default=1,
        help="Minimum clips returned by clip-level Adaptive-k.",
    )
    parser.add_argument(
        "--clip_adaptive_k_max_clips",
        type=int,
        default=0,
        help="Optional safety cap for clip-level Adaptive-k; 0 means no cap.",
    )
    parser.add_argument(
        "--clip_adaptive_k_extra_clips",
        type=int,
        default=0,
        help="Append this many additional ranked clip candidates after clip-level Adaptive-k, still capped by max clips.",
    )
    parser.add_argument(
        "--clip_adaptive_k_score_source",
        type=str,
        default="max_node",
        choices=["clip_score", "max_node"],
        help="Score curve for clip-level Adaptive-k: existing clip aggregate score or max node relevance projected to each unique clip.",
    )
    parser.add_argument(
        "--dynamicrag_clip_retrieval",
        action="store_true",
        help="Use the official DynamicRAG document-id generation prompt/model to select final clips from the diverse candidate pool.",
    )
    parser.add_argument(
        "--dynamicrag_model",
        type=str,
        default="gasolsun/DynamicRAG-8B",
        help="DynamicRAG model name served by the OpenAI-compatible sidecar, e.g. gasolsun/DynamicRAG-8B.",
    )
    parser.add_argument(
        "--dynamicrag_api_base",
        type=str,
        default=os.getenv("DYNAMICRAG_API_BASE"),
        help="OpenAI-compatible API base for the DynamicRAG sidecar, e.g. http://127.0.0.1:8123/v1.",
    )
    parser.add_argument(
        "--dynamicrag_api_key",
        type=str,
        default=os.getenv("DYNAMICRAG_API_KEY", "EMPTY"),
        help="API key for the DynamicRAG sidecar. vLLM local servers usually accept EMPTY.",
    )
    parser.add_argument(
        "--dynamicrag_temperature",
        type=float,
        default=0.4,
        help="Sampling temperature for DynamicRAG's document identifier generation.",
    )
    parser.add_argument(
        "--dynamicrag_max_tokens",
        type=int,
        default=100,
        help="Maximum generated tokens for DynamicRAG's document identifier generation.",
    )
    parser.add_argument(
        "--dynamicrag_timeout",
        type=float,
        default=60,
        help="HTTP timeout in seconds for one DynamicRAG sidecar request.",
    )
    parser.add_argument(
        "--dynamicrag_min_clips",
        type=int,
        default=0,
        help="Optional minimum selected clips after DynamicRAG output. 0 keeps official no-min behavior.",
    )
    parser.add_argument(
        "--dynamicrag_max_clips",
        type=int,
        default=0,
        help="Optional safety cap after DynamicRAG output. 0 keeps candidate-pool-only cap.",
    )
    parser.add_argument(
        "--dynamicrag_max_nodes_per_clip",
        type=int,
        default=4,
        help="Maximum memory text nodes summarized into each candidate clip document for DynamicRAG. <=0 uses all text nodes in the clip.",
    )
    parser.add_argument(
        "--dynamicrag_max_doc_chars",
        type=int,
        default=1600,
        help="Maximum characters per candidate clip document sent to DynamicRAG.",
    )
    parser.add_argument(
        "--df_rag_clip_retrieval",
        action="store_true",
        help="Use a DF-RAG-style query-aware diversity selector: planner, gMMR lambda sweep, evaluator, upper-median tie break.",
    )
    parser.add_argument(
        "--df_rag_model",
        type=str,
        default=os.getenv("DF_RAG_MODEL", "models/DynamicRAG-8B"),
        help="OpenAI-compatible model name for the DF-RAG planner/evaluator sidecar.",
    )
    parser.add_argument(
        "--df_rag_api_base",
        type=str,
        default=os.getenv("DF_RAG_API_BASE"),
        help="OpenAI-compatible API base for the DF-RAG planner/evaluator sidecar.",
    )
    parser.add_argument(
        "--df_rag_api_key",
        type=str,
        default=os.getenv("DF_RAG_API_KEY", "EMPTY"),
        help="API key for the DF-RAG sidecar. vLLM local servers usually accept EMPTY.",
    )
    parser.add_argument(
        "--df_rag_temperature",
        type=float,
        default=0.0,
        help="Sampling temperature for DF-RAG planner/evaluator calls.",
    )
    parser.add_argument(
        "--df_rag_planner_max_tokens",
        type=int,
        default=256,
        help="Maximum generated tokens for the DF-RAG Planner.",
    )
    parser.add_argument(
        "--df_rag_evaluator_max_tokens",
        type=int,
        default=512,
        help="Maximum generated tokens for each DF-RAG Evaluator call.",
    )
    parser.add_argument(
        "--df_rag_timeout",
        type=float,
        default=60,
        help="HTTP timeout in seconds for one DF-RAG planner/evaluator request.",
    )
    parser.add_argument(
        "--df_rag_lambdas",
        type=str,
        default="0.1,0.2,0.3,0.4,0.5,0.6,0.7,0.8,0.9,1.0",
        help="Comma-separated DF-RAG lambda grid. Default follows the paper's 0.1 step sweep and excludes 0.",
    )
    parser.add_argument(
        "--df_rag_set_size",
        type=int,
        default=5,
        help="Fixed number of clips in each lambda-specific gMMR candidate set. The paper retrieves 5 chunks.",
    )
    parser.add_argument(
        "--df_rag_max_nodes_per_clip",
        type=int,
        default=4,
        help="Maximum memory text nodes summarized into each candidate clip document for DF-RAG. <=0 uses all text nodes in the clip.",
    )
    parser.add_argument(
        "--df_rag_max_doc_chars",
        type=int,
        default=1600,
        help="Maximum characters per candidate clip document sent to the DF-RAG evaluator.",
    )
    parser.add_argument(
        "--df_rag_fallback_lambda",
        type=float,
        default=0.5,
        help="Fallback gMMR lambda if no evaluator result can be parsed.",
    )
    parser.add_argument(
        "--adaptive_rag_retrieval",
        action="store_true",
        help="Use starsuzi/Adaptive-RAG-style query-complexity routing before selecting clips.",
    )
    parser.add_argument(
        "--adaptive_rag_route_source",
        type=str,
        default=os.getenv("ADAPTIVE_RAG_ROUTE_SOURCE", "heuristic"),
        choices=["heuristic", "file", "classifier_file", "precomputed", "api", "llm", "openai", "constant", "fixed"],
        help="Source for Adaptive-RAG A/B/C route labels. file/precomputed reuses official classifier outputs.",
    )
    parser.add_argument(
        "--adaptive_rag_classifier_path",
        type=str,
        default=os.getenv("ADAPTIVE_RAG_CLASSIFIER_PATH"),
        help="JSON/JSONL with official Adaptive-RAG classifier labels keyed by question id or question text.",
    )
    parser.add_argument(
        "--adaptive_rag_classifier_model",
        type=str,
        default=os.getenv("ADAPTIVE_RAG_CLASSIFIER_MODEL", "adaptive-rag-classifier"),
        help="OpenAI-compatible classifier model name when --adaptive_rag_route_source is api/llm/openai.",
    )
    parser.add_argument(
        "--adaptive_rag_classifier_api_base",
        type=str,
        default=os.getenv("ADAPTIVE_RAG_CLASSIFIER_API_BASE"),
        help="OpenAI-compatible API base for an Adaptive-RAG classifier sidecar.",
    )
    parser.add_argument(
        "--adaptive_rag_classifier_api_key",
        type=str,
        default=os.getenv("ADAPTIVE_RAG_CLASSIFIER_API_KEY", "EMPTY"),
        help="API key for Adaptive-RAG classifier sidecar. Local vLLM usually accepts EMPTY.",
    )
    parser.add_argument(
        "--adaptive_rag_classifier_temperature",
        type=float,
        default=0.0,
        help="Sampling temperature for Adaptive-RAG API classifier.",
    )
    parser.add_argument(
        "--adaptive_rag_classifier_max_tokens",
        type=int,
        default=4,
        help="Maximum generated tokens for Adaptive-RAG API classifier.",
    )
    parser.add_argument(
        "--adaptive_rag_classifier_timeout",
        type=float,
        default=60,
        help="HTTP timeout in seconds for Adaptive-RAG API classifier.",
    )
    parser.add_argument(
        "--adaptive_rag_fallback_label",
        type=str,
        default="B",
        choices=["A", "B", "C", "zero", "single", "multi"],
        help="Fallback Adaptive-RAG route if classifier output is missing/unparseable.",
    )
    parser.add_argument(
        "--adaptive_rag_zero_clips",
        type=int,
        default=0,
        help="Clip count for Adaptive-RAG route A. Default preserves the original no-retrieval route.",
    )
    parser.add_argument(
        "--adaptive_rag_single_clips",
        type=int,
        default=2,
        help="Clip count for Adaptive-RAG route B, corresponding to single-step retrieval.",
    )
    parser.add_argument(
        "--adaptive_rag_multi_clips",
        type=int,
        default=5,
        help="Clip count for Adaptive-RAG route C, corresponding to multi-step retrieval.",
    )
    parser.add_argument(
        "--adaptive_rag_selector",
        type=str,
        default="top",
        choices=["mmr", "top"],
        help="Clip selector inside each Adaptive-RAG route. top uses ordinary relevance ranking; mmr uses M3Agent's diverse clip selector.",
    )
    parser.add_argument(
        "--adaptive_rag_score_source",
        type=str,
        default="max_node",
        choices=["clip_score", "max_node"],
        help="Clip relevance source for Adaptive-RAG top selection. max_node uses the original node-relevance ordering projected to unique clips.",
    )
    parser.add_argument(
        "--dynamic_mmr_type_switch",
        action="store_true",
        help="Switch dynamic-MMR policy for samples whose annotation type matches --dynamic_mmr_type_switch_types.",
    )
    parser.add_argument(
        "--dynamic_mmr_type_switch_types",
        type=str,
        default="Cross-Modal Reasoning,Multi-Hop Reasoning",
        help="Comma-separated annotation types that should use the switch policy.",
    )
    parser.add_argument(
        "--dynamic_mmr_type_switch_policy",
        type=str,
        default="threshold",
        choices=[
            "threshold",
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
        ],
        help="Dynamic-MMR policy to use for matching annotation types.",
    )
    parser.add_argument(
        "--dynamic_mmr_type_switch_stop_threshold",
        type=float,
        default=0.20,
        help="Stop threshold used by the type-switch dynamic-MMR policy.",
    )
    parser.add_argument(
        "--dynamic_mmr_type_switch_min_clips",
        type=int,
        default=2,
        help="Minimum clips used by the type-switch dynamic-MMR policy.",
    )
    parser.add_argument(
        "--dynamic_mmr_type_switch_max_clips",
        type=int,
        default=5,
        help="Maximum clips used by the type-switch dynamic-MMR policy.",
    )
    parser.add_argument(
        "--evidence_saturation_retrieval",
        action="store_true",
        help="Use memory evidence saturation to dynamically select the number of final clips from the diverse candidate pool.",
    )
    parser.add_argument(
        "--evidence_saturation_min_clips",
        type=int,
        default=1,
        help="Minimum clips returned before evidence-saturation early stopping may trigger.",
    )
    parser.add_argument(
        "--evidence_saturation_max_clips",
        type=int,
        default=5,
        help="Maximum clips returned by evidence-saturation retrieval.",
    )
    parser.add_argument(
        "--evidence_saturation_stop_threshold",
        type=float,
        default=0.02,
        help="Stop adding clips once marginal non-redundant evidence gain falls below this threshold.",
    )
    parser.add_argument(
        "--evidence_saturation_relevance_weight",
        type=float,
        default=0.75,
        help="Weight for original clip relevance in evidence-saturation selection.",
    )
    parser.add_argument(
        "--evidence_saturation_semantic_gain_weight",
        type=float,
        default=0.10,
        help="Reward for adding new memory-node semantic clusters.",
    )
    parser.add_argument(
        "--evidence_saturation_temporal_gain_weight",
        type=float,
        default=0.08,
        help="Reward for covering a new temporal bucket.",
    )
    parser.add_argument(
        "--evidence_saturation_entity_gain_weight",
        type=float,
        default=0.06,
        help="Reward for adding new entity/person/object tokens.",
    )
    parser.add_argument(
        "--evidence_saturation_action_state_gain_weight",
        type=float,
        default=0.06,
        help="Reward for adding new action/state tokens.",
    )
    parser.add_argument(
        "--evidence_saturation_semantic_redundancy_weight",
        type=float,
        default=0.15,
        help="Penalty for embedding-level redundancy with already selected clips.",
    )
    parser.add_argument(
        "--evidence_saturation_temporal_redundancy_weight",
        type=float,
        default=0.0,
        help="Penalty for nearby clips with high lexical evidence overlap.",
    )
    parser.add_argument(
        "--evidence_saturation_temporal_bucket_size",
        type=int,
        default=4,
        help="Clip count per temporal bucket for temporal coverage gain.",
    )
    parser.add_argument(
        "--evidence_saturation_near_clip_window",
        type=int,
        default=2,
        help="Clip-distance window used for temporal redundancy.",
    )
    parser.add_argument(
        "--role_aware_clip_retrieval",
        action="store_true",
        help="Use question-conditioned evidence-role coverage to select final clips from the diverse candidate pool.",
    )
    parser.add_argument(
        "--role_aware_model",
        type=str,
        default="local-qwen3-vl",
        help="Model used to label evidence roles. Use local-qwen3-vl / qwen3-vl-8b / a local path to avoid API calls.",
    )
    parser.add_argument(
        "--role_aware_model_device",
        type=str,
        default=None,
        help="Optional device for local role labeler, e.g. cuda:1. Also honors ROLE_AWARE_MODEL_DEVICE.",
    )
    parser.add_argument(
        "--role_aware_max_new_tokens",
        type=int,
        default=2048,
        help="Maximum new tokens for local/API evidence-role JSON labeling.",
    )
    parser.add_argument(
        "--role_aware_cache_dir",
        type=str,
        default=None,
        help="Optional directory for per-query evidence-role JSON cache files.",
    )
    parser.add_argument(
        "--role_aware_precomputed_dir",
        type=str,
        default=None,
        help="Directory with offline clip role labels, e.g. data/role_evidence/robot. Skips online role labeling when available.",
    )
    parser.add_argument(
        "--role_aware_max_nodes_per_clip",
        type=int,
        default=4,
        help="Maximum memory snippets per candidate clip sent to the role-aware labeler.",
    )
    parser.add_argument(
        "--role_aware_question_role_mode",
        type=str,
        default="heuristic",
        choices=["heuristic", "qwen_binary"],
        help="How to infer evidence roles needed by the question. qwen_binary uses binary local/precomputed Qwen labels.",
    )
    parser.add_argument(
        "--role_aware_question_roles_dir",
        type=str,
        default=None,
        help="Optional directory with precomputed question-role JSON files keyed by question hash.",
    )
    parser.add_argument(
        "--role_aware_role_match_weight",
        type=float,
        default=0.0,
        help="Bonus for clips that contain question-needed evidence roles, independent of whether the role was already covered.",
    )
    parser.add_argument(
        "--role_aware_relevance_weight",
        type=float,
        default=0.55,
        help="Weight for original clip relevance in role-aware final selection.",
    )
    parser.add_argument(
        "--role_aware_coverage_weight",
        type=float,
        default=0.30,
        help="Weight for covering question-needed evidence roles.",
    )
    parser.add_argument(
        "--role_aware_instance_weight",
        type=float,
        default=0.25,
        help="Weight for adding distinct evidence units within needed roles.",
    )
    parser.add_argument(
        "--role_aware_role_redundancy_weight",
        type=float,
        default=0.0,
        help="Deprecated compatibility flag; role-level redundancy is no longer used in role-aware selection.",
    )
    parser.add_argument(
        "--role_aware_semantic_redundancy_weight",
        type=float,
        default=0.10,
        help="Penalty for embedding-level redundancy with selected clips.",
    )
    parser.add_argument(
        "--role_aware_secondary_role_weight",
        type=float,
        default=0.0,
        help="Relative reward for non-question-needed evidence roles. 0 keeps strict question-role gating.",
    )
    parser.add_argument(
        "--role_aware_soft_query_prior",
        action="store_true",
        help="Use S2-style all-role coverage/instance gains, with question-needed roles as a soft prior.",
    )
    parser.add_argument(
        "--role_aware_query_coverage_weight",
        type=float,
        default=0.0,
        help="Extra bonus for covering a not-yet-covered question-needed evidence role in soft-query-prior mode.",
    )
    parser.add_argument(
        "--role_aware_gate_semantic_redundancy",
        action="store_true",
        help="Reduce semantic redundancy penalty when a clip contributes new needed evidence units.",
    )
    parser.add_argument(
        "--role_aware_fix_first_relevance",
        action="store_true",
        help="Keep the first selected evidence clip as the highest-relevance candidate; apply role-aware diversity only to subsequent clips.",
    )
    parser.add_argument(
        "--robot_dev_mode",
        action="store_true",
        help="Augment normal retrieved robot clips with cached Qwen3-VL detailed descriptions.",
    )
    parser.add_argument(
        "--robot_dev_context_root",
        type=str,
        default="data/vl_contexts/robot",
        help="Precomputed robot dev VL contexts exported from memory graphs.",
    )
    parser.add_argument(
        "--robot_dev_clip_root",
        type=str,
        default="data/clips/robot",
        help="Legacy path used by offline exporters; control no longer rebuilds contexts online.",
    )
    parser.add_argument(
        "--robot_dev_intermediate_root",
        type=str,
        default="data/intermediate_outputs/robot",
        help="Legacy path used by offline exporters; control no longer rebuilds contexts online.",
    )
    parser.add_argument(
        "--robot_dev_detail_root",
        type=str,
        default="data/robot_dev_descriptions/robot",
        help="Cache directory for robot dev VL descriptions.",
    )
    parser.add_argument(
        "--robot_dev_vl_model_path",
        type=str,
        default="models/Qwen3-VL-8B-Instruct",
        help="Local Qwen3-VL model path used for robot dev clip descriptions.",
    )
    parser.add_argument(
        "--robot_dev_vl_device",
        type=str,
        default=None,
        help="Optional torch device for local Qwen3-VL, e.g. cuda:1.",
    )
    parser.add_argument(
        "--robot_dev_force_regen",
        action="store_true",
        help="Force regeneration of cached robot dev VL descriptions.",
    )
    parser.add_argument(
        "--robot_dev_faces_input",
        type=str,
        default="face_only",
        choices=["face_only", "face_frames"],
        help="Face representation style passed into the robot dev VL prompt.",
    )
    parser.add_argument(
        "--robot_dev_merge_mode",
        type=str,
        default="replace",
        choices=["replace", "append", "prepend"],
        help="How robot dev VL details should be combined with retrieved memory lines.",
    )
    parser.add_argument(
        "--robot_dev_max_detail_items",
        type=int,
        default=12,
        help="Maximum number of VL detail lines appended per retrieved clip.",
    )
    parser.add_argument(
        "--robot_dev_max_new_tokens",
        type=int,
        default=1024,
        help="Maximum tokens generated by the robot dev VL model per clip.",
    )
    args = parser.parse_args()
    EQUIVALENCE_MODE = args.equivalence_mode
    SPEAKER_AWARE_RETRIEVAL = args.speaker_aware_retrieval
    SPEAKER_RETRIEVAL_BIAS = max(0.0, float(args.speaker_retrieval_bias))
    SPEAKER_RETRIEVAL_HARD_FILTER = args.speaker_retrieval_hard_filter
    STRIP_TEMPORAL_EDGES = args.strip_temporal_edges
    STRIP_SCENE_NODES = args.strip_scene_nodes
    SCENE_RERANK = args.scene_rerank
    SCENE_RERANK_WEIGHT = args.scene_rerank_weight
    SCENE_BACKGROUND = args.scene_background
    DIVERSE_CLIP_RETRIEVAL = args.diverse_clip_retrieval
    FIXED_CLIP_BACKFILL_CURRENT = args.fixed_clip_backfill_current
    DIVERSE_CLIP_POOL_SIZE = max(2, int(args.diverse_clip_pool_size))
    DIVERSE_CLIP_MMR_CANDIDATE_POOL_SIZE = max(0, int(args.diverse_clip_mmr_candidate_pool_size))
    CLIP_INTRA_SIMILARITY_THRESHOLD = float(args.clip_intra_similarity_threshold)
    CLIP_MMR_LAMBDA = float(args.clip_mmr_lambda)
    CLIP_MAX_NODES_FOR_DIVERSITY = max(1, int(args.clip_max_nodes_for_diversity))
    DYNAMIC_MMR_CLIP_RETRIEVAL = args.dynamic_mmr_clip_retrieval
    DYNAMIC_MMR_MIN_CLIPS = max(1, int(args.dynamic_mmr_min_clips))
    DYNAMIC_MMR_MAX_CLIPS = max(DYNAMIC_MMR_MIN_CLIPS, int(args.dynamic_mmr_max_clips))
    DYNAMIC_MMR_STOP_THRESHOLD = float(args.dynamic_mmr_stop_threshold)
    DYNAMIC_MMR_EXTRA_CLIPS = max(0, int(args.dynamic_mmr_extra_clips))
    DYNAMIC_MMR_TRACE_PATH = args.dynamic_mmr_trace_path
    DYNAMIC_MMR_LOG_SCORES = args.dynamic_mmr_log_scores
    DYNAMIC_MMR_SCORE_SOURCE = args.dynamic_mmr_score_source
    DYNAMIC_MMR_POLICY = args.dynamic_mmr_policy
    DYNAMIC_MMR_CONFIDENCE_THRESHOLD = float(args.dynamic_mmr_confidence_threshold)
    DYNAMIC_MMR_AMBIGUITY_GAP_THRESHOLD = float(args.dynamic_mmr_ambiguity_gap_threshold)
    DYNAMIC_MMR_KNEE_MIN_DROP = float(args.dynamic_mmr_knee_min_drop)
    DYNAMIC_MMR_KNEE_ALPHA = float(args.dynamic_mmr_knee_alpha)
    DYNAMIC_MMR_UNCERTAINTY_ALPHA = float(args.dynamic_mmr_uncertainty_alpha)
    FULL_ADAPTIVE_K_RETRIEVAL = args.full_adaptive_k_retrieval
    FULL_ADAPTIVE_K_STRATEGY = args.full_adaptive_k_strategy
    FULL_ADAPTIVE_K_IGNORE_EXTREME = float(args.full_adaptive_k_ignore_extreme)
    FULL_ADAPTIVE_K_IGNORE_EXTREME_TAIL = float(args.full_adaptive_k_ignore_extreme_tail)
    FULL_ADAPTIVE_K_IGNORE_BELOW_MEDIAN = args.full_adaptive_k_ignore_below_median
    FULL_ADAPTIVE_K_RETRIEVE_MORE = parse_int_or_float(args.full_adaptive_k_retrieve_more)
    FULL_ADAPTIVE_K_CANDIDATE_NODES = None if int(args.full_adaptive_k_candidate_nodes) <= 0 else int(args.full_adaptive_k_candidate_nodes)
    FULL_ADAPTIVE_K_MIN_NODES = max(1, int(args.full_adaptive_k_min_nodes))
    FULL_ADAPTIVE_K_MAX_NODES = None if int(args.full_adaptive_k_max_nodes) <= 0 else int(args.full_adaptive_k_max_nodes)
    FULL_ADAPTIVE_K_MIN_CLIPS = None if int(args.full_adaptive_k_min_clips) <= 0 else int(args.full_adaptive_k_min_clips)
    FULL_ADAPTIVE_K_MAX_CLIPS = None if int(args.full_adaptive_k_max_clips) <= 0 else int(args.full_adaptive_k_max_clips)
    FULL_ADAPTIVE_K_EXTRA_CLIPS = max(0, int(args.full_adaptive_k_extra_clips))
    CLIP_ADAPTIVE_K_RETRIEVAL = args.clip_adaptive_k_retrieval
    CLIP_ADAPTIVE_K_STRATEGY = args.clip_adaptive_k_strategy
    CLIP_ADAPTIVE_K_IGNORE_EXTREME = float(args.clip_adaptive_k_ignore_extreme)
    CLIP_ADAPTIVE_K_IGNORE_EXTREME_TAIL = float(args.clip_adaptive_k_ignore_extreme_tail)
    CLIP_ADAPTIVE_K_IGNORE_BELOW_MEDIAN = args.clip_adaptive_k_ignore_below_median
    CLIP_ADAPTIVE_K_RETRIEVE_MORE = parse_int_or_float(args.clip_adaptive_k_retrieve_more)
    CLIP_ADAPTIVE_K_MIN_CLIPS = max(1, int(args.clip_adaptive_k_min_clips))
    CLIP_ADAPTIVE_K_MAX_CLIPS = None if int(args.clip_adaptive_k_max_clips) <= 0 else int(args.clip_adaptive_k_max_clips)
    CLIP_ADAPTIVE_K_EXTRA_CLIPS = max(0, int(args.clip_adaptive_k_extra_clips))
    CLIP_ADAPTIVE_K_SCORE_SOURCE = args.clip_adaptive_k_score_source
    DYNAMICRAG_CLIP_RETRIEVAL = args.dynamicrag_clip_retrieval
    DYNAMICRAG_MODEL = args.dynamicrag_model
    DYNAMICRAG_API_BASE = args.dynamicrag_api_base
    DYNAMICRAG_API_KEY = args.dynamicrag_api_key
    DYNAMICRAG_TEMPERATURE = float(args.dynamicrag_temperature)
    DYNAMICRAG_MAX_TOKENS = max(1, int(args.dynamicrag_max_tokens))
    DYNAMICRAG_TIMEOUT = max(1.0, float(args.dynamicrag_timeout))
    DYNAMICRAG_MIN_CLIPS = max(0, int(args.dynamicrag_min_clips))
    DYNAMICRAG_MAX_CLIPS = None if int(args.dynamicrag_max_clips) <= 0 else int(args.dynamicrag_max_clips)
    DYNAMICRAG_MAX_NODES_PER_CLIP = int(args.dynamicrag_max_nodes_per_clip)
    DYNAMICRAG_MAX_DOC_CHARS = max(256, int(args.dynamicrag_max_doc_chars))
    DF_RAG_CLIP_RETRIEVAL = args.df_rag_clip_retrieval
    DF_RAG_MODEL = args.df_rag_model
    DF_RAG_API_BASE = args.df_rag_api_base
    DF_RAG_API_KEY = args.df_rag_api_key
    DF_RAG_TEMPERATURE = float(args.df_rag_temperature)
    DF_RAG_PLANNER_MAX_TOKENS = max(1, int(args.df_rag_planner_max_tokens))
    DF_RAG_EVALUATOR_MAX_TOKENS = max(1, int(args.df_rag_evaluator_max_tokens))
    DF_RAG_TIMEOUT = max(1.0, float(args.df_rag_timeout))
    DF_RAG_LAMBDAS = args.df_rag_lambdas
    DF_RAG_SET_SIZE = max(1, int(args.df_rag_set_size))
    DF_RAG_MAX_NODES_PER_CLIP = int(args.df_rag_max_nodes_per_clip)
    DF_RAG_MAX_DOC_CHARS = max(256, int(args.df_rag_max_doc_chars))
    DF_RAG_FALLBACK_LAMBDA = float(args.df_rag_fallback_lambda)
    ADAPTIVE_RAG_RETRIEVAL = args.adaptive_rag_retrieval
    ADAPTIVE_RAG_ROUTE_SOURCE = args.adaptive_rag_route_source
    ADAPTIVE_RAG_CLASSIFIER_PATH = args.adaptive_rag_classifier_path
    ADAPTIVE_RAG_CLASSIFIER_MODEL = args.adaptive_rag_classifier_model
    ADAPTIVE_RAG_CLASSIFIER_API_BASE = args.adaptive_rag_classifier_api_base
    ADAPTIVE_RAG_CLASSIFIER_API_KEY = args.adaptive_rag_classifier_api_key
    ADAPTIVE_RAG_CLASSIFIER_TEMPERATURE = float(args.adaptive_rag_classifier_temperature)
    ADAPTIVE_RAG_CLASSIFIER_MAX_TOKENS = max(1, int(args.adaptive_rag_classifier_max_tokens))
    ADAPTIVE_RAG_CLASSIFIER_TIMEOUT = max(1.0, float(args.adaptive_rag_classifier_timeout))
    ADAPTIVE_RAG_FALLBACK_LABEL = args.adaptive_rag_fallback_label
    ADAPTIVE_RAG_ZERO_CLIPS = max(0, int(args.adaptive_rag_zero_clips))
    ADAPTIVE_RAG_SINGLE_CLIPS = max(0, int(args.adaptive_rag_single_clips))
    ADAPTIVE_RAG_MULTI_CLIPS = max(0, int(args.adaptive_rag_multi_clips))
    ADAPTIVE_RAG_SELECTOR = args.adaptive_rag_selector
    ADAPTIVE_RAG_SCORE_SOURCE = args.adaptive_rag_score_source
    if DYNAMICRAG_CLIP_RETRIEVAL and not DYNAMICRAG_API_BASE:
        raise ValueError(
            "--dynamicrag_clip_retrieval requires --dynamicrag_api_base or DYNAMICRAG_API_BASE. "
            "Serve gasolsun/DynamicRAG-8B with vLLM's OpenAI-compatible server first."
        )
    if DF_RAG_CLIP_RETRIEVAL and not DF_RAG_API_BASE:
        raise ValueError(
            "--df_rag_clip_retrieval requires --df_rag_api_base or DF_RAG_API_BASE. "
            "Serve a local OpenAI-compatible instruct model for the DF-RAG planner/evaluator first."
        )
    if ADAPTIVE_RAG_RETRIEVAL and ADAPTIVE_RAG_ROUTE_SOURCE in {"api", "llm", "openai"} and not ADAPTIVE_RAG_CLASSIFIER_API_BASE:
        raise ValueError(
            "--adaptive_rag_retrieval with API route source requires --adaptive_rag_classifier_api_base "
            "or ADAPTIVE_RAG_CLASSIFIER_API_BASE."
        )
    if ADAPTIVE_RAG_RETRIEVAL and not DIVERSE_CLIP_RETRIEVAL:
        raise ValueError(
            "--adaptive_rag_retrieval currently runs inside --diverse_clip_retrieval. "
            "Enable --diverse_clip_retrieval so Adaptive-RAG routing is actually applied."
        )
    DYNAMIC_MMR_TYPE_SWITCH = args.dynamic_mmr_type_switch
    DYNAMIC_MMR_TYPE_SWITCH_TYPES = {
        item.strip()
        for item in str(args.dynamic_mmr_type_switch_types).split(",")
        if item.strip()
    }
    DYNAMIC_MMR_TYPE_SWITCH_POLICY = args.dynamic_mmr_type_switch_policy
    DYNAMIC_MMR_TYPE_SWITCH_STOP_THRESHOLD = float(args.dynamic_mmr_type_switch_stop_threshold)
    DYNAMIC_MMR_TYPE_SWITCH_MIN_CLIPS = max(1, int(args.dynamic_mmr_type_switch_min_clips))
    DYNAMIC_MMR_TYPE_SWITCH_MAX_CLIPS = max(
        DYNAMIC_MMR_TYPE_SWITCH_MIN_CLIPS,
        int(args.dynamic_mmr_type_switch_max_clips),
    )
    EVIDENCE_SATURATION_RETRIEVAL = args.evidence_saturation_retrieval
    EVIDENCE_SATURATION_MIN_CLIPS = max(1, int(args.evidence_saturation_min_clips))
    EVIDENCE_SATURATION_MAX_CLIPS = max(EVIDENCE_SATURATION_MIN_CLIPS, int(args.evidence_saturation_max_clips))
    EVIDENCE_SATURATION_STOP_THRESHOLD = float(args.evidence_saturation_stop_threshold)
    EVIDENCE_SATURATION_RELEVANCE_WEIGHT = float(args.evidence_saturation_relevance_weight)
    EVIDENCE_SATURATION_SEMANTIC_GAIN_WEIGHT = float(args.evidence_saturation_semantic_gain_weight)
    EVIDENCE_SATURATION_TEMPORAL_GAIN_WEIGHT = float(args.evidence_saturation_temporal_gain_weight)
    EVIDENCE_SATURATION_ENTITY_GAIN_WEIGHT = float(args.evidence_saturation_entity_gain_weight)
    EVIDENCE_SATURATION_ACTION_STATE_GAIN_WEIGHT = float(args.evidence_saturation_action_state_gain_weight)
    EVIDENCE_SATURATION_SEMANTIC_REDUNDANCY_WEIGHT = float(args.evidence_saturation_semantic_redundancy_weight)
    EVIDENCE_SATURATION_TEMPORAL_REDUNDANCY_WEIGHT = float(args.evidence_saturation_temporal_redundancy_weight)
    EVIDENCE_SATURATION_TEMPORAL_BUCKET_SIZE = max(1, int(args.evidence_saturation_temporal_bucket_size))
    EVIDENCE_SATURATION_NEAR_CLIP_WINDOW = max(0, int(args.evidence_saturation_near_clip_window))
    ROLE_AWARE_CLIP_RETRIEVAL = args.role_aware_clip_retrieval
    ROLE_AWARE_MODEL = args.role_aware_model
    ROLE_AWARE_MODEL_DEVICE = args.role_aware_model_device
    ROLE_AWARE_MAX_NEW_TOKENS = max(256, int(args.role_aware_max_new_tokens))
    ROLE_AWARE_CACHE_DIR = args.role_aware_cache_dir
    ROLE_AWARE_PRECOMPUTED_DIR = args.role_aware_precomputed_dir
    ROLE_AWARE_MAX_NODES_PER_CLIP = max(1, int(args.role_aware_max_nodes_per_clip))
    ROLE_AWARE_QUESTION_ROLE_MODE = args.role_aware_question_role_mode
    ROLE_AWARE_QUESTION_ROLES_DIR = args.role_aware_question_roles_dir
    ROLE_AWARE_ROLE_MATCH_WEIGHT = float(args.role_aware_role_match_weight)
    ROLE_AWARE_RELEVANCE_WEIGHT = float(args.role_aware_relevance_weight)
    ROLE_AWARE_COVERAGE_WEIGHT = float(args.role_aware_coverage_weight)
    ROLE_AWARE_INSTANCE_WEIGHT = float(args.role_aware_instance_weight)
    ROLE_AWARE_ROLE_REDUNDANCY_WEIGHT = float(args.role_aware_role_redundancy_weight)
    ROLE_AWARE_SEMANTIC_REDUNDANCY_WEIGHT = float(args.role_aware_semantic_redundancy_weight)
    ROLE_AWARE_SECONDARY_ROLE_WEIGHT = max(0.0, float(args.role_aware_secondary_role_weight))
    ROLE_AWARE_SOFT_QUERY_PRIOR = args.role_aware_soft_query_prior
    ROLE_AWARE_QUERY_COVERAGE_WEIGHT = float(args.role_aware_query_coverage_weight)
    ROLE_AWARE_GATE_SEMANTIC_REDUNDANCY = args.role_aware_gate_semantic_redundancy
    ROLE_AWARE_FIX_FIRST_RELEVANCE = args.role_aware_fix_first_relevance
    DEV_VOICE_EQUIV_THRESHOLD = args.dev_voice_equiv_threshold
    DEV_VOICE_CLUSTER_THRESHOLD = args.dev_voice_cluster_threshold
    IDENTITY_HINT_MODE = args.identity_hint_mode
    ROBOT_DEV_MODE = args.robot_dev_mode
    ROBOT_DEV_CONTEXT_ROOT = args.robot_dev_context_root
    ROBOT_DEV_CLIP_ROOT = args.robot_dev_clip_root
    ROBOT_DEV_INTERMEDIATE_ROOT = args.robot_dev_intermediate_root
    ROBOT_DEV_DETAIL_ROOT = args.robot_dev_detail_root
    ROBOT_DEV_VL_MODEL_PATH = args.robot_dev_vl_model_path
    ROBOT_DEV_VL_DEVICE = args.robot_dev_vl_device
    ROBOT_DEV_FORCE_REGEN = args.robot_dev_force_regen
    ROBOT_DEV_FACES_INPUT = args.robot_dev_faces_input
    ROBOT_DEV_MERGE_MODE = args.robot_dev_merge_mode
    ROBOT_DEV_MAX_DETAIL_ITEMS = args.robot_dev_max_detail_items
    ROBOT_DEV_MAX_NEW_TOKENS = args.robot_dev_max_new_tokens
    if args.topk is not None:
        processing_config["topk"] = args.topk
    processing_config["retrieval_threshold"] = float(args.retrieval_threshold)
    if args.batch_size is not None:
        processing_config["batch_size"] = max(1, int(args.batch_size))
    consumer_workers = None
    if args.consumer_workers is not None:
        consumer_workers = max(0, int(args.consumer_workers))
    dataset_name = args.data_file.split("/")[-1].split(".")[0]
    if args.output_name:
        dataset_name = args.output_name
    output_dir = "data/results"
    os.makedirs(output_dir, exist_ok=True)
    output_path = os.path.join(output_dir, f"{dataset_name}.jsonl")
    llm_kwargs = {
        "model": model_name,
        "tensor_parallel_size": args.tensor_parallel_size,
        "disable_custom_all_reduce": args.disable_custom_all_reduce,
        "gpu_memory_utilization": float(args.gpu_memory_utilization),
    }
    if int(args.max_model_len) > 0:
        llm_kwargs["max_model_len"] = int(args.max_model_len)
    if os.getenv("M3AGENT_ENFORCE_EAGER", "").strip().lower() in {"1", "true", "yes", "on"}:
        llm_kwargs["enforce_eager"] = True
    model = LLM(**llm_kwargs)

    batched_datas, data = [], []
    datas = json.load(open(args.data_file))
    id_list = None
    if args.list_file:
        with open(args.list_file, "r", encoding="utf-8") as f:
            id_list = [line.strip() for line in f if line.strip() and not line.strip().startswith("#")]
        missing = [vid for vid in id_list if vid not in datas]
        if missing:
            print(f"[WARN] {len(missing)} IDs not found in {args.data_file}: {', '.join(missing[:10])}" + (" ..." if len(missing) > 10 else ""))
    question_ids = None
    if args.question_ids_file:
        with open(args.question_ids_file, "r", encoding="utf-8") as f:
            question_ids = {
                line.strip()
                for line in f
                if line.strip() and not line.strip().startswith("#")
            }
    seen_question_ids = set()
    items = ((vid, datas[vid]) for vid in id_list if vid in datas) if id_list else datas.items()
    for _, v in items:
        for qa in v["qa_list"]:
            question_id = str(qa["question_id"])
            if question_ids is not None and question_id not in question_ids:
                continue
            seen_question_ids.add(question_id)
            item = {
                "id": qa["question_id"],
                "mem_path": v["mem_path"],
                "question": qa["question"],
                "answer": qa["answer"],
            }
            if "type" in qa:
                item["question_types"] = qa["type"]
            if "choices" in qa:
                item["choices"] = normalize_choices(qa["choices"])
            data.append(item)
            if "before_clip" in qa:
                data[-1]["before_clip"] = qa["before_clip"]
            if len(data) == processing_config["batch_size"]:
                batched_datas.append(data)
                data = []
    if len(data) > 0:
        batched_datas.append(data)

    if question_ids is not None:
        missing_question_ids = sorted(question_ids - seen_question_ids)
        if missing_question_ids:
            print(
                f"[WARN] {len(missing_question_ids)} question IDs were not found in the selected data: "
                f"{', '.join(missing_question_ids[:10])}"
                + (" ..." if len(missing_question_ids) > 10 else ""),
                flush=True,
            )

    total_questions = sum(len(batch) for batch in batched_datas)
    print(
        f"[progress] Prepared {total_questions} questions in {len(batched_datas)} batches "
        f"(batch_size={processing_config['batch_size']}, consumer_workers="
        f"{consumer_workers if consumer_workers is not None else 'default'})",
        flush=True,
    )

    completed_questions = 0
    with open(output_path, "w", encoding="utf-8") as f:
        for batch_idx, batched_data in enumerate(batched_datas, start=1):
            batch_ids = [str(item["id"]) for item in batched_data]
            print(
                f"[progress] Starting batch {batch_idx}/{len(batched_datas)} "
                f"with {len(batched_data)} questions; completed={completed_questions}/{total_questions}; "
                f"first_id={batch_ids[0]} last_id={batch_ids[-1]}",
                flush=True,
            )
            for i in range(len(batched_data)):
                prompt_question = format_question_for_prompt(
                    batched_data[i]["question"],
                    batched_data[i].get("choices"),
                )
                batched_data[i]["conversations"] = [{"role": "system", "content": system_prompt.format(question=prompt_question)}, {"role": "user", "content": "Searched knowledge: {}"}]
                batched_data[i]["finish"] = False
                batched_data[i]["currenr_clips"] = []

            for idx in range(processing_config["total_round"]):
                unfinished_before_round = sum(1 for item in batched_data if not item["finish"])
                print(
                    f"[progress] Batch {batch_idx}/{len(batched_datas)} round "
                    f"{idx + 1}/{processing_config['total_round']} starts with "
                    f"{unfinished_before_round}/{len(batched_data)} unfinished questions",
                    flush=True,
                )
                vllm_inputs = []
                for data in batched_data:
                    if data["finish"]:
                        continue
                    data["conversations"][-1]["content"] += instruction
                    if idx == processing_config["total_round"] - 1:
                        data["conversations"][-1]["content"] += "\n(The Action of this round must be [Answer]. If there is insufficient information, you can make reasonable guesses.)"
                    text = tokenizer.apply_chat_template(
                        data["conversations"],
                        tokenize=True,
                        add_generation_prompt=True,
                        enable_thinking=True
                    )
                    vllm_inputs.append({"prompt_token_ids": text})

                outputs = model.generate(
                    prompts=vllm_inputs,
                    sampling_params=sampling_params,
                    use_tqdm=False,
                )

                i = 0
                for data in batched_data:
                    if data["finish"]:
                        continue
                    data["conversations"].append({"role": "assistant", "content": outputs[i].outputs[0].text})
                    i += 1
                assert i == len(vllm_inputs)

                if ROBOT_DEV_MODE:
                    batched_data = [consumer(item) for item in batched_data]
                else:
                    if consumer_workers == 0:
                        batched_data = [consumer(item) for item in batched_data]
                    else:
                        pool_kwargs = {}
                        if consumer_workers is not None:
                            pool_kwargs["processes"] = consumer_workers
                        with multiprocessing.Pool(**pool_kwargs) as pool:
                            batched_data = pool.map(consumer, batched_data)

                finished_after_round = sum(1 for item in batched_data if item["finish"])
                print(
                    f"[progress] Batch {batch_idx}/{len(batched_datas)} round "
                    f"{idx + 1}/{processing_config['total_round']} ends with "
                    f"{finished_after_round}/{len(batched_data)} finished questions",
                    flush=True,
                )

            for data in batched_data:
                if "response" in data:
                    if data.get("choices"):
                        parsed_choice = extract_option_letter(data["response"], data["choices"])
                        data["parsed_choice"] = parsed_choice
                        data["gpt_eval"] = parsed_choice == str(data["answer"]).strip().upper()
                        data["judge_usage"] = {
                            "model": "choice_parser",
                            "input_tokens": 0,
                            "output_tokens": 0,
                            "total_tokens": 0,
                        }
                        data["judge_cost_usd"] = 0.0
                    else:
                        gpt_eval, judge_usage, judge_cost = eval_answer(
                            data["question"],
                            data["response"],
                            data["answer"],
                        )
                        data["gpt_eval"] = gpt_eval
                        data["judge_usage"] = {
                            "model": eval_model,
                            "input_tokens": int(judge_usage.get("input_tokens", 0) or 0),
                            "output_tokens": int(judge_usage.get("output_tokens", 0) or 0),
                            "total_tokens": int(judge_usage.get("total_tokens", 0) or 0),
                        }
                        data["judge_cost"] = judge_cost
                        data["judge_cost_usd"] = float(judge_cost["usd"])
                        time.sleep(0.5)
                else:
                    data["gpt_eval"] = False
                    data["judge_usage"] = {
                        "model": "none",
                        "input_tokens": 0,
                        "output_tokens": 0,
                        "total_tokens": 0,
                    }
                    data["judge_cost_usd"] = 0.0
                f.write(json.dumps(data, ensure_ascii=False) + '\n')

            f.flush()
            completed_questions += len(batched_data)
            print(
                f"[progress] Completed batch {batch_idx}/{len(batched_datas)}; "
                f"completed={completed_questions}/{total_questions}; output={output_path}",
                flush=True,
            )
