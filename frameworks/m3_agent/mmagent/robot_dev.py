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
import base64
import json
import logging
import os
from collections import defaultdict
from io import BytesIO

from PIL import Image, ImageDraw

from .prompts import prompt_generate_robot_dev_clip_detail
from .utils.chat_qwen3_vl import (
    generate_messages as generate_vl_messages,
    get_response as get_vl_response,
)
from .utils.general import validate_and_fix_json
from .utils.video_processing import process_video_clip

processing_config = json.load(open("configs/processing_config.json"))
logger = logging.getLogger(__name__)

DEFAULT_CONTEXT_ROOT = "data/vl_contexts/robot"
FACE_MATCH_MARGIN = 0.05
VOICE_MATCH_MARGIN = 0.03


def _parse_clip_key(clip_key):
    if isinstance(clip_key, int):
        return clip_key
    if isinstance(clip_key, str) and clip_key.startswith("CLIP_"):
        return int(clip_key.split("_", 1)[1])
    raise ValueError(f"Invalid clip key: {clip_key}")


def _video_id_from_mem_path(mem_path):
    return os.path.splitext(os.path.basename(mem_path))[0]


def _context_path(context_root, mem_path, clip_id):
    video_id = _video_id_from_mem_path(mem_path)
    return os.path.join(context_root, video_id, f"clip_{clip_id}.json")


def _detail_path(detail_root, mem_path, clip_id):
    video_id = _video_id_from_mem_path(mem_path)
    return os.path.join(detail_root, video_id, f"clip_{clip_id}_detail.json")


def _filter_face(face):
    try:
        det = float(face["extra_data"]["face_detection_score"])
        quality = float(face["extra_data"]["face_quality_score"])
    except Exception:
        return False
    return (
        det > processing_config["face_detection_score_threshold"]
        and quality > processing_config["face_quality_score_threshold"]
    )


def _load_raw_json(path):
    if not os.path.exists(path):
        return []
    with open(path, "r", encoding="utf-8") as f:
        payload = json.load(f)
    return payload if isinstance(payload, list) else []


def _load_faces_for_context(save_path):
    faces_json = _load_raw_json(save_path)
    id2faces = defaultdict(list)
    for face in faces_json:
        cluster_id = int(face.get("cluster_id", -1))
        if cluster_id == -1 or not _filter_face(face):
            continue
        id2faces[cluster_id].append(face)

    max_faces = processing_config["max_faces_per_character"]
    selected = {}
    for cluster_id, faces in id2faces.items():
        selected[cluster_id] = sorted(
            faces,
            key=lambda x: (
                float(x["extra_data"]["face_detection_score"]),
                float(x["extra_data"]["face_quality_score"]),
            ),
            reverse=True,
        )[:max_faces]
    return selected


def _collect_graph_memories(video_graph, clip_id):
    memories = []
    seen = set()
    for node_id in getattr(video_graph, "text_nodes_by_clip", {}).get(clip_id, []):
        node = video_graph.nodes.get(node_id)
        if node is None:
            continue
        for content in node.metadata.get("contents", []):
            if not isinstance(content, str):
                continue
            normalized = content.strip()
            if not normalized or normalized in seen:
                continue
            seen.add(normalized)
            memories.append(normalized)
    return memories


def _build_face_content_to_node_map(video_graph):
    mapping = {}
    for node_id, node in video_graph.nodes.items():
        if node.type != "img":
            continue
        for face_base64 in node.metadata.get("contents", []):
            mapping[face_base64] = node_id
    return mapping


def _build_voice_text_to_node_map(video_graph):
    mapping = defaultdict(list)
    for node_id, node in video_graph.nodes.items():
        if node.type != "voice":
            continue
        for utterance in node.metadata.get("contents", []):
            if isinstance(utterance, str) and utterance.strip():
                mapping[utterance.strip()].append(node_id)
    return mapping


def _label_from_node(reverse_map, node_prefix, node_id, fallback_label):
    if node_id is None:
        return fallback_label
    character_id = reverse_map.get(f"{node_prefix}_{node_id}")
    if character_id:
        return f"<{character_id}>"
    return f"<{node_prefix}_{node_id}>"


def _pick_confident_match(matches, min_margin):
    if not matches:
        return None
    if len(matches) == 1:
        return matches[0][0]
    top_node_id, top_score = matches[0]
    second_score = matches[1][1]
    if top_score - second_score >= min_margin:
        return top_node_id
    return None


def _resolve_face_node_id(video_graph, face_to_node, faces):
    for face in faces:
        face_base64 = face.get("extra_data", {}).get("face_base64")
        if face_base64 in face_to_node:
            return face_to_node[face_base64]

    embeddings = [face.get("face_emb") for face in faces if face.get("face_emb") is not None]
    if not embeddings:
        return None
    matches = video_graph.search_img_nodes({"embeddings": embeddings, "contents": []})
    return _pick_confident_match(matches, FACE_MATCH_MARGIN)


def _resolve_voice_node_id(video_graph, voice_text_to_node, voice):
    embedding = voice.get("embedding")
    if embedding is not None:
        matches = video_graph.search_voice_nodes({"embeddings": [embedding], "contents": []})
        node_id = _pick_confident_match(matches, VOICE_MATCH_MARGIN)
        if node_id is not None:
            return node_id

    utterance = (voice.get("asr") or "").strip()
    candidates = voice_text_to_node.get(utterance, [])
    if len(candidates) == 1:
        return candidates[0]
    return None


def _render_face_payloads_from_graph(video_graph, base64_frames, faces_list):
    reverse_map = getattr(video_graph, "reverse_character_mappings", {})
    face_to_node = _build_face_content_to_node_map(video_graph)
    face_only = []
    face_frames = []
    used_labels = set()

    for local_id, faces in faces_list.items():
        if not faces:
            continue
        representative = faces[0]
        node_id = _resolve_face_node_id(video_graph, face_to_node, faces)
        label = _label_from_node(reverse_map, "face", node_id, f"<face_{local_id}>")
        if label in used_labels:
            continue
        used_labels.add(label)

        face_only.append((f"{label}:", representative["extra_data"]["face_base64"]))

        if base64_frames is None:
            continue
        try:
            frame_id = representative["frame_id"]
            frame_bytes = base64.b64decode(base64_frames[frame_id])
            frame_img = Image.open(BytesIO(frame_bytes))
            draw = ImageDraw.Draw(frame_img)
            bbox = representative["bounding_box"]
            draw.rectangle(
                [(bbox[0], bbox[1]), (bbox[2], bbox[3])],
                outline=(0, 255, 0),
                width=4,
            )
            buffered = BytesIO()
            frame_img.save(buffered, format="JPEG")
            face_frames.append(
                (f"{label}:", base64.b64encode(buffered.getvalue()).decode("utf-8"))
            )
        except Exception as exc:
            logger.warning("Failed to render face frame for %s: %s", label, exc)

    return face_only, face_frames


def _render_voice_payloads(video_graph, raw_voices):
    reverse_map = getattr(video_graph, "reverse_character_mappings", {})
    voice_text_to_node = _build_voice_text_to_node_map(video_graph)
    packed = {}
    unmatched_counter = 0

    for voice in raw_voices:
        transcript = (voice.get("asr") or "").strip()
        if not transcript:
            continue
        node_id = _resolve_voice_node_id(video_graph, voice_text_to_node, voice)
        if node_id is None:
            label = f"<voice_{unmatched_counter}>"
            unmatched_counter += 1
        else:
            label = _label_from_node(reverse_map, "voice", node_id, f"<voice_{node_id}>")
        packed.setdefault(label, []).append(
            {
                "start_time": voice["start_time"],
                "end_time": voice["end_time"],
                "asr": transcript,
            }
        )

    return packed


def _build_context_payload(
    video_graph,
    mem_path,
    clip_id,
    clip_dir,
    intermediate_dir,
):
    video_id = _video_id_from_mem_path(mem_path)
    clip_path = os.path.join(clip_dir, f"{clip_id}.mp4")
    faces_path = os.path.join(intermediate_dir, f"clip_{clip_id}_faces.json")
    voices_path = os.path.join(intermediate_dir, f"clip_{clip_id}_voices.json")

    faces_list = _load_faces_for_context(faces_path)
    raw_voices = _load_raw_json(voices_path)

    base64_frames = None
    if os.path.exists(clip_path) and faces_list:
        try:
            _, base64_frames, _ = process_video_clip(clip_path, fps=processing_config["fps"])
        except Exception as exc:
            logger.warning("Failed to process %s for face frame rendering: %s", clip_path, exc)

    face_only, face_frames = _render_face_payloads_from_graph(
        video_graph, base64_frames, faces_list
    )
    voices = _render_voice_payloads(video_graph, raw_voices)

    return {
        "version": 1,
        "video_id": video_id,
        "clip_id": clip_id,
        "clip_path": clip_path,
        "graph_memories": _collect_graph_memories(video_graph, clip_id),
        "faces": {
            "face_only": face_only,
            "face_frames": face_frames,
        },
        "voices": voices,
    }


def export_robot_dev_contexts_for_video(
    video_graph,
    mem_path,
    clip_dir,
    intermediate_dir,
    context_root=DEFAULT_CONTEXT_ROOT,
    overwrite=False,
    clip_ids=None,
):
    video_id = _video_id_from_mem_path(mem_path)
    clip_ids = clip_ids or sorted(getattr(video_graph, "text_nodes_by_clip", {}).keys())
    written = 0

    if not getattr(video_graph, "reverse_character_mappings", None):
        video_graph.refresh_equivalences()

    for clip_id in clip_ids:
        context_path = _context_path(context_root, mem_path, int(clip_id))
        if os.path.exists(context_path) and not overwrite:
            continue
        if not os.path.isdir(intermediate_dir):
            logger.warning("Intermediate output directory missing for %s: %s", video_id, intermediate_dir)
            continue

        payload = _build_context_payload(
            video_graph=video_graph,
            mem_path=mem_path,
            clip_id=int(clip_id),
            clip_dir=clip_dir,
            intermediate_dir=intermediate_dir,
        )
        os.makedirs(os.path.dirname(context_path), exist_ok=True)
        with open(context_path, "w", encoding="utf-8") as f:
            json.dump(payload, f, ensure_ascii=False, indent=2)
        written += 1

    return written


def _load_context_payload(context_path):
    if not os.path.exists(context_path):
        return None
    with open(context_path, "r", encoding="utf-8") as f:
        payload = json.load(f)
    return payload if isinstance(payload, dict) else None


def _build_video_context_from_payload(payload, faces_input):
    faces = payload.get("faces", {}).get(faces_input) or payload.get("faces", {}).get("face_only") or []
    faces = [
        tuple(face) if isinstance(face, list) and len(face) == 2 else face
        for face in faces
    ]
    voices = payload.get("voices", {})
    return [
        {
            "type": "video_path/mp4",
            "content": payload["clip_path"],
        },
        {
            "type": "text",
            "content": "Face features:",
        },
        {
            "type": "images/jpeg",
            "content": faces,
        },
        {
            "type": "text",
            "content": "Voice features:",
        },
        {
            "type": "text",
            "content": json.dumps(voices, ensure_ascii=False),
        },
    ]


def _coerce_detail_text(result):
    if isinstance(result, dict):
        for key in ["detailed_description", "detail", "memory", "details"]:
            value = result.get(key)
            if isinstance(value, str) and value.strip():
                return value.strip()
            if isinstance(value, list):
                result = value
                break
        else:
            return ""
    if isinstance(result, str):
        return result.strip()
    if not isinstance(result, list):
        return ""

    paragraphs = []
    for item in result:
        if isinstance(item, str):
            item = item.strip()
            if item:
                paragraphs.append(item)
        elif isinstance(item, dict):
            for value in item.values():
                if isinstance(value, str) and value.strip():
                    paragraphs.append(value.strip())
    return " ".join(paragraphs).strip()


def _generate_clip_detail_from_payload(
    payload,
    model_path,
    faces_input,
    max_new_tokens,
    model_device=None,
):
    video_context = _build_video_context_from_payload(payload, faces_input)
    inputs = [
        {
            "type": "text",
            "content": prompt_generate_robot_dev_clip_detail,
        },
        {
            "type": "text",
            "content": "Clip memories already stored for this clip:\n"
            + json.dumps(payload.get("graph_memories", []), ensure_ascii=False),
        },
    ] + video_context
    messages = generate_vl_messages(inputs)
    response, _ = get_vl_response(
        messages,
        model_path=model_path,
        model_device=model_device,
        max_new_tokens=max_new_tokens,
        do_sample=False,
    )
    parsed = validate_and_fix_json(response)
    detail_text = _coerce_detail_text(parsed)
    if detail_text:
        return detail_text
    raw = (response or "").strip()
    if raw:
        logger.warning(
            "Falling back to raw robot dev VL response for %s clip %s",
            payload.get("video_id"),
            payload.get("clip_id"),
        )
        return raw
    return ""


def _load_cached_detail(detail_path):
    if not os.path.exists(detail_path):
        return None
    with open(detail_path, "r", encoding="utf-8") as f:
        payload = json.load(f)
    if isinstance(payload, str):
        return payload.strip()
    if isinstance(payload, list):
        return " ".join(
            item.strip() for item in payload if isinstance(item, str) and item.strip()
        ).strip()
    if isinstance(payload, dict):
        detail = payload.get("detail")
        if isinstance(detail, str) and detail.strip():
            return detail.strip()
        details = payload.get("details")
        if isinstance(details, list):
            return " ".join(
                item.strip() for item in details if isinstance(item, str) and item.strip()
            ).strip()
    return None


def _save_cached_detail(detail_path, detail_text, model_path):
    os.makedirs(os.path.dirname(detail_path), exist_ok=True)
    with open(detail_path, "w", encoding="utf-8") as f:
        json.dump(
            {
                "model_path": model_path,
                "detail": detail_text,
            },
            f,
            ensure_ascii=False,
            indent=2,
        )


def _generate_or_load_detail(
    mem_path,
    clip_id,
    context_root,
    detail_root,
    vl_model_path,
    force_regen,
    faces_input,
    max_new_tokens,
    vl_model_device=None,
):
    detail_path = _detail_path(detail_root, mem_path, clip_id)
    if not force_regen:
        cached = _load_cached_detail(detail_path)
        if cached is not None:
            return cached

    payload = _load_context_payload(_context_path(context_root, mem_path, clip_id))
    if payload is None:
        logger.warning(
            "Robot dev VL context missing for %s clip %s under %s",
            mem_path,
            clip_id,
            context_root,
        )
        return ""

    detail_text = _generate_clip_detail_from_payload(
        payload=payload,
        model_path=vl_model_path,
        faces_input=faces_input,
        max_new_tokens=max_new_tokens,
        model_device=vl_model_device,
    )
    _save_cached_detail(detail_path, detail_text, vl_model_path)
    return detail_text


def augment_robot_dev_memories(
    video_graph,
    mem_path,
    clip_memories,
    context_root=DEFAULT_CONTEXT_ROOT,
    detail_root="data/robot_dev_descriptions/robot",
    vl_model_path="models/Qwen3-VL-8B-Instruct",
    vl_model_device=None,
    force_regen=False,
    faces_input="face_only",
    merge_mode="replace",
    max_detail_items=12,
    max_new_tokens=1024,
):
    del video_graph, max_detail_items
    if merge_mode not in {"replace", "append", "prepend"}:
        raise ValueError(f"Unsupported robot dev merge_mode: {merge_mode}")
    augmented = {}
    for clip_key, memory_lines in clip_memories.items():
        clip_id = _parse_clip_key(clip_key)
        try:
            detail_text = _generate_or_load_detail(
                mem_path=mem_path,
                clip_id=clip_id,
                context_root=context_root,
                detail_root=detail_root,
                vl_model_path=vl_model_path,
                vl_model_device=vl_model_device,
                force_regen=force_regen,
                faces_input=faces_input,
                max_new_tokens=max_new_tokens,
            )
        except Exception as exc:
            logger.warning(
                "Robot dev VL augmentation failed for %s clip %s: %s",
                mem_path,
                clip_id,
                exc,
            )
            detail_text = ""

        memory_lines = [
            line.strip()
            for line in memory_lines
            if isinstance(line, str) and line.strip()
        ]
        detail_text = detail_text.strip() if isinstance(detail_text, str) else ""
        if detail_text:
            if merge_mode == "replace":
                merged_lines = [detail_text]
            elif merge_mode == "append":
                merged_lines = memory_lines + [detail_text]
            else:
                merged_lines = [detail_text] + memory_lines

            deduped_lines = []
            seen = set()
            for line in merged_lines:
                if line in seen:
                    continue
                seen.add(line)
                deduped_lines.append(line)
            augmented[clip_key] = deduped_lines
        else:
            augmented[clip_key] = memory_lines

    return augmented
