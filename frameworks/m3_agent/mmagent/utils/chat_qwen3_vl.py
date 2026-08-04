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
import gc
import io
import logging
import os

import torch
from PIL import Image
from transformers import AutoProcessor, Qwen3VLForConditionalGeneration

logger = logging.getLogger(__name__)

_PROCESSOR = None
_MODEL = None
_MODEL_PATH = None
_MODEL_DEVICE = None


def _decode_image(image_content):
    if isinstance(image_content, Image.Image):
        return image_content.convert("RGB")
    if isinstance(image_content, str) and os.path.exists(image_content):
        return image_content
    if isinstance(image_content, str) and image_content.startswith(("http://", "https://", "data:image/")):
        return image_content
    if isinstance(image_content, str):
        return Image.open(io.BytesIO(base64.b64decode(image_content))).convert("RGB")
    raise ValueError(f"Unsupported image content type: {type(image_content)}")


def generate_messages(inputs):
    content = []
    for item in inputs:
        if not item.get("content"):
            logger.warning("empty content, skip")
            continue
        if item["type"] == "text":
            content.append({"type": "text", "text": item["content"]})
        elif item["type"] in ["images/jpeg", "images/png"]:
            images = item["content"]
            if not images:
                continue
            if isinstance(images[0], tuple):
                for label, image_content in images:
                    content.append({"type": "text", "text": label})
                    content.append({"type": "image", "image": _decode_image(image_content)})
            else:
                for image_content in images:
                    content.append({"type": "image", "image": _decode_image(image_content)})
        elif item["type"] in ["video_path/mp4", "video_url", "video_base64/mp4", "video_base64/webm"]:
            video_content = item["content"]
            if isinstance(video_content, str) and os.path.exists(video_content):
                content.append({"type": "video", "path": video_content})
            elif isinstance(video_content, str) and video_content.startswith(("http://", "https://")):
                content.append({"type": "video", "url": video_content})
            else:
                raise ValueError(
                    "Qwen3-VL local wrapper currently expects a local video path "
                    "or URL for video inputs."
                )
        else:
            raise ValueError(f"Invalid input type: {item['type']}")
    return [{"role": "user", "content": content}]


def _resolve_model_device(model_device=None):
    if model_device is not None:
        model_device = str(model_device).strip()
        return model_device or None
    env_device = os.environ.get("ROBOT_DEV_VL_DEVICE", "").strip()
    return env_device or None


def _build_device_map(model_device):
    if model_device:
        return {"": model_device}
    return "auto"


def _unload_model():
    global _PROCESSOR, _MODEL, _MODEL_PATH, _MODEL_DEVICE
    _PROCESSOR = None
    _MODEL = None
    _MODEL_PATH = None
    _MODEL_DEVICE = None
    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()


def _load_model(model_path, attn_implementation="flash_attention_2", model_device=None):
    global _PROCESSOR, _MODEL, _MODEL_PATH, _MODEL_DEVICE
    resolved_device = _resolve_model_device(model_device)
    if (
        _MODEL is not None
        and _MODEL_PATH == model_path
        and _MODEL_DEVICE == resolved_device
    ):
        return

    if _MODEL is not None:
        logger.info(
            "Reloading Qwen3-VL model due to path/device change: %s -> %s",
            _MODEL_DEVICE,
            resolved_device,
        )
        _unload_model()

    logger.info(
        "Loading Qwen3-VL model from %s on %s",
        model_path,
        resolved_device or "auto device map",
    )
    _PROCESSOR = AutoProcessor.from_pretrained(model_path)
    device_map = _build_device_map(resolved_device)
    try:
        _MODEL = Qwen3VLForConditionalGeneration.from_pretrained(
            model_path,
            torch_dtype="auto",
            device_map=device_map,
            attn_implementation=attn_implementation,
        )
    except Exception as exc:
        logger.warning(
            "Failed to load Qwen3-VL with %s (%s). Falling back to default attention.",
            attn_implementation,
            exc,
        )
        _MODEL = Qwen3VLForConditionalGeneration.from_pretrained(
            model_path,
            torch_dtype="auto",
            device_map=device_map,
        )
    _MODEL.eval()
    _MODEL_PATH = model_path
    _MODEL_DEVICE = resolved_device


@torch.inference_mode()
def get_response(
    messages,
    model_path,
    model_device=None,
    max_new_tokens=1024,
    do_sample=False,
    temperature=0.2,
    top_p=0.8,
    top_k=20,
):
    _load_model(model_path, model_device=model_device)

    inputs = _PROCESSOR.apply_chat_template(
        messages,
        tokenize=True,
        add_generation_prompt=True,
        return_dict=True,
        return_tensors="pt",
    )
    inputs = inputs.to(_MODEL.device)

    generation_kwargs = {
        "max_new_tokens": max_new_tokens,
        "do_sample": do_sample,
    }
    if do_sample:
        generation_kwargs.update(
            {
                "temperature": temperature,
                "top_p": top_p,
                "top_k": top_k,
            }
        )

    generated_ids = _MODEL.generate(**inputs, **generation_kwargs)
    generated_ids_trimmed = [
        out_ids[len(in_ids):] for in_ids, out_ids in zip(inputs.input_ids, generated_ids)
    ]
    response = _PROCESSOR.batch_decode(
        generated_ids_trimmed,
        skip_special_tokens=True,
        clean_up_tokenization_spaces=False,
    )[0]

    del generated_ids
    del generated_ids_trimmed
    del inputs
    if torch.cuda.is_available():
        torch.cuda.empty_cache()

    return response, len(response)
