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
import mimetypes
import os
from concurrent.futures import ThreadPoolExecutor
from time import sleep
import logging

from google import genai
from google.genai import types
from openai import OpenAI
from .usage_logger import log_api_usage

# Configure logging
logger = logging.getLogger(__name__)

# Disable httpx logging
logging.getLogger("httpx").setLevel(logging.CRITICAL)
logging.getLogger("urllib3").setLevel(logging.CRITICAL)
logging.getLogger("httpcore").setLevel(logging.CRITICAL)

processing_config = json.load(open("configs/processing_config.json"))
temp = processing_config["temperature"]

try:
    config = json.load(open("configs/api_config.json"))
except Exception:
    config = {}

MAX_RETRIES = 5

_GEMINI_CLIENT = None
_GEMINI_CLIENT_KEY = None
_OPENAI_CLIENT = None
_OPENAI_CLIENT_KEY = None


def _safe_positive_int(value, default):
    try:
        value = int(value)
        if value > 0:
            return value
    except Exception:
        pass
    return default


def _get_model_config(model):
    if model in config:
        return config[model]
    if model.startswith("models/"):
        return config.get(model[len("models/"):], {})
    return {}


def _is_openai_chat_model(model):
    normalized = model[len("models/"):] if model.startswith("models/") else model
    return normalized.startswith("gpt-")


def _normalize_openai_model(model):
    return model[len("models/"):] if model.startswith("models/") else model


def _configure_gemini(model):
    model_config = _get_model_config(model)
    api_key = model_config.get("api_key") or os.getenv("GEMINI_API_KEY")
    if not api_key:
        raise ValueError(f"Missing api_key for model: {model}")

    global _GEMINI_CLIENT, _GEMINI_CLIENT_KEY
    if _GEMINI_CLIENT is None or _GEMINI_CLIENT_KEY != api_key:
        _GEMINI_CLIENT = genai.Client(api_key=api_key)
        _GEMINI_CLIENT_KEY = api_key
    return _GEMINI_CLIENT


def _configure_openai(model):
    model_config = _get_model_config(model)
    api_key = model_config.get("api_key") or os.getenv("OPENAI_API_KEY")
    if not api_key:
        raise ValueError(f"Missing api_key for model: {model}")

    global _OPENAI_CLIENT, _OPENAI_CLIENT_KEY
    if _OPENAI_CLIENT is None or _OPENAI_CLIENT_KEY != api_key:
        _OPENAI_CLIENT = OpenAI(api_key=api_key)
        _OPENAI_CLIENT_KEY = api_key
    return _OPENAI_CLIENT


def _model_for_embedding(model):
    if model.startswith("models/"):
        return model[len("models/"):]
    return model


def _decode_base64(data):
    try:
        return base64.b64decode(data)
    except Exception:
        return None


def _build_gemini_parts(inputs):
    parts = []
    for input_item in inputs:
        if not input_item["content"]:
            logger.warning("empty content, skip")
            continue

        if input_item["type"] == "text":
            parts.append(types.Part.from_text(text=input_item["content"]))
        elif input_item["type"] in ["images/jpeg", "images/png"]:
            img_format = input_item["type"].split("/")[1]
            if isinstance(input_item["content"][0], str):
                for img in input_item["content"]:
                    img_bytes = _decode_base64(img)
                    if img_bytes:
                        parts.append(types.Part.from_bytes(data=img_bytes, mime_type=f"image/{img_format}"))
            else:
                for img in input_item["content"]:
                    parts.append(types.Part.from_text(text=img[0]))
                    img_bytes = _decode_base64(img[1])
                    if img_bytes:
                        parts.append(types.Part.from_bytes(data=img_bytes, mime_type=f"image/{img_format}"))
        elif input_item["type"] in ["video_base64/mp4", "video_base64/webm"]:
            video_format = input_item["type"].split("/")[1]
            video_bytes = _decode_base64(input_item["content"])
            if video_bytes:
                parts.append(types.Part.from_bytes(data=video_bytes, mime_type=f"video/{video_format}"))
        elif input_item["type"] == "video_url":
            parts.append(types.Part.from_text(text=f"Video URL: {input_item['content']}"))
        elif input_item["type"] in ["audio_base64/mp3", "audio_base64/wav"]:
            audio_format = input_item["type"].split("/")[1]
            audio_bytes = _decode_base64(input_item["content"])
            if audio_bytes:
                parts.append(types.Part.from_bytes(data=audio_bytes, mime_type=f"audio/{audio_format}"))
        else:
            raise ValueError(f"Invalid input type: {input_item['type']}")
    return parts


def _build_openai_messages(inputs):
    content = []
    for input_item in inputs:
        if not input_item["content"]:
            logger.warning("empty content, skip")
            continue
        if input_item["type"] != "text":
            raise ValueError(f"OpenAI eval path only supports text inputs, got: {input_item['type']}")
        content.append({"type": "input_text", "text": input_item["content"]})
    return [{"role": "user", "content": content}]


def _extract_openai_text(response):
    text = getattr(response, "output_text", None)
    if text:
        return text

    output = getattr(response, "output", None)
    texts = []
    for item in output or []:
        for part in getattr(item, "content", None) or []:
            if getattr(part, "type", None) in {"output_text", "text"}:
                texts.append(getattr(part, "text", ""))
    return "".join(texts)


def _usage_to_dict(usage):
    if usage is None:
        return {
            "input_tokens": 0,
            "output_tokens": 0,
            "total_tokens": 0,
        }
    if hasattr(usage, "model_dump"):
        raw_usage = usage.model_dump()
    elif isinstance(usage, dict):
        raw_usage = usage
    else:
        raw_usage = {
            key: getattr(usage, key)
            for key in dir(usage)
            if not key.startswith("_") and not callable(getattr(usage, key))
        }
    return {
        "input_tokens": int(raw_usage.get("input_tokens") or raw_usage.get("prompt_tokens") or 0),
        "output_tokens": int(raw_usage.get("output_tokens") or raw_usage.get("completion_tokens") or 0),
        "total_tokens": int(raw_usage.get("total_tokens") or 0),
        "raw": raw_usage,
    }


def _extract_gemini_text(response):
    text = getattr(response, "text", None)
    if text:
        return text
    if getattr(response, "candidates", None):
        content = getattr(response.candidates[0], "content", None)
        parts = getattr(content, "parts", None) if content else None
        if parts:
            return "".join([getattr(p, "text", "") for p in parts])
    return ""


def get_response(model, messages, timeout=30):
    text, usage = get_response_with_usage(model, messages, timeout=timeout)
    return text, int(usage.get("total_tokens", 0))


def get_response_with_usage(model, messages, timeout=30):
    if _is_openai_chat_model(model):
        client = _configure_openai(model)
        response = client.responses.create(
            model=_normalize_openai_model(model),
            input=_build_openai_messages(messages),
            temperature=temp,
            max_output_tokens=8192,
            timeout=timeout,
        )
        usage = _usage_to_dict(getattr(response, "usage", None))
        log_api_usage("response", _normalize_openai_model(model), usage)
        return _extract_openai_text(response), usage

    client = _configure_gemini(model)
    response = client.models.generate_content(
        model=model,
        contents=_build_gemini_parts(messages),
        config=types.GenerateContentConfig(
            temperature=temp,
            max_output_tokens=8192,
            system_instruction="You are an expert in video understanding.",
        ),
    )
    usage = getattr(response, "usage_metadata", None)
    usage_dict = {
        "input_tokens": int(getattr(usage, "prompt_token_count", 0) or 0) if usage else 0,
        "output_tokens": int(getattr(usage, "candidates_token_count", 0) or 0) if usage else 0,
        "total_tokens": int(getattr(usage, "total_token_count", 0) or 0) if usage else 0,
    }
    log_api_usage("response", model, usage_dict)
    return _extract_gemini_text(response), usage_dict


def get_response_with_retry(model, messages, timeout=30):
    for i in range(MAX_RETRIES):
        try:
            return get_response(model, messages, timeout)
        except Exception as e:
            sleep(20)
            logger.warning(f"Retry {i} times, exception: {e} from message {messages}")
            continue
    raise Exception(f"Failed to get response after {MAX_RETRIES} retries")


def get_response_with_usage_retry(model, messages, timeout=30):
    for i in range(MAX_RETRIES):
        try:
            return get_response_with_usage(model, messages, timeout)
        except Exception as e:
            sleep(20)
            logger.warning(f"Retry {i} times, exception: {e} from message {messages}")
            continue
    raise Exception(f"Failed to get response after {MAX_RETRIES} retries")


def parallel_get_response(model, messages, timeout=30):
    batch_size = _get_model_config(model).get("qpm", len(messages)) or len(messages)
    responses = []
    total_tokens = 0

    for i in range(0, len(messages), batch_size):
        batch = messages[i:i + batch_size]
        with ThreadPoolExecutor(max_workers=len(batch)) as executor:
            batch_responses = list(executor.map(lambda msg: get_response_with_retry(model, msg, timeout), batch))

        batch_answers = [response[0] for response in batch_responses]
        batch_tokens = [response[1] for response in batch_responses]
        responses.extend(batch_answers)
        total_tokens += sum(batch_tokens)

    return responses, total_tokens


def get_embedding(model, text, timeout=15):
    client = _configure_openai(model)
    response = client.embeddings.create(
        model=_model_for_embedding(model),
        input=text,
        timeout=timeout,
    )
    usage = getattr(response, "usage", None)
    total_tokens = getattr(usage, "total_tokens", 0) if usage else 0
    log_api_usage(
        "embedding",
        _model_for_embedding(model),
        {"input_tokens": int(total_tokens or 0), "output_tokens": 0, "total_tokens": int(total_tokens or 0)},
        metadata={"input_chars": len(str(text or ""))},
    )
    return response.data[0].embedding, total_tokens


def get_embedding_with_retry(model, text, timeout=15):
    for i in range(MAX_RETRIES):
        try:
            return get_embedding(model, text, timeout)
        except Exception as e:
            sleep(20)
            logger.warning(f"Retry {i} times, exception: {e} from get embedding")
            continue
    raise Exception(f"Failed to get embedding after {MAX_RETRIES} retries")


def parallel_get_embedding(model, texts, timeout=15):
    model_config = _get_model_config(model)
    batch_size = model_config.get("qpm", len(texts)) or len(texts)
    batch_size = _safe_positive_int(os.getenv("EMBEDDING_BATCH_SIZE"), batch_size)
    max_workers_cap = model_config.get("embedding_max_workers")
    if max_workers_cap is None:
        max_workers_cap = os.getenv("EMBEDDING_MAX_WORKERS", 4)
    max_workers_cap = _safe_positive_int(max_workers_cap, 4)
    embeddings = []
    total_tokens = 0

    for i in range(0, len(texts), batch_size):
        batch = texts[i:i + batch_size]
        max_workers = min(len(batch), max_workers_cap)

        with ThreadPoolExecutor(max_workers=max_workers) as executor:
            results = list(executor.map(lambda x: get_embedding_with_retry(model, x, timeout), batch))

        batch_embeddings = [result[0] for result in results]
        batch_tokens = [result[1] for result in results]
        embeddings.extend(batch_embeddings)
        total_tokens += sum(batch_tokens)

    return embeddings, total_tokens


def get_whisper(model, file_path):
    client = _configure_gemini(model)
    with open(file_path, "rb") as file:
        audio_bytes = file.read()
    mime_type = mimetypes.guess_type(file_path)[0] or "audio/mpeg"
    prompt = "Transcribe the audio to text. Return only the transcription."
    response = client.models.generate_content(
        model=model,
        contents=[
            types.Part.from_text(text=prompt),
            types.Part.from_bytes(data=audio_bytes, mime_type=mime_type),
        ],
        config=types.GenerateContentConfig(temperature=0),
    )
    return getattr(response, "text", None) or ""


def get_whisper_with_retry(model, file_path):
    for i in range(MAX_RETRIES):
        try:
            return get_whisper(model, file_path)
        except Exception as e:
            sleep(20)
            logger.warning(f"Retry {i} times, exception: {e}")
    raise Exception(f"Failed to get response after {MAX_RETRIES} retries")


def parallel_get_whisper(model, file_paths):
    batch_size = _get_model_config(model).get("qpm", len(file_paths)) or len(file_paths)
    responses = []

    for i in range(0, len(file_paths), batch_size):
        batch = file_paths[i:i + batch_size]
        max_workers = len(batch)

        with ThreadPoolExecutor(max_workers=max_workers) as executor:
            batch_responses = list(executor.map(lambda x: get_whisper_with_retry(model, x), batch))

        responses.extend(batch_responses)

    return responses


def generate_messages(inputs):
    return inputs


def print_messages(messages):
    logger.debug(json.dumps(messages, ensure_ascii=False))
