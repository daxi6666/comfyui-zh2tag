# -*- coding: utf-8 -*-
"""AI 后端：负责把中文拆成 正/负 短语（只输出中文短语，不翻译）

支持：
  local   本地 transformers（Qwen 等）
  openai  任意 OpenAI 兼容接口（DeepSeek / 中转 / Ollama / 硅基流动 / OpenAI）
"""

import json
import os
import threading

_lock = threading.Lock()
_cache = {"tok": None, "model": None, "key": None}


# ---------------- 本地模型 ----------------
def _load_local(model_dir, dtype="float16"):
    import torch
    if _cache["model"] is not None and _cache["key"] == model_dir:
        return _cache["tok"], _cache["model"]

    free_local()
    try:
        import comfy.model_management as mm
        mm.unload_all_models()
        mm.soft_empty_cache()
    except Exception:
        pass
    try:
        torch.cuda.empty_cache()
        torch.cuda.synchronize()
    except Exception:
        pass

    from transformers import AutoTokenizer, AutoModelForCausalLM
    torch_dtype = torch.float16 if dtype == "float16" else torch.bfloat16
    tok = AutoTokenizer.from_pretrained(model_dir)
    model = AutoModelForCausalLM.from_pretrained(
        model_dir, torch_dtype=torch_dtype, device_map="cuda"
    )
    model.eval()
    _cache["tok"], _cache["model"], _cache["key"] = tok, model, model_dir
    return tok, model


def free_local():
    with _lock:
        _cache["tok"] = None
        _cache["model"] = None
        _cache["key"] = None
    try:
        import gc
        import torch
        gc.collect()
        torch.cuda.empty_cache()
        import comfy.model_management as mm
        mm.soft_empty_cache()
    except Exception:
        pass


def _parse_json(raw):
    import re
    m = re.search(r"\{.*\}", raw or "", re.S)
    if not m:
        return None
    try:
        return json.loads(m.group(0))
    except Exception:
        return None


def call_local(system, user, model_dir, max_new_tokens=160, dtype="float16"):
    tok, model = _load_local(model_dir, dtype)
    import torch
    msgs = [{"role": "system", "content": system}, {"role": "user", "content": user}]
    prompt = tok.apply_chat_template(msgs, tokenize=False, add_generation_prompt=True)
    inputs = tok([prompt], return_tensors="pt").to("cuda")
    with torch.no_grad():
        out = model.generate(
            **inputs, max_new_tokens=max_new_tokens, do_sample=False,
            temperature=None, top_p=None, top_k=None,
            pad_token_id=tok.eos_token_id,
        )
    raw = tok.decode(out[0][inputs["input_ids"].shape[1]:], skip_special_tokens=True).strip()
    return _parse_json(raw)


# ---------------- OpenAI 兼容接口 ----------------
def call_openai(system, user, base_url, api_key, model, timeout=60, extra_headers=None):
    import urllib.request
    import urllib.error

    base = (base_url or "").rstrip("/")
    if not base:
        raise ValueError("api_base 为空")
    if not base.endswith("/chat/completions"):
        if base.endswith("/v1"):
            base += "/chat/completions"
        else:
            base += "/v1/chat/completions"

    body = {
        "model": model,
        "messages": [
            {"role": "system", "content": system},
            {"role": "user", "content": user},
        ],
        "temperature": 0,
        "max_tokens": 300,
        "stream": False,
    }
    headers = {
        "Content-Type": "application/json",
        "Authorization": f"Bearer {api_key}",
    }
    if extra_headers:
        headers.update(extra_headers)

    req = urllib.request.Request(
        base, data=json.dumps(body).encode("utf-8"), headers=headers, method="POST"
    )
    try:
        with urllib.request.urlopen(req, timeout=timeout) as r:
            data = json.loads(r.read().decode("utf-8", errors="replace"))
    except urllib.error.HTTPError as e:
        detail = e.read().decode("utf-8", errors="replace")[:300]
        raise RuntimeError(f"HTTP {e.code}: {detail}")

    try:
        raw = data["choices"][0]["message"]["content"]
    except Exception:
        raise RuntimeError(f"响应格式异常: {str(data)[:300]}")
    return _parse_json(raw)


# ---------------- 统一入口 ----------------
def segment(text, system, backend="local", model_dir="", dtype="float16",
            base_url="", api_key="", model="", timeout=60):
    """返回 {"positive": [...], "negative": [...]} 或 None"""
    if backend == "openai":
        return call_openai(system, text, base_url, api_key, model, timeout)
    return call_local(system, text, model_dir, dtype=dtype)
