# -*- coding: utf-8 -*-
"""
提示词翻译节点
中文自然语言 → 正向标签 / 负向标签
- text / prefix / suffix / default_negative 均支持中英混写
- 英文原样保留，中文翻译成标签
"""

import os
import os
import sys
import re
import json
import sqlite3
import difflib
import hashlib
import threading

import torch

_HERE = os.path.dirname(os.path.abspath(__file__))
_DEFAULT_DB = os.path.join(_HERE, "data", "prompt_db.db")
if not os.path.exists(_DEFAULT_DB):
    _DEFAULT_DB = r"E:\PrompDB\prompt_db.db"   # 兼容本地旧安装
DB_PATH = os.environ.get("PT_DB_PATH", _DEFAULT_DB)
MODEL_DIR = os.environ.get("PT_MODEL_DIR", r"E:\AI\Models\Qwen2.5-1.5B-Instruct")
LAST_FILE = os.environ.get("PT_LAST_FILE", os.path.join(_HERE, "data", "最近一次.txt"))

DEFAULT_PREFIX = "masterpiece, best quality, 1girl, solo, upper body,"
DEFAULT_NEGATIVE = "lowres, bad anatomy, bad hands, extra fingers, watermark, text, worst quality"

SYSTEM = """你是提示词拆解助手。把中文描述拆成关键词短语，用中文输出。
规则：
- positive：画面里要出现的内容
- negative：只在出现"不要/避免/去掉/别/除了"时才有内容
- 没有否定词时，negative 必须是空数组 []
- 短语 2-6 字，不要单独输出"的/地/了/着/很/在/是/她/他/它"

示例1
输入：她站在花海，害羞地微笑，不要水印
输出：{"positive": ["站在花海", "害羞", "微笑"], "negative": ["水印"]}

示例2
输入：薄雾中的木栈道，黄昏
输出：{"positive": ["薄雾", "木栈道", "黄昏"], "negative": []}

示例3
输入：少女在图书馆看书，避免过多人物
输出：{"positive": ["少女", "图书馆", "看书"], "negative": ["过多人物"]}

示例4
输入：少女微笑，不要崩坏，不要文字
输出：{"positive": ["少女", "微笑"], "negative": ["崩坏", "文字"]}

每个"不要/避免/去掉"后面的内容都必须放进 negative，一个都不能漏。
角色名、专有名词、英文字母组合（如 莉莉、lilyww）必须原样保留，不要翻译、不要丢弃。
"站在花海""坐在窗边"这类"动词+场景"要整体保留，不要拆成两段。
只输出 JSON，不要解释。"""

ASCII_ONLY = re.compile(r"^[\x00-\x7F]+$")   # 纯英文/符号：不参与模糊匹配

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
try:
    import ai_backends
except Exception:
    ai_backends = None

_lock = threading.Lock()
_cache = {"tok": None, "model": None}


def _ensure_db():
    """数据库不存在时，从 data/seed.csv 自动建库"""
    if os.path.exists(DB_PATH):
        return
    os.makedirs(os.path.dirname(DB_PATH) or ".", exist_ok=True)
    conn = sqlite3.connect(DB_PATH)
    conn.execute(
        "CREATE TABLE IF NOT EXISTS prompt_map ("
        "id INTEGER PRIMARY KEY AUTOINCREMENT, category TEXT, zh TEXT UNIQUE, "
        "tags TEXT, hits INTEGER DEFAULT 0, created TIMESTAMP DEFAULT CURRENT_TIMESTAMP, "
        "polarity TEXT DEFAULT 'pos')"
    )
    seed = os.path.join(os.path.dirname(os.path.abspath(__file__)), "data", "seed.csv")
    if os.path.exists(seed):
        import csv
        with open(seed, "r", encoding="utf-8-sig", newline="") as f:
            for row in csv.DictReader(f):
                try:
                    conn.execute(
                        "INSERT OR IGNORE INTO prompt_map (category, zh, tags, polarity) VALUES (?,?,?,?)",
                        (row.get("category", "自定义"), row["zh"], row["tags"], row.get("polarity", "pos")),
                    )
                except Exception:
                    pass
        conn.commit()
        print(f"[PromptTranslator] 已从 seed.csv 初始化词库: {DB_PATH}")
    conn.close()


# ---------------- 基础工具 ----------------
def dedupe_tags(s):
    out, seen = [], set()
    for t in (s or "").split(","):
        t = t.strip()
        if t and t not in seen:
            seen.add(t)
            out.append(t)
    return ", ".join(out)


def is_cn_char(ch):
    return "\u4e00" <= ch <= "\u9fff"


CN_PUNCT = "，。、；：！？（）【】《》“”‘’·…—"


def split_lang(s):
    """把字符串切成 [(是否中文, 片段), ...]，中文标点当分隔符"""
    s = "".join("," if ch in CN_PUNCT else ch for ch in s)
    out, buf, cur = [], "", None
    for ch in s:
        flag = is_cn_char(ch)
        if cur is None or flag == cur:
            buf += ch
            cur = flag
        else:
            out.append((cur, buf))
            buf, cur = ch, flag
    if buf:
        out.append((cur, buf))
    return out


def load_db():
    _ensure_db()
    conn = sqlite3.connect(DB_PATH)
    try:
        rows = conn.execute("SELECT zh, tags FROM prompt_map").fetchall()
    finally:
        conn.close()
    return dict(rows)


def load_neg_keys():
    """polarity='neg' 的词：强制归入负向"""
    try:
        conn = sqlite3.connect(DB_PATH)
        rows = conn.execute("SELECT zh FROM prompt_map WHERE polarity='neg'").fetchall()
        conn.close()
        return {r[0] for r in rows}
    except Exception:
        return set()


def load_auto_keys():
    """polarity='auto' 的词：标签由词库给，正负由模型按语境决定"""
    try:
        conn = sqlite3.connect(DB_PATH)
        rows = conn.execute("SELECT zh FROM prompt_map WHERE polarity='auto'").fetchall()
        conn.close()
        return {r[0] for r in rows}
    except Exception:
        return set()


def load_lora_keys():
    """LoRA 分区的键：只允许精确匹配，禁止模糊"""
    try:
        conn = sqlite3.connect(DB_PATH)
        rows = conn.execute("SELECT zh FROM prompt_map WHERE category='LoRA'").fetchall()
        conn.close()
        return {r[0] for r in rows}
    except Exception:
        return set()


def match_phrase(phrase, mapping, keys, lora_keys=None):
    """返回 (命中的键, 标签)；未命中返回 (None, None)"""
    if phrase in mapping:
        return phrase, mapping[phrase]
    # 纯英文/数字（含 LoRA 触发词）：只允许精确匹配，禁止模糊
    if ASCII_ONLY.match(phrase):
        return None, None
    lora_keys = lora_keys or set()
    best = None
    for k in keys:
        if k in lora_keys:          # LoRA 触发词不参与模糊
            continue
        if len(k) < 2 or abs(len(k) - len(phrase)) > 1:
            continue
        r = difflib.SequenceMatcher(None, phrase, k).ratio()
        if r >= 0.68 and (best is None or r > best[0]):
            best = (r, k, mapping[k])
    if best:
        return best[1], best[2]
    return None, None


def match_with_fallback(phrase, mapping, keys, lora_keys=None):
    """返回 (标签, 未收录, 命中键列表)"""
    k, t = match_phrase(phrase, mapping, keys, lora_keys)
    if t:
        return t, [], [k]

    # 纯英文/数字原样透传，不模糊、不算未收录
    if ASCII_ONLY.match(phrase):
        return phrase, [], []

    parts = [x for x in re.split(r"[的地得了是里中上]", phrase) if len(x) >= 2]
    tags, miss, hits = [], [], []
    for p in parts:
        k2, t2 = match_phrase(p, mapping, keys, lora_keys)
        if t2:
            tags.append(t2)
            hits.append(k2)
        else:
            miss.append(p)
    if tags:
        return ", ".join(dict.fromkeys(tags)), miss, hits

    found, rest, hits2 = [], phrase, []
    for k3 in keys:
        if k3 in rest:
            found.append(mapping[k3])
            hits2.append(k3)
            rest = rest.replace(k3, "", 1)
    if found:
        return ", ".join(dict.fromkeys(found)), [], hits2
    return None, [phrase], []


def db_only(text, mapping=None, keys=None, neg_keys=None):
    """纯词库模式，按极性分流。返回 (正向, 负向, 未收录)"""
    mapping = mapping or load_db()
    keys = keys or sorted(mapping.keys(), key=len, reverse=True)
    lora_keys = load_lora_keys()
    neg_keys = neg_keys if neg_keys is not None else load_neg_keys()

    segs = [s for s in re.split(r"[，,。.；;：:\s、]+", text) if s]
    pos_tags, neg_tags, miss = [], [], []

    def route(key, tags):
        if key in neg_keys:
            neg_tags.append(tags)
        else:
            # auto 与 pos 都先放正向（纯词库模式没有语境可判断）
            pos_tags.append(tags)

    for seg in segs:
        if seg in mapping:
            route(seg, mapping[seg])
            continue
        rest, hit = seg, False
        for k in keys:
            if k in rest:
                route(k, mapping[k])
                rest = rest.replace(k, "", 1)
                hit = True
        if not hit:
            t, m, hits = match_with_fallback(seg, mapping, keys, lora_keys)
            if t:
                for h in (hits or [seg]):
                    if h in neg_keys:
                        neg_tags.append(t)
                        break
                else:
                    pos_tags.append(t)
            miss.extend(m)

    return dedupe_tags(", ".join(pos_tags)), dedupe_tags(", ".join(neg_tags)), miss


AI_CONFIG_PATH = os.environ.get("PT_AI_CONFIG", os.path.join(_HERE, "ai_config.json"))


def load_ai_config():
    """读取 AI 后端配置（ai_config.json）"""
    cfg = {"backend": "local", "base_url": "", "model": "", "api_key_env": "PT_API_KEY", "api_key": ""}
    try:
        if os.path.exists(AI_CONFIG_PATH):
            with open(AI_CONFIG_PATH, "r", encoding="utf-8") as f:
                data = json.load(f)
            for k in cfg:
                if k in data and data[k]:
                    cfg[k] = data[k]
    except Exception as e:
        print(f"[PromptTranslator] 读取 ai_config.json 失败: {e}")
    return cfg


# ---------------- 模型 ----------------
def _read_api_key(env_name="PT_API_KEY"):
    """优先级：环境变量 → 用户目录 ~/.pt_api_key 文件"""
    if env_name:
        v = os.environ.get(env_name, "").strip()
        if v:
            return v
    for path in (os.path.join(os.path.expanduser("~"), ".pt_api_key"),
                 os.path.join(os.path.dirname(os.path.abspath(__file__)), "api_key.txt")):
        try:
            if os.path.exists(path):
                with open(path, "r", encoding="utf-8") as f:
                    v = f.read().strip()
                if v:
                    return v
        except Exception:
            pass
    return ""


def free_model():
    try:
        if ai_backends is not None:
            ai_backends.free_local()
    except Exception:
        pass
    with _lock:
        _cache["tok"] = None
        _cache["model"] = None


def ask_model(text, cfg=None):
    """按配置调用对应的 AI 后端，返回 JSON dict 或 None"""
    cfg = cfg or {}
    if ai_backends is None:
        return None
    backend = cfg.get("backend", "local")
    try:
        return ai_backends.segment(
            text,
            SYSTEM,
            backend=backend,
            model_dir=cfg.get("model_dir", MODEL_DIR),
            base_url=cfg.get("base_url", ""),
            api_key=cfg.get("api_key", ""),
            model=cfg.get("model", ""),
            timeout=int(cfg.get("timeout", 60)),
        )
    except Exception as e:
        print(f"[PromptTranslator] AI 后端({backend})调用失败: {e}")
        return None


NEG_MARKERS = ["不要", "避免", "去掉", "别用", "除了", "别让", "别"]
NEG_PREFIXES = ["出现", "有", "任何", "太", "变得", "让", "再", "显得", "看起来", "过于", "太过"]
NEG_POST = re.compile(r"([^\s，,。.；;：:]{2,6})(?:不要|别)(?:出现|存在|有)?(?:[了啦]?)")


def _clean_neg(seg):
    seg = seg.strip()
    for pre in NEG_PREFIXES:
        if seg.startswith(pre) and len(seg) > len(pre) + 1:
            seg = seg[len(pre):]
    return seg.strip()


def extract_forced_negatives(text):
    """从原文里抓否定内容：前置（不要X）+ 后置（X不要）"""
    out = []
    for m in NEG_MARKERS:
        if m not in text:
            continue
        for part in text.split(m)[1:]:
            seg = re.split(r"[，,。.；;：:\s、]", part)[0].strip()
            seg = _clean_neg(seg)
            if 2 <= len(seg) <= 8:
                out.append(seg)
    # 后置："水印不要出现"
    for m in NEG_POST.finditer(text):
        seg = _clean_neg(m.group(1))
        if 2 <= len(seg) <= 8:
            out.append(seg)
    return list(dict.fromkeys(out))


# ---------------- 核心翻译 ----------------
def translate_core(text, use_ai=True, cfg=None):
    """返回 (正向标签, 负向标签, 未收录列表)"""
    text = (text or "").strip()
    if not text:
        return "", "", [], []

    # 纯英文（无汉字）直接当标签用，不翻译
    if not any(is_cn_char(ch) for ch in text):
        return dedupe_tags(text), "", [], []

    mapping = load_db()
    keys = sorted(mapping.keys(), key=len, reverse=True)
    lora_keys = load_lora_keys()
    neg_keys = load_neg_keys()
    auto_keys = load_auto_keys()

    obj = None
    if use_ai:
        try:
            with _lock:
                obj = ask_model(text, cfg)
        except Exception as e:
            print(f"[PromptTranslator] 模型调用失败: {e}")
            obj = None

    if obj is None:
        p0, n0, m0 = db_only(text, mapping, keys, neg_keys)
        return p0, n0, m0, []

    pos_list, neg_list, miss = [], [], []
    auto_log = []
    for p in obj.get("positive", []):
        t, m, hits = match_with_fallback(p, mapping, keys, lora_keys)
        if t:
            hs = hits or []
            if any(h in neg_keys for h in hs):
                neg_list.append(t)      # 词库标记 neg → 强制负向
            elif any(h in auto_keys for h in hs):
                pos_list.append(t)      # auto → 跟随模型（这里模型说是正向）
                auto_log.append(f"{t}=正向")
            else:
                pos_list.append(t)
        miss.extend(m)
    for p in obj.get("negative", []):
        t, m, hits = match_with_fallback(p, mapping, keys, lora_keys)
        if t:
            hs = hits or []
            if any(h in auto_keys for h in hs):
                neg_list.append(t)      # auto → 跟随模型（这里模型说是负向）
                auto_log.append(f"{t}=负向")
            else:
                neg_list.append(t)
        miss.extend(m)

    return dedupe_tags(", ".join(pos_list)), dedupe_tags(", ".join(neg_list)), miss, auto_log


def localize_field(value, use_ai=True, want_negative=False, mixed=True, cfg=None):
    """字段里的中文翻成标签，英文原样保留"""
    value = (value or "").strip()
    if not value:
        return "", []
    if not mixed:
        return value, []

    parts, miss_all = [], []
    for is_cn, chunk in split_lang(value):
        chunk = chunk.strip().strip(",")
        if not chunk:
            continue
        if not is_cn:
            parts.append(chunk)
            continue
        # 先查词库（短词直命中最准，也避免模型"纠正"语义）
        _mapping = load_db()
        _keys = sorted(_mapping.keys(), key=len, reverse=True)
        if chunk in _mapping:
            parts.append(_mapping[chunk])
            continue
        pos, neg, miss = translate_core(chunk, use_ai)
        if not pos and not neg:
            pos2, _neg2, miss2 = db_only(chunk)
            pos = pos2
            miss = miss2
        miss_all.extend(miss)
        tags = dedupe_tags(", ".join([x for x in (neg, pos) if x])) if want_negative else pos
        if tags:
            parts.append(tags)

    return dedupe_tags(", ".join(parts)), miss_all


# ---------------- 对外接口 ----------------
def translate(text, use_ai=True, unload_after=True, prefix="", suffix="",
              default_negative="", mixed=True, ai_backend="local"):
    ai_cfg = load_ai_config()
    if ai_backend in ("", "auto"):
        ai_backend = ai_cfg.get("backend", "local")
    cfg = {
        "backend": ai_backend,
        "base_url": ai_cfg.get("base_url", ""),
        "model": ai_cfg.get("model", ""),
        "api_key": ai_cfg.get("api_key") or _read_api_key(ai_cfg.get("api_key_env", "PT_API_KEY")),
        "model_dir": MODEL_DIR,
    }
    text = (text or "").strip()

    if not (prefix or "").strip():
        prefix = DEFAULT_PREFIX
    custom_neg = (default_negative or "").strip()
    if custom_neg:
        # 自定义负向与内置默认叠加，避免丢掉基础负面词
        default_negative = DEFAULT_NEGATIVE + ", " + custom_neg
    else:
        default_negative = DEFAULT_NEGATIVE

    if not text:
        return "", "", "【提示】输入为空"

    try:
        core_pos, core_neg, miss, auto_log = translate_core(text, use_ai, cfg)

        p_out, p_miss = localize_field(prefix, use_ai, False, mixed, cfg)
        s_out, s_miss = localize_field(suffix, use_ai, False, mixed, cfg)
        n_out, n_miss = localize_field(default_negative, use_ai, True, mixed, cfg)
        miss = list(miss) + list(p_miss) + list(s_miss) + list(n_miss)

        pos = dedupe_tags(", ".join([x for x in (p_out, core_pos, s_out) if x]))
        neg = dedupe_tags(", ".join([x for x in (n_out, core_neg) if x]))

        # 强制负向：原文"不要X"里的 X，即使模型判成正向也要挪到负向
        forced_tags = []
        forced_log = []
        for seg in extract_forced_negatives(text):
            p_f, n_f, _m = db_only(seg)
            t_forced = dedupe_tags(", ".join([x for x in (p_f, n_f) if x]))
            if t_forced:
                forced_tags.append(t_forced)
                forced_log.append(f"{seg}→{t_forced}")
        if forced_tags:
            ft = dedupe_tags(", ".join(forced_tags))
            keep = [x.strip() for x in pos.split(",") if x.strip() and x.strip() not in
                    [y.strip() for y in ft.split(",")]]
            pos = dedupe_tags(", ".join(keep))
            neg = dedupe_tags(", ".join([neg, ft]))

        # 英文 token 保护：正文里出现的英文词（如 LoRA 触发词）如果被模型丢了，补回来
        auto_back = []
        for tok in re.findall(r"[A-Za-z][A-Za-z0-9_\-]{2,}", text):
            if tok.lower() not in pos.lower():
                auto_back.append(tok)
        if auto_back:
            pos = dedupe_tags(pos + ", " + ", ".join(dict.fromkeys(auto_back)))

        info = (
            "【正向】" + (pos or "(无)") + "\n"
            "【负向】" + (neg or "(无)") + "\n"
            "【翻译】" + (core_pos or "(无)") + "\n"
            + (("【语境判定】" + ", ".join(auto_log) + "\n") if auto_log else "")
            + (("【强制负向】" + ", ".join(forced_log) + "\n") if forced_log else "")
            + "【未收录】" + (", ".join(miss) if miss else "(无)") + "\n"
            + "【prefix】" + (p_out or "(空)") + "\n"
            + "【suffix】" + (s_out or "(空)") + "\n"
            + "【negative】" + (n_out or "(空)")
            + (("\n【自动补回】" + ", ".join(dict.fromkeys(auto_back))) if auto_back else "")
        )

        try:
            with open(LAST_FILE, "w", encoding="utf-8") as f:
                f.write(info + "\n")
        except Exception:
            pass

        print("[PromptTranslator] " + info.replace("\n", " | "))
        return pos, neg, info

    except Exception as e:
        core, _n, _ = db_only(text)
        return core, "", f"【正向】{core}\n【错误】{e}"
    finally:
        if unload_after:
            free_model()


# ---------------- 缓存指纹 ----------------
def _cache_fingerprint(kwargs):
    """输入 + 词库修改时间 + 模型目录修改时间"""
    parts = []
    for k in sorted(kwargs.keys()):
        v = kwargs[k]
        if isinstance(v, (str, int, float, bool)):
            parts.append(f"{k}={v}")
    try:
        db_m = os.path.getmtime(DB_PATH)
    except Exception:
        db_m = 0
    try:
        model_m = os.path.getmtime(MODEL_DIR)
    except Exception:
        model_m = 0
    raw = "|".join(parts) + f"|db={db_m}|model={model_m}"
    return hashlib.md5(raw.encode("utf-8")).hexdigest()


# ---------------- 节点 ----------------
class ZH2TagTranslator:
    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "text": ("STRING", {"multiline": True, "default": "她站在花海，害羞地微笑，不要水印"}),
                "use_ai": ("BOOLEAN", {"default": True}),
                "unload_after": ("BOOLEAN", {"default": True}),
                "mixed_language": ("BOOLEAN", {"default": True}),
                "ai_backend": (["local", "openai"], {"default": "local"}),

                "prefix": ("STRING", {"multiline": True, "default": DEFAULT_PREFIX}),
                "suffix": ("STRING", {"multiline": True, "default": ""}),
                "default_negative": ("STRING", {"multiline": True, "default": DEFAULT_NEGATIVE}),
            }
        }

    RETURN_TYPES = ("STRING", "STRING", "STRING")
    RETURN_NAMES = ("positive", "negative", "info")
    FUNCTION = "run"
    CATEGORY = "提示词工具"
    OUTPUT_NODE = True

    @classmethod
    def fingerprint_inputs(cls, **kwargs):
        return _cache_fingerprint(kwargs)

    @classmethod
    def IS_CHANGED(cls, **kwargs):
        return _cache_fingerprint(kwargs)

    def run(self, text, use_ai=True, unload_after=True, mixed_language=True,
            ai_backend="local", prefix="", suffix="", default_negative=""):
        pos, neg, info = translate(text, use_ai, unload_after, prefix, suffix,
                                   default_negative, mixed_language, ai_backend)
        return {"ui": {"text": [info]}, "result": (pos, neg, info)}


class PromptTranslatorCLIP:
    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "text": ("STRING", {"multiline": True, "default": "她站在花海，害羞地微笑，不要水印"}),
                "use_ai": ("BOOLEAN", {"default": True}),
                "unload_after": ("BOOLEAN", {"default": True}),
                "mixed_language": ("BOOLEAN", {"default": True}),
                "ai_backend": (["local", "openai"], {"default": "local"}),

                "prefix": ("STRING", {"multiline": True, "default": DEFAULT_PREFIX}),
                "suffix": ("STRING", {"multiline": True, "default": ""}),
                "default_negative": ("STRING", {"multiline": True, "default": DEFAULT_NEGATIVE}),
                "clip": ("CLIP",),
            }
        }

    RETURN_TYPES = ("CONDITIONING", "CONDITIONING", "STRING")
    RETURN_NAMES = ("positive", "negative", "info")
    FUNCTION = "run"
    CATEGORY = "提示词工具"
    OUTPUT_NODE = True

    @classmethod
    def fingerprint_inputs(cls, **kwargs):
        return _cache_fingerprint(kwargs)

    @classmethod
    def IS_CHANGED(cls, **kwargs):
        return _cache_fingerprint(kwargs)

    def run(self, text, use_ai=True, unload_after=True, mixed_language=True,
            ai_backend="local", prefix="", suffix="", default_negative="", clip=None):
        pos_s, neg_s, info = translate(text, use_ai, unload_after, prefix, suffix,
                                       default_negative, mixed_language, ai_backend)
        import nodes as comfy_nodes
        enc = comfy_nodes.CLIPTextEncode()
        pos_cond = enc.encode(clip, pos_s or "masterpiece")[0]
        neg_cond = enc.encode(clip, neg_s or "lowres, worst quality")[0]
        return {"ui": {"text": [info]}, "result": (pos_cond, neg_cond, info)}


class PromptTranslatorUnload:
    @classmethod
    def INPUT_TYPES(cls):
        return {"required": {}}

    RETURN_TYPES = ()
    FUNCTION = "run"
    CATEGORY = "提示词工具"
    OUTPUT_NODE = True

    def run(self):
        free_model()
        return {"ui": {"text": ["翻译模型已卸载，显存已释放"]}}


NODE_CLASS_MAPPINGS = {
    "ZH2TagTranslator": ZH2TagTranslator,
    "PromptTranslatorCLIP": PromptTranslatorCLIP,
    "PromptTranslatorUnload": PromptTranslatorUnload,
}

NODE_DISPLAY_NAME_MAPPINGS = {
    "ZH2TagTranslator": "提示词翻译（中文→标签）",
    "PromptTranslatorCLIP": "提示词翻译（直连采样器）",
    "PromptTranslatorUnload": "卸载翻译模型",
}
