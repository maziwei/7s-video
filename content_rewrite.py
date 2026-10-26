#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""内容二创引擎 —— 原文/图片 → 标题1 / 标题2 / 胶囊 / 内容 / 开场配音文案

把「粘贴的一段文字，或一张图片」按《内容生成prompt.txt》交给大模型，
产出工作台（榜单视频生成器.html）可直接使用的四要素，以及一句约 5 秒的开场配音：

    标题1(title1) / 标题2(title2) / 胶囊(pill) / 内容(items)

推理优先级（可通过配置 rwOrder 调整，默认 auto）：
    本地 GGUF（进程内 llama.cpp，零安装，ComfyUI 已自带）
      → 本地 OpenAI 兼容服务（Ollama / LM Studio / llama-server）
        → 在线 OpenAI 兼容 API（DeepSeek / 豆包 / 通义 …）

对外主入口：rewrite(cfg, text="", image_data_url=None) -> dict
本模块只依赖标准库 + 可选 llama_cpp，不 import comfy_bridge，避免循环依赖。
"""
import base64
import collections
import json
import logging
import os
import re
import threading
import time

log = logging.getLogger("bridge")

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))

# ----------------------------------------------------------------------------
# Prompt
# ----------------------------------------------------------------------------
# 提示词一律从程序目录读取（相对路径，不写死任何绝对路径），谁先存在用谁：
#   ① 内容生成prompt.txt —— 正文清单规范（当前主用；只讲正文，标题/胶囊由代码补齐）
#   ② 飞书用prompt.txt   —— 旧的四段式规范（内容 / 标题1 / 标题2 / 胶囊）
#   ③ 内容二创Prompt.txt / 内容二创.txt —— 旧文件名，兼容保留
# 实际生效的文件会显示在 /rewrite/status 的 promptPath。
_PROMPT_CANDIDATES = [
    os.path.join(SCRIPT_DIR, "内容生成prompt.txt"),
    os.path.join(SCRIPT_DIR, "飞书用prompt.txt"),
    os.path.join(SCRIPT_DIR, "内容二创Prompt.txt"),
    os.path.join(SCRIPT_DIR, "内容二创.txt"),
]

_PROMPT_FALLBACK = (
    "你是短视频信息流文案编辑。根据原文产出手机竖屏文字视频文案，"
    "严格按顺序输出：【排版风格】标题形式 / 正文形式、【主标题】、【副标题】或【胶囊】、【内容】多行。"
    "标题形式只能是 主标题/主副标题/胶囊标签；正文形式只能是 观点冲击/数字清单/双栏对比/问题拆解/流程。"
    "只输出最终文案，不要解释。"
)

_prompt_cache = {"path": None, "mtime": None, "text": None}
_prompt_warned = {"path": None}


def prompt_path():
    """返回实际生效的提示词文件；一个都不存在时返回首选路径（读取时退回内置兜底）。"""
    for p in _PROMPT_CANDIDATES:
        if os.path.isfile(p):
            return p
    # 一个都没有：告警一次，避免「静默用了内置兜底」而没人发现
    if _prompt_warned["path"] != _PROMPT_CANDIDATES[0]:
        _prompt_warned["path"] = _PROMPT_CANDIDATES[0]
        log.warning("Prompt 文件不存在：%s（请把「内容生成prompt.txt」放到程序目录），暂用内置兜底",
                    _PROMPT_CANDIDATES[0])
    return _PROMPT_CANDIDATES[0]


def load_prompt():
    """读取 Prompt 文件（带 mtime 缓存，改文件即生效，无需重启）。"""
    p = prompt_path()
    try:
        mtime = os.path.getmtime(p)
    except OSError:
        return _PROMPT_FALLBACK
    if (_prompt_cache["path"] != p or _prompt_cache["mtime"] != mtime
            or not _prompt_cache["text"]):
        try:
            with open(p, "r", encoding="utf-8-sig") as f:
                _prompt_cache["text"] = f.read().strip()
            _prompt_cache["path"] = p
            _prompt_cache["mtime"] = mtime
        except Exception as e:
            log.warning("读取 Prompt 失败 %s: %s", p, e)
            return _PROMPT_FALLBACK
    return _prompt_cache["text"] or _PROMPT_FALLBACK


# ----------------------------------------------------------------------------
# 开场配音文案 prompt（项目目录内，可随文件修改即时生效）
# ----------------------------------------------------------------------------
_VOICE_PROMPT_FILE = os.path.join(SCRIPT_DIR, "配音文案prompt.txt")
_VOICE_PROMPT_FALLBACK = (
    "你是短视频信息流开场配音策划。根据主标题、副标题、正文，写一句 18~22 个汉字、"
    "最多两个逗号的开场配音：前 2 秒用人群/数字/反差/损失做钩子，后 3 秒给一个正文能兑现的"
    "硬利益；不复述标题、不编造、不承诺上岸；不要问号/感叹号/冒号/引号/括号/英文/emoji。"
    "只输出这一句。"
)
_voice_prompt_cache = {"mtime": None, "text": None}


def load_voice_prompt():
    """读取《配音文案prompt.txt》；改文件即生效，找不到时用内置兜底。"""
    try:
        mtime = os.path.getmtime(_VOICE_PROMPT_FILE)
    except OSError:
        return _VOICE_PROMPT_FALLBACK
    if _voice_prompt_cache["mtime"] != mtime or not _voice_prompt_cache["text"]:
        try:
            with open(_VOICE_PROMPT_FILE, "r", encoding="utf-8-sig") as f:
                _voice_prompt_cache["text"] = f.read().strip()
            _voice_prompt_cache["mtime"] = mtime
        except Exception as e:
            log.warning("读取配音文案 Prompt 失败 %s: %s", _VOICE_PROMPT_FILE, e)
            return _VOICE_PROMPT_FALLBACK
    return _voice_prompt_cache["text"] or _VOICE_PROMPT_FALLBACK


# ----------------------------------------------------------------------------
# 输出解析：把模型输出的【…】格式解析成四要素
# ----------------------------------------------------------------------------
_TITLE_STYLES = ("主标题", "主副标题", "胶囊标签")
_BODY_STYLES = ("观点冲击", "数字清单", "双栏对比", "问题拆解", "流程")

# 同一个字段的多种可能写法（模型偶尔会换标签名）
_ALIASES = {
    "style": ("排版风格", "风格", "样式"),
    "t1": ("主标题", "标题1", "标题一", "标题"),
    "t2": ("副标题", "标题2", "标题二"),
    "pill": ("胶囊", "胶囊标签"),
    "body": ("内容", "正文"),
}


def _grab(txt, names):
    """取【标签】后面同一行的内容（方括号/方括号半角/冒号均可）。"""
    for name in names:
        m = re.search(r"[【\[]\s*" + re.escape(name) + r"\s*[】\]]\s*[:：]?\s*([^\n\r]*)", txt)
        if m and m.group(1).strip():
            return m.group(1).strip()
    return ""


def _strip_fences(txt):
    t = (txt or "").strip()
    t = re.sub(r"^```[a-zA-Z]*\s*\n", "", t)
    t = re.sub(r"\n\s*```\s*$", "", t)
    return t.strip()


def _norm_style(s, allow):
    """把风格名归一化到允许的枚举值；无法识别返回原串。"""
    if not s:
        return ""
    for a in allow:
        if a in s:
            return a
    return s.strip()


# 角色标记行：小模型答完常以「Assistant:」再答一遍，按此分段
_ROLE_RE = re.compile(r"(?m)^\s*(?:assistant|user|system|ai|助手|用户|系统)\s*[:：].*$", re.I)
# 附加说明段落开头（结尾噪音）
_NOTE_RE = re.compile(r"^[（(]?\s*(?:注|说明|备注|自检|分析)\s*[:：]")


def _clean_lines(text):
    """清洗一个片段：去项目符号/分隔线，遇到「附加说明」段落即停止。"""
    out = []
    for ln in (text or "").splitlines():
        s = ln.strip()
        if not s:
            continue
        # 模型有时会把后面的字段又列一遍，丢弃该行
        if re.match(r"^[【\[]\s*(排版风格|风格|样式|主标题|标题|副标题|胶囊|内容|正文)\s*[】\]]", s):
            continue
        # 丢弃纯分隔线 / markdown 代码围栏
        if re.fullmatch(r"[-=_*~`\s]{3,}", s):
            continue
        s = re.sub(r"^[-•·▪◦]\s+", "", s)   # 去掉行首项目符号
        # 附加说明（注/说明/自检/分析）、Markdown 加粗或标题 —— 到此为止
        if "**" in s or s.startswith("#") or _NOTE_RE.match(s):
            break
        out.append(s)
    return out


def _clean_body(body):
    """清洗正文：按角色标记分段，取最长的一段（通常是模型最终/最完整的答案）。"""
    best = []
    for seg in _ROLE_RE.split(body or ""):
        lines = _clean_lines(seg)
        if len(lines) > len(best):
            best = lines
    return best


def _strip_think(txt):
    """去掉思考链：已闭合的整块删除；未闭合的取最后一个 </think> 之后。"""
    t = re.sub(r"<think\b[^>]*>[\s\S]*?</think\s*>", "", txt)
    if "<think" in t:
        i = t.rfind("</think")
        t = t[i + len("</think>"):] if i >= 0 else t
    t = re.sub(r"</?think[^>]*>", "", t)
    return t.strip()


def _trim_to_format(txt):
    """丢掉【排版风格】之前的寒暄/思考残渣：从第一个「【」开始。"""
    i = txt.find("【")
    if i > 0:
        return txt[i:]
    return txt


def _try_json(txt):
    """兜底：模型违抗指令直接吐了 JSON。"""
    m = re.search(r"\{[\s\S]*\}", txt)
    if not m:
        return None
    try:
        j = json.loads(m.group(0))
    except Exception:
        return None
    if not isinstance(j, dict):
        return None
    pick = lambda *keys: next((str(j[k]).strip() for k in keys if j.get(k)), "")
    items = j.get("items") or j.get("内容") or ""
    if isinstance(items, list):
        items = "\n".join(str(x).strip() for x in items if str(x).strip())
    out = {
        "titleStyle": pick("titleStyle", "排版风格", "style"),
        "bodyStyle": pick("bodyStyle", "正文形式"),
        "title1": pick("title1", "t1", "主标题", "标题1", "标题"),
        "title2": pick("title2", "t2", "副标题", "标题2"),
        "pill": pick("pill", "胶囊"),
        "items": str(items).strip(),
    }
    return out if any(out.values()) else None


def _trim_repeat(txt):
    """小模型常把整段答案重复多遍：只保留第一次【排版风格】到第二次出现之前。"""
    hits = [m.start() for m in re.finditer(r"【\s*排版风格\s*】", txt)]
    if len(hits) > 1:
        cut = txt.rfind("\n", 0, hits[1])
        txt = txt[:cut if cut > 0 else hits[1]]
    return txt


def parse_output(raw):
    """解析模型输出 → {titleStyle, bodyStyle, title1, title2, pill, items, warnings}"""
    warnings = []
    txt = _strip_fences(raw)
    txt = _strip_think(txt)
    txt = _trim_to_format(txt)
    txt = _trim_repeat(txt)
    if not txt:
        raise ValueError("模型返回空内容")

    if "【" not in txt and "[" not in txt:
        j = _try_json(txt)
        if j:
            j["warnings"] = ["模型输出为 JSON，已自动兼容"]
            return _finalize(j, warnings)

    style = _grab(txt, _ALIASES["style"])
    t1 = _grab(txt, _ALIASES["t1"])
    t2 = _grab(txt, _ALIASES["t2"])
    pill = _grab(txt, _ALIASES["pill"])

    body = ""
    m = re.search(r"[【\[]\s*(?:内容|正文)\s*[】\]]\s*[:：]?", txt)
    if m:
        body = txt[m.end():]
    else:
        # 没有【内容】标签：取最后一个已知字段之后的所有行
        last = -1
        for nm in ("胶囊", "副标题", "主标题"):
            mm = re.search(r"[【\[]\s*" + re.escape(nm) + r"\s*[】\]]", txt)
            if mm:
                last = max(last, mm.end())
        if last >= 0:
            body = txt[last:]

    items = "\n".join(_clean_body(body))

    if not t1 and not items:
        j = _try_json(txt)
        if j:
            j["warnings"] = ["未识别到【…】格式，已按 JSON 兼容解析"]
            return _finalize(j, warnings)
        raise ValueError("无法解析模型输出（既不是【…】格式也不是 JSON）：" + txt[:160])

    outs = {
        "style": style, "title1": t1, "title2": t2, "pill": pill,
        "items": items, "titleStyle": "", "bodyStyle": "", "warnings": warnings,
    }
    return _finalize(outs, warnings)


def _finalize(d, warnings):
    title_style = _norm_style(d.get("titleStyle") or d.get("style") or "", _TITLE_STYLES)
    body_style = _norm_style(d.get("bodyStyle") or "", _BODY_STYLES)

    # 有些模型把「主标题 / 数字清单」整体写在 style 里
    if title_style and body_style == "" and "/" in str(d.get("style") or ""):
        parts = re.split(r"[/／]", str(d["style"]))
        if len(parts) >= 2:
            title_style = _norm_style(parts[0], _TITLE_STYLES)
            body_style = _norm_style(parts[1], _BODY_STYLES)

    t1 = (d.get("title1") or "").strip().strip("【】[]")
    t2 = (d.get("title2") or "").strip().strip("【】[]")
    pill = (d.get("pill") or "").strip().strip("【】[]")
    items = (d.get("items") or "").strip()

    # 标题形式与「副标题/胶囊是否存在」互相校准
    if not title_style:
        if pill:
            title_style = "胶囊标签"
        elif t2:
            title_style = "主副标题"
        elif t1:
            title_style = "主标题"
    if title_style == "主标题":
        t2 = "" if t2 else t2
        pill = ""
    elif title_style == "主副标题":
        pill = ""
    elif title_style == "胶囊标签":
        t2 = ""

    if not t1:
        warnings.append("未解析到主标题")
    if not items:
        warnings.append("未解析到内容")
    if not body_style:
        warnings.append("未识别正文形式（不影响出片）")

    res = {
        "titleStyle": title_style,
        "bodyStyle": body_style,
        "title1": t1,
        "title2": t2,
        "pill": pill,
        "items": items,
        "warnings": warnings,
    }
    return res


# ----------------------------------------------------------------------------
# 本地 GGUF（进程内 llama.cpp）
# ----------------------------------------------------------------------------
_COMFY_ROOTS = [
    os.path.join(os.environ.get("LOCALAPPDATA", ""), "Comfy-Desktop", "ComfyUI-Installs", "本地", "ComfyUI"),
    os.path.join(os.environ.get("LOCALAPPDATA", ""), "Comfy-Desktop", "ComfyUI-Installs", "ComfyUI"),
]

_gguf = {"tried": False, "ok": False, "llm": None, "err": "", "path": "", "mmproj": "", "vision": False}
_gguf_lock = threading.Lock()

# 推理串行锁：收件箱监听线程与 HTTP 请求可能同时触发推理，
# 而 llama-cpp-python 的 Llama 实例并非线程安全，必须串行化。
_gen_lock = threading.RLock()

# 停止序列：小模型答完题后常会继续编造多轮对话/重复整段，遇到这些标记立即截断（省时间）。
# 注意：不要在这里停「Assistant:/ASSISTANT:」——模型常先给草稿、再以该标记重开给出更完整的答案，
#       停掉会丢失后半段；改由 _clean_body 分段取最长。这里只停「结尾噪音」。
_STOPS = ["<end>", "\n<|im_end|>",
          "\nUSER:", "\nUSER：", "\n用户：",
          "\n###", "\n---", "\n**",
          "\n自检", "\n再次输出", "\n重新输出", "\n复盘",
          "\n备注：", "\n注：", "\n（注：", "\n(注：", "\n说明：", "\n分析"]


def _gguf_search_dirs():
    dirs = []
    for root in _COMFY_ROOTS:
        if os.path.isdir(root):
            dirs.append(os.path.join(root, "models", "LLM"))
    dirs.append(os.path.join(SCRIPT_DIR, "models", "LLM"))
    dirs.append(os.path.join(SCRIPT_DIR, "models"))
    return [d for d in dirs if os.path.isdir(d)]


def discover_gguf():
    """找一个可用的文本 GGUF（优先最大的非 mmproj 模型），并配对同目录 mmproj。"""
    best, best_size = "", 0
    for base in _gguf_search_dirs():
        for dirpath, _dirs, files in os.walk(base):
            for fn in files:
                if not fn.lower().endswith(".gguf"):
                    continue
                if "mmproj" in fn.lower() or "clip" in fn.lower():
                    continue
                p = os.path.join(dirpath, fn)
                try:
                    sz = os.path.getsize(p)
                except OSError:
                    continue
                if sz > best_size:
                    best, best_size = p, sz
    if not best:
        return "", ""
    mmproj = ""
    d = os.path.dirname(best)
    for fn in sorted(os.listdir(d)):
        if fn.lower().endswith(".gguf") and ("mmproj" in fn.lower()):
            mmproj = os.path.join(d, fn)
            break
    return best, mmproj


def gguf_status(reset=False):
    """返回本地 GGUF 的当前状态（可用性/路径/是否支持视觉），reset=True 时强制重建。"""
    if reset:
        with _gguf_lock:
            if _gguf["llm"] is not None:
                try:
                    _gguf["llm"].close()
                except Exception:
                    pass
            _gguf.update({"tried": False, "ok": False, "llm": None, "err": ""})
    return dict(_gguf)


def _gguf_load(cfg):
    with _gguf_lock:
        if _gguf["tried"]:
            return _gguf
        _gguf["tried"] = True
        path = (cfg.get("rwGgufPath") or "").strip()
        mmproj = (cfg.get("rwGgufMmproj") or "").strip()
        if not path or not os.path.isfile(path):
            path, auto_mm = discover_gguf()
            if not mmproj:
                mmproj = auto_mm
        if not path or not os.path.isfile(path):
            _gguf["err"] = "未找到本地 GGUF 模型（可在设置里指定路径，或安装 Ollama）"
            _gguf["ok"] = False
            return _gguf
        _gguf["path"], _gguf["mmproj"] = path, mmproj
        try:
            from llama_cpp import Llama
        except Exception as e:
            _gguf["err"] = "llama_cpp 不可用：%s" % e
            _gguf["ok"] = False
            return _gguf
        ctx = int(cfg.get("rwGgufCtx") or 8192)
        gpu = int(cfg.get("rwGgufGpuLayers") if str(cfg.get("rwGgufGpuLayers") or "").strip() else -1)
        kwargs = dict(model_path=path, n_ctx=ctx, n_gpu_layers=gpu, verbose=False, logits_all=False)
        last_err = ""
        # 优先带视觉投影（能读图）；失败则退回纯文本
        if mmproj and os.path.isfile(mmproj):
            try:
                from llama_cpp.llama_chat_format import MTMDChatHandler
                handler = MTMDChatHandler(clip_model_path=mmproj, verbose=False)
                _gguf["llm"] = Llama(chat_handler=handler, **kwargs)
                _gguf["vision"] = True
                _gguf["ok"] = True
                log.info("本地 GGUF 已加载(含视觉)：%s", path)
                return _gguf
            except Exception as e:
                last_err = "视觉加载失败(%s)，退回纯文本" % e
                log.warning("GGUF 视觉加载失败，退回纯文本：%s", e)
        try:
            _gguf["llm"] = Llama(**kwargs)
            _gguf["vision"] = False
            _gguf["ok"] = True
            log.info("本地 GGUF 已加载(纯文本)：%s", path)
        except Exception as e:
            _gguf["err"] = ("%s；%s" % (last_err, e)).strip("；")
            _gguf["ok"] = False
            log.warning("本地 GGUF 加载失败：%s", e)
        return _gguf


def _gguf_generate(cfg, system, user, image_data_url=None):
    st = _gguf_load(cfg)
    if not st["ok"]:
        raise RuntimeError(st["err"] or "本地 GGUF 不可用")
    if image_data_url and not st["vision"]:
        raise RuntimeError("本地 GGUF 未加载视觉投影，无法读图")
    msgs = [{"role": "system", "content": system}]
    if image_data_url:
        msgs.append({"role": "user", "content": [
            {"type": "text", "text": user or "请识别图片中的原文，并按格式产出文案。"},
            {"type": "image_url", "image_url": {"url": image_data_url}},
        ]})
    else:
        msgs.append({"role": "user", "content": user})
    kwargs = dict(
        messages=msgs,
        temperature=float(cfg.get("rwTemp") or 0.4),
        top_p=0.9,
        repeat_penalty=1.05,
        max_tokens=int(cfg.get("rwMaxTokens") or 900),
        stop=_STOPS,
    )
    try:
        # reasoning_budget=0：强制结束思考块（Qwen3/3.5 等默认开思考，会吃掉全部输出）
        out = st["llm"].create_chat_completion(reasoning_budget=0, **kwargs)
    except TypeError:
        out = st["llm"].create_chat_completion(**kwargs)
    return (out["choices"][0]["message"]["content"] or "").strip()


# ----------------------------------------------------------------------------
# OpenAI 兼容服务（本地 Ollama/LM Studio，或在线 API）
# ----------------------------------------------------------------------------
def _thinking_param(base, cfg, model=""):
    """DeepSeek 思考模式的开关参数（其他厂商不发送，避免 400）。

    新版 DeepSeek 默认开启思考模式：会先吐一大段 CoT（走 reasoning_content），
    而我们的停止序列含 `\\n**`/`\\n###`/`\\n注：` 等，极易命中 CoT 导致正文为空。
    因此对 DeepSeek 端点默认显式关闭。

    rwApiThinking: ""=自动（DeepSeek 关、其他不发） / "0"=强制关 / "1"=开启。
    """
    mode = str(cfg.get("rwApiThinking") or "").strip()
    if mode == "1":
        return {"type": "enabled"}
    if mode == "0":
        return {"type": "disabled"}
    probe = ((base or "") + " " + (model or "")).lower()
    if "deepseek" in probe:
        return {"type": "disabled"}
    return None


def _openai_chat(base, api_key, model, system, user, cfg, image_data_url=None, timeout=300,
                 vendor_tweaks=False):
    import urllib.request
    import urllib.error
    url = base.rstrip("/") + "/chat/completions"
    if image_data_url:
        umsg = {"role": "user", "content": [
            {"type": "text", "text": user or "请识别图片中的原文，并按格式产出文案。"},
            {"type": "image_url", "image_url": {"url": image_data_url}},
        ]}
    else:
        umsg = {"role": "user", "content": user}
    payload = {
        "model": model,
        "messages": [{"role": "system", "content": system}, umsg],
        "temperature": float(cfg.get("rwTemp") or 0.4),
        "top_p": 0.9,
        "max_tokens": int(cfg.get("rwMaxTokens") or 900),
        "stop": _STOPS[:16],   # DeepSeek 等在线接口限制 stop 数组 ≤16 条
        "stream": False,
    }
    if vendor_tweaks:
        th = _thinking_param(base, cfg, model)
        if th is not None:
            payload["thinking"] = th
    req = urllib.request.Request(url, data=json.dumps(payload).encode("utf-8"), method="POST")
    req.add_header("Content-Type", "application/json")
    if api_key:
        req.add_header("Authorization", "Bearer " + api_key)
    try:
        with urllib.request.urlopen(req, timeout=timeout) as r:
            d = json.loads(r.read().decode("utf-8"))
    except urllib.error.HTTPError as e:
        raise ValueError("HTTP %s: %s" % (e.code, e.read().decode("utf-8", "replace")[:300]))
    try:
        msg = d["choices"][0]["message"] or {}
    except Exception as e:
        raise ValueError("响应格式异常: %s" % json.dumps(d, ensure_ascii=False)[:300]) from e
    content = (msg.get("content") or "").strip()
    if not content:
        reasoning = (msg.get("reasoning_content") or "").strip()
        if reasoning:
            raise ValueError("模型只返回了思考过程、正文为空（思考模式被停止序列截断）；"
                             "请在设置里把「API 思考模式」设为 关闭（rwApiThinking=0），"
                             "或加大「最大输出 token」")
        raise ValueError("返回空")
    return content


# ----------------------------------------------------------------------------
# 本地服务自愈：Ollama 掉线时自动拉起
# ----------------------------------------------------------------------------
_ensure_lock = threading.Lock()
_ensure_ts = 0.0


def _port_open(host, port, timeout=1.0):
    import socket
    try:
        with socket.create_connection((host, port), timeout=timeout):
            return True
    except Exception:
        return False


def _find_ollama():
    """找 ollama.exe：优先安装目录，其次 PATH。"""
    la = os.environ.get("LOCALAPPDATA") or ""
    pf = os.environ.get("ProgramFiles") or ""
    cands = [
        os.path.join(la, "Programs", "Ollama", "ollama.exe"),
        os.path.join(pf, "Ollama", "ollama.exe"),
    ]
    for c in cands:
        if c and os.path.isfile(c):
            return c
    for d in (os.environ.get("PATH") or "").split(os.pathsep):
        if d:
            p = os.path.join(d, "ollama.exe")
            if os.path.isfile(p):
                return p
    return ""


def _cleanup_orphan_runners(ollama_exe):
    """清理 Ollama 的僵尸 runner。

    Ollama 的 llama-server.exe 是独立进程；服务端异常退出（桌面版关窗）时它可能
    残留并一直占着显存。此处仅清理可执行文件位于 Ollama 安装目录、且父进程已死的
    llama-server.exe，避免误伤其他程序。
    """
    import subprocess
    d = os.path.dirname(os.path.abspath(ollama_exe)).replace("'", "''")
    ps = (
        "$d='" + d + "';"
        "Get-CimInstance Win32_Process -Filter \"Name='llama-server.exe'\" -ErrorAction SilentlyContinue | "
        "Where-Object { $_.ExecutablePath -and $_.ExecutablePath.ToLower().StartsWith($d.ToLower()) } | "
        "ForEach-Object { $p=$_.ParentProcessId; "
        "if(-not (Get-Process -Id $p -ErrorAction SilentlyContinue)){ "
        "Stop-Process -Id $_.ProcessId -Force -ErrorAction SilentlyContinue; "
        "Write-Output ('killed ' + $_.ProcessId) } }"
    )
    try:
        r = subprocess.run(["powershell", "-NoProfile", "-NonInteractive", "-Command", ps],
                           capture_output=True, text=True, timeout=40)
        killed = [x for x in (r.stdout or "").split() if x.isdigit()]
        if killed:
            log.info("清理 Ollama 僵尸 runner：pid %s", ", ".join(killed))
    except Exception as e:
        log.warning("清理僵尸 runner 失败：%s", e)


def _ensure_local_server(cfg, wait=120):
    """本地服务未监听时尝试自动启动 Ollama（仅 127.0.0.1/localhost 生效）。

    桌面版 Ollama 是「托盘程序 + 子进程」，窗口一关服务就停；这里用
    `ollama serve` 直接拉起常驻服务（后台、无窗口），带 60 秒冷却避免反复拉起。
    """
    global _ensure_ts
    base = (cfg.get("rwLocalBase") or "").strip()
    if not base or str(cfg.get("rwAutoStartLocal", "1")) == "0":
        return
    m = re.match(r"^https?://([^/:]+)(?::(\d+))?", base, re.I)
    if not m:
        return
    host = (m.group(1) or "").lower()
    port = int(m.group(2) or (443 if base.lower().startswith("https") else 80))
    if host not in ("127.0.0.1", "localhost", "::1", "0.0.0.0"):
        return
    if _port_open(host, port):
        return
    with _ensure_lock:
        if _port_open(host, port):
            return
        now = time.time()
        if now - _ensure_ts < 60:
            return
        _ensure_ts = now
        exe = _find_ollama()
        if not exe:
            log.warning("本地服务 %s:%d 未监听，且未找到 ollama.exe；请手动启动 Ollama", host, port)
            return
        import subprocess
        env = dict(os.environ)
        env.setdefault("OLLAMA_HOST", "%s:%d" % (host, port))
        env.setdefault("OLLAMA_CONTEXT_LENGTH", "8192")
        env.setdefault("OLLAMA_FLASH_ATTENTION", "1")
        env.setdefault("OLLAMA_KV_CACHE_TYPE", "q8_0")
        env.setdefault("OLLAMA_KEEP_ALIVE", "10m")
        log.info("本地服务 %s:%d 未监听 → 自动启动 ollama serve（%s）", host, port, exe)
        _cleanup_orphan_runners(exe)
        try:
            logf = open(os.path.join(SCRIPT_DIR, "ollama_serve.log"), "ab")
            subprocess.Popen([exe, "serve"], stdout=logf, stderr=logf, stdin=subprocess.DEVNULL,
                             env=env, close_fds=True,
                             creationflags=0x00000008 | 0x00000200)  # DETACHED | NEW_PROCESS_GROUP
        except Exception as e:
            log.warning("自动启动 ollama 失败：%s", e)
            return
        t0 = time.time()
        while time.time() - t0 < wait:
            if _port_open(host, port):
                log.info("ollama 已就绪（%.0f 秒）", time.time() - t0)
                return
            time.sleep(1)
        log.warning("ollama 启动超时（%d 秒），本次回退其他通道", wait)


# ----------------------------------------------------------------------------
# V2.41 四段式 prompt（《飞书用prompt.txt》）＋ 硬指标程序化兜底
#
# 为什么要在程序里兜底：本地小模型（qwen2.5vl:7b）数不清汉字，也常把
# 「{编号 名称}」和说明挤在同一行。实测（原文 395 字，《飞书用prompt.txt》原文调用）：
#     · 正文要求「{} 独占一行」   → 有时 3 条、有时 0 条（不确定性）
#     · 胶囊要求 ≤8 字            → 给出 21 字（第 2 次 14 字）
#     · 标题2 要求 6~10 汉字      → 给出 13 字
#   并且往 system prompt 里再追加「硬约束」反而更差（独占一行条数 3 → 0）。
# 因此：用户 prompt 原文照用、不加料；长度/换行/编号等硬指标一律在程序里确定性修正。
# ----------------------------------------------------------------------------
_SECTION_KEYS = ("内容", "标题1", "标题2", "胶囊")

# 单段提示词（如《内容生成prompt.txt》）只规定正文清单，没有标题1/标题2/胶囊。
# 这时代码按同一份文件生成正文，再用下面三条内置指令补齐标题与胶囊，
# 保证工作台拿到的仍是完整的四要素；正文规则始终以用户文件为准。
_TITLE1_ASK = (
    "你是短视频信息流文案编辑。根据用户给出的原文，写 1 个 4~10 个字的主标题："
    "点明人群和机会（如「应届生」「专科」「央企」），可带数字，口语化、有吸引力；"
    "不要标点、不要引号、不要 emoji、不要「震惊 / 速看」这类营销腔，不要解释。只输出标题。"
)
_TITLE2_ASK = (
    "你是短视频信息流文案编辑。根据用户给出的原文，写 1 个 6~10 个字的副标题："
    "补充主标题没提到的条件、对象或细节，不要与主标题重复核心词；"
    "不要标点、不要引号、不要 emoji，不要解释。只输出副标题。"
)
_PILL_ASK = (
    "你是短视频信息流文案编辑。根据原文和正文，写 1 个不超过 8 个字的胶囊标签："
    "用作开头 3 秒的钩子，优先用原文里真实出现的数字、时间或对象，"
    "不要与标题重复，不要标点、不要引号、不要 emoji，不要解释。只输出标签。"
)


def _is_body_only_prompt(text):
    """提示词是否只规定正文（没有标题1 / 标题2 / 胶囊的产出要求）。

    典型：《内容生成prompt.txt》只写「{编号 名称} + 说明 + 结尾引导语」，
    不含「主标题 / 副标题 / 胶囊 / 排版风格 / 标题1 / 标题2」这些字样。
    """
    t = text or ""
    if any(k in t for k in ("排版风格", "主标题", "副标题", "标题1", "标题2", "胶囊")):
        return False
    return "内容" in t or "正文" in t


_EMOJI_RE = re.compile("[\U0001F000-\U0001FAFF\u2190-\u21FF\u2600-\u27BF\u2B00-\u2BFF\uFE0F\u200d]")
_TRIM_CHARS = " \t\u3000“”\"'‘’「」『』《》〈〉（）()【】[]{}:：,，.。!！?？;；、~～-—_*#"
# 成对符号（引号/括号/书名号）交给「配平」逻辑处理，**不能当尾巴噪声 strip**：
# 先 strip 掉配对的一半，下一次配平就会把另一半也删掉（实测说明里的「（不限专业 / 户籍）」
# 被剥成「不限专业 / 户籍」、标题「上海留学生的“百人工程”」被剥成带半个引号）。
_PAIR_CHARS = "“”‘’「」『』《》〈〉（）()【】[]{}｛｝"
_TAIL_TRIM = "".join(ch for ch in _TRIM_CHARS if ch not in _PAIR_CHARS)
# 规范化「条目名」时要连花括号一起清掉（名称不该带括号），与上面的尾巴噪声不同用途
_NAME_TRIM = _TAIL_TRIM + "{}｛｝【】[]"
# {01 名称} / 【1 名称】 / ｛01 名称｝ 都认（行首锚定版，用于「下一行是不是条目」判断）
# 注意：数字后必须跟「空格」或「.、:：」分隔符，否则 {01} 会被 \d{1,2} 回溯成 0 + 名称「1」
_BULLET_RE = re.compile(r"^[{\[｛【]\s*\d{1,2}\s*(?:[.、:：]|[ \t])\s*([^}\]｝】]+?)\s*[}\]｝】]")
# 全文宽松扫描版：不锚定行首，从任意位置抓「{编号 名称} + 同行说明」
# （模型常把答案包在 ``` 里、或先寒暄/自检一大段，严格按行清洗会提前中断而丢正文）
_BULLET_ANY = re.compile(r"[{\[｛【]\s*(\d{1,2})\s*(?:[.、:：]|[ \t])\s*([^}\]｝】\n]{1,30}?)\s*[}\]｝】]")
# 兜底①：模型把编号单独包起来、名称写在花括号后面 ——「{01} 入党流程」
_BRACE_NUM_LINE = re.compile(r"(?m)^[ \t]*[{\[｛【]\s*(\d{1,2})\s*[}\]｝】][ \t]*(\S[^\n]{0,26})$")
# 兜底②：模型忘了花括号但写了「01 名称」这种编号行（含「1. 名称」「1、名称」）
_NUM_LINE = re.compile(r"(?m)^[ \t]*(\d{1,2})\s*[.、)）:：]?\s+(\S[^\n]{0,26})$")
# 兜底②b：编号列表的「名称 说明」写在同一行、且说明很长（超上面 26 字上限）
# （实测素材1：`1. 中国有色矿业 矿业冷门，无笔试，采矿/地质等专业招，编制+年薪18-30万…`）
_NUM_LINE_LONG = re.compile(r"(?m)^[ \t]*(\d{1,2})\s*[.、)）:：]\s*(\S[^\n]*)$")
# 走样：编号自己进花括号、名称与说明跟在后面同一行 ——「{02} 竞争强度：大一开学提交申请…」
# （实测素材2 出现；若不预处理，「名称：说明」会被下一条吸成「说明」，输出里就会留一行残废的 `{02} …`）
_ENTRY_PREFIX_RE = re.compile(r"(?m)^[ \t]*[{\[｛【]\s*(\d{1,2})\s*[}\]｝】][ \t]*(\S[^\n]*)$")
# 某些模型把编号单独放进花括号、名称丢到下一行（实验 V3）——判定为条目行
_CTA_RE = re.compile(r"赶紧|快|转告|转给|告诉|收藏|存下|存起来|先存|用得上|别错过|码住|记下"
                     r"|分享|发给|发到|收好|存好|留给|身边人|记得")
# 寒暄/解释行（取「一句」时要跳过）
_JUNK_LINE_RE = re.compile(r"[:：]\s*$|^(?:以下|如下|输出|好的|可以|建议|根据|原文|以下是)")
# prompt 回显：模型偶尔把指令原文（「8~14 个字，催观众做一个动作」）当成内容输出
_ECHO_RE = re.compile(r"催观众|做一个动作|不要问号|不要提问|点赞关注|只输出|占位词|\d+\s*[~～-]\s*\d+\s*个字")
# 最近用过的引导语：规则 17 要求同一批不重复（本引擎按行逐条生成，靠这里记住上一批）
_recent_cta = collections.deque(maxlen=8)


def _han(s):
    """汉字个数（规则里的「字」都指汉字）。"""
    return len(re.findall(r"[\u4e00-\u9fff]", s or ""))


def _overlap(a, b):
    """a 的汉字有多大比例也出现在 b 里（用来判断标题1/标题2 是否语义重复）。"""
    ca = set(re.findall(r"[\u4e00-\u9fff]", a or ""))
    if not ca:
        return 0.0
    return len(ca & set(re.findall(r"[\u4e00-\u9fff]", b or ""))) / len(ca)


def split_prompt_sections(text):
    """把 prompt 文件拆成 4 段：内容 / 标题1 / 标题2 / 胶囊。

    兼容「-----」分隔线 + 「内容prompt:」小标题的写法（即《飞书用prompt.txt》）。
    拆不出完整 4 段就返回 None，调用方退回到旧的「单次合并调用」。
    """
    found = {}
    for part in re.split(r"(?m)^\s*-{5,}\s*$", text or ""):
        p = (part or "").lstrip()
        # 段首必须是「内容prompt:」/「标题1prompt」/「胶囊prompt」这类独占一行的标签
        # （实测《飞书用prompt.txt》里 标题1/标题2/胶囊 三段都**没有**结尾冒号）
        m = re.match(r"^(内容|标题1|标题2|胶囊)\s*(?:prompt|Prompt|PROMPT)?\s*[:：]?[ \t]*(?:\r?\n|$)", p)
        if not m:
            continue
        k = m.group(1)
        if k in found:
            continue
        body = p[m.end():].strip()
        if body:
            found[k] = body
    if all(found.get(k) for k in _SECTION_KEYS):
        return found
    return None


def _clean_single(raw):
    """从原始输出里取「一句」：去围栏/思考链/标签前缀/emoji/首尾引号标点。"""
    t = _strip_think(_strip_fences(raw or ""))
    picked = ""
    for ln in t.splitlines():
        s = ln.strip()
        if not s:
            continue
        s = re.sub(r"^[【\[]\s*(?:主标题|标题1|标题一|副标题|标题2|标题二|胶囊|内容|正文)\s*[】\]]\s*[:：]?\s*", "", s)
        s = re.sub(r"^(?:主标题|副标题|标题1|标题2|标题一|标题二|胶囊|内容)\s*[:：]\s*", "", s)
        s = _EMOJI_RE.sub("", s).strip().strip(_TRIM_CHARS).strip()
        if not s:
            continue
        if not picked and not _JUNK_LINE_RE.match(s) and not _BULLET_RE.match(s):
            picked = s
        if picked:
            break
    if picked:
        return _drop_dangling(picked)
    for ln in t.splitlines():                      # 全是寒暄行时退回第一行
        s = _EMOJI_RE.sub("", ln.strip()).strip(_TRIM_CHARS).strip()
        if s:
            return _drop_dangling(s)
    return ""


# 标题/胶囊受影响最深的「结尾词」：硬裁前先看能不能整词去掉（避免把词切成两半）
_TITLE_SUFFIX = (
    "时间表", "时间安排", "计划", "攻略", "指南", "安排", "汇总", "盘点", "名单", "名单汇总",
    "机会", "方向", "条件", "要求", "待遇", "薪资", "报名", "流程", "步骤", "解析", "解读",
    "提醒", "须知", "合集", "一览", "专场", "答疑", "大全", "门道", "建议", "推荐", "要点",
    # 招聘/考公类标题最常见的结尾词（实测「上海留学生“百人工程”招聘」被切成「…招」）
    "招聘", "招募", "招考", "招人", "岗位", "职位", "名额", "编制", "补贴", "福利", "年薪",
    "学历", "专业", "年龄", "院校", "截止", "时间", "公告", "通知", "详情", "门槛", "攻略",
    "优势", "趋势", "前景", "渠道", "入口", "清单", "汇总表", "一览表", "时间表", "机会",
)


def _shorten_text(s, target):
    """把一段说明压到 ~target 字：**从头保留**，在标点处收尾，并清掉落单的括号/引号。

    原先直接调 `_fit_len(desc, 8, target)`，它会「取第一个落进区间的片段」——
    长说明里往往是**中间**某个片段合格，于是产出「具留服认证的本科及以上留学生（不限专业」
    这种既不从头读、又断在括号里的残句。压缩说明应该保头去尾。
    """
    s = (s or "").strip()
    vis = lambda x: len(re.sub(r"\s", "", x))
    if vis(s) <= target:
        return s
    parts = re.split(r"(?<=[，,；;。！!？?、])", s)      # 切句但保留标点
    out = ""
    for p in parts:
        if not out:
            out = p
            continue
        if vis(out + p) > target:
            break
        out += p
    if not out:
        out = s
    out = out.strip(_TAIL_TRIM).strip()
    while vis(out) > target and len(out) > 1:          # 还是超 → 从尾部逐字收（配合去悬挂符号）
        out = _drop_dangling(out[:-1])
    return _drop_dangling(out) or s[:max(1, target)]


def _fit_len(text, lo, hi, hard_max=None):
    """把标题/胶囊裁到「汉字数」落进 [lo, hi]：先按标点切句，再整词去尾，最后才硬裁。"""
    s = (text or "").strip().strip(_TRIM_CHARS).strip()
    if not s:
        return ""
    n = _han(s)
    if n <= hi:
        return s
    hard = hard_max or hi
    segs = [x.strip(_TRIM_CHARS).strip() for x in re.split(r"[，,；;。！!？?、／/\s：:—–～]+", s) if x.strip(_TRIM_CHARS).strip()]
    for c in segs:                                  # ① 有整段就落在区间内的，直接用
        if lo <= _han(c) <= hi:
            return c
    acc, accn = [], 0                               # ② 否则从前往后整段累加，凑够上限
    for c in segs:
        if accn + _han(c) > hi:
            break
        acc.append(c)
        accn += _han(c)
    if acc and accn >= lo:
        return _drop_dangling("，".join(acc))
    out, cnt = [], 0                                # ③ 最后才汉字级硬裁
    # 硬裁最容易把词切成两半（实测「邯郸大学生就业帮扶计」「天赋型选手：突出能力轻」）。
    # 先试试「去掉一个常见结尾词」能不能刚好落进区间 —— 不行再按字硬裁。
    for suf in _TITLE_SUFFIX:
        if len(s) > len(suf) and s.endswith(suf) and lo <= _han(s[:-len(suf)]) <= hi:
            return _drop_dangling(s[:-len(suf)].strip(_TAIL_TRIM).strip())
    for ch in s:
        if "\u4e00" <= ch <= "\u9fff":
            if cnt >= hard:
                break
            cnt += 1
        out.append(ch)
    cut = _drop_dangling("".join(out).strip(_TAIL_TRIM).strip())
    # 别把「揭秘3大冷门但待遇」这种半截短语交出去：回退到最后一个连接词之前
    for i in range(len(cut) - 1, 0, -1):
        if cut[i] in "但而和与及或·、":
            head = cut[:i].strip(_TAIL_TRIM).strip()
            if _han(head) >= max(lo, 4):       # 回退后仍得是个完整短语
                cut = head
            break
    return cut or s[:hard]


def _drop_dangling(s):
    """裁完可能留下「半个」引号/括号（如「揭秘央国企校招“冷」），这里把落单的左符号去掉。

    注意：**成对的括号/引号要留着** —— 「上海留学生“百人工程”」结尾的 `”`、说明结尾的 `）`
    都不是噪声；它们只在「没了另一半」时才该被删。
    """
    for op, cl in (("“", "”"), ("‘", "’"), ("「", "」"), ("『", "』"), ("（", "）"), ("(", ")"),
                   ("【", "】"), ("[", "]"), ("｛", "｝"), ("{", "}"), ("《", "》"), ("〈", "〉")):
        while s.count(op) > s.count(cl):
            i = s.rfind(op)
            s = s[:i] + s[i + 1:]
    return s.strip(_TAIL_TRIM).strip()


def _looks_like_cta(s):
    """像不像一句合格引导语：8~14 字（规则 10）+ 含催促动作词（规则 12）。"""
    if not s or not (8 <= _han(s) <= 14) or not _CTA_RE.search(s):
        return False
    # 排除模型把指令原文回显成「引导语」的情况
    if re.search(r"\d+\s*[~～-]\s*\d+\s*字|不要问句|不要提问|点赞关注|例如|占位", s):
        return False
    return True


def _cut_noise(t):
    """只砍掉「自检/说明」段落（模型爱在正文后附一段自我检查），保留其余一切。"""
    out = []
    for ln in (t or "").splitlines():
        if _NOTE_RE.match(ln.strip()):
            break
        out.append(ln.replace("**", ""))
    return "\n".join(out)


def _render_body(items, cta):
    out = []
    for name, desc in items:
        out.append(name)
        if desc:
            out.append(desc)
    if cta:
        out.append(cta)
    return "\n".join(out)


def _example_items(section):
    """抠出 内容prompt 里自带的示例条目（名称 → 说明）。

    用户的 prompt 里有 few-shot 示例（「{01 中国有色矿业} / 冷门央企…年薪18-30万」）。
    实测本地小模型遇到不合拍的原文（如「国考时间安排」）会**整段照抄这个示例**，
    凭空编造出「中国有色矿业招人」，直接违反「零编造」。所以必须识别并按不合格处理。
    """
    out = {}
    lines = (section or "").splitlines()
    for i, ln in enumerate(lines):
        s = ln.strip()
        m = _BULLET_ANY.match(s) if s else None
        if not m:
            continue
        nm = re.sub(r"\s", "", m.group(2))
        desc = s[m.end():].strip()
        if not desc:
            for j in range(i + 1, len(lines)):
                t = lines[j].strip()
                if t and not _BULLET_ANY.match(t):
                    desc = t
                    break
        out[nm] = desc
    return out


def _is_entry_line(s):
    """这一行是不是「条目行」（用于判断上一条的说明到哪儿结束）。"""
    s = (s or "").strip()
    if not s:
        return False
    return bool(_BULLET_RE.match(s) or _BULLET_ANY.match(s)
                or _BRACE_NUM_LINE.match(s) or _NUM_LINE.match(s)
                or _NUM_LINE_LONG.match(s) or _ENTRY_PREFIX_RE.match(s))


def _pre_normalize(text):
    """把「编号自己进花括号」的走样改写成规范形态，交给统一的解析器处理。

    实测素材2 的原始输出：
        {01} 时间跨度：每个班级每年仅2-3个名额，周期至少三年。
        {02} 竞争强度：大一开学提交申请，优先班长、团支书及成绩前三。
    改写为：
        {01 时间跨度}
        每个班级每年仅2-3个名额，周期至少三年。
    不做这一步的话，「{02} 竞争强度：…」既不算条目行、又会被上一条当成「说明」，
    结果正文里会留下一行残废的 `{02} 竞争强度：…`（实测复现）。
    """

    def sub(m):
        num = int(m.group(1))
        # 名称里不该出现花括号；走样时可能出现「{02} 名称} 说明」，先清掉，否则会让
        # 工作台的「{} 独占一行」判定失效（实测会整段被判成 0 条）
        tail = re.sub(r"[\[\]{}｛｝【】]", " ", m.group(2)).strip()
        mm = re.match(r"^([^：:]{2,12})\s*[：:]\s*(.{2,})$", tail)
        # 名称部分必须含至少 2 个汉字：否则会被「00:16」这类时间码误拆（实测复现）
        if mm and _han(mm.group(1)) >= 2:
            return "{%02d %s}\n%s" % (num, mm.group(1).strip(), mm.group(2).strip())
        return "{%02d %s}" % (num, tail)        # 没有合格冒号 → 整段当名称（超长后面会截）

    return _ENTRY_PREFIX_RE.sub(sub, text)


def _desc_ok(ds):
    """这一条的「说明」算不算数。

    汉字 ≥3 就算；纯日期/数字的说明（「10月14日」只有 2 个汉字）也算 ——
    实测「国考时间表」这类原文，说明天然就是日期，若判成「没说明」会把合格清单误杀。
    """
    s = re.sub(r"\s", "", ds or "")
    return _han(s) >= 3 or len(s) >= 5


def _fit_total(items, cta, warnings):
    """全文 ≤140 字：先减条目，再对最长的说明对半压缩，最后收一下引导语长度。

    抽成独立函数是因为四段式链路里**引导语可能被换掉**（从别的候选版本择优 / 补写），
    换完长度会变，必须再收敛一次（实测曾因此冲到 153 字）。
    返回 (items, cta, chars)。items 会被就地修改（弹掉末尾条目 / 截短说明）。
    """
    def total():
        return len(re.sub(r"\s", "", _render_body(items, cta)))

    while len(items) > 3 and total() > 140:
        items.pop()
        warnings.append("全文超 140 字，已从末尾减少条目")
    guard = 0
    while items and total() > 140 and guard < 60:
        guard += 1
        longest = max(items, key=lambda it: len(re.sub(r"\s", "", it[1])))
        cur = len(re.sub(r"\s", "", longest[1] or ""))
        if cur <= 8:
            break
        target = max(8, cur // 2)              # 每次砍一半，4 次左右收敛
        longest[1] = _shorten_text(longest[1], target)
        if len(re.sub(r"\s", "", longest[1])) >= cur:
            break
        warnings.append("全文超 140 字，已精简说明")

    if cta:
        cta = _fit_len(cta, 1, 14, hard_max=14)
    return items, cta, total()


def normalize_body(raw, pair=False, example_items=None):
    """把「内容prompt」的输出规范成工作台可渲染的清单。

    · {编号 名称} 独占一行、说明另起一行（模型挤在一行时确定性拆开）
    · 编号统一成 {01 名称} / {02 名称} …
    · 末行单独作为结尾引导语（8~14 字）
    · 全文 ≤140 字：超了先删条目，再对半压缩说明
    返回 {items, cta, n, chars, rows, has_desc, copied, warnings}

    pair=True：兜底解读。模型偶尔把「名称」和「说明」**都**写成条目行
    （实测「{01} 提交入党申请 / {18} 年满十八岁共青团员主动提交」），
    此时把相邻两条重新配对成（名称, 说明）。仅在常规解析拿不到任何说明时使用。
    example_items：内容prompt 自带的示例条目（名称 → 说明）。某条的「名称命中示例 + 说明与示例
    高度重合」即判定「照抄示例」（copied），按不合格处理（零编造）。
    实现要点：**全文宽松扫描**而不是逐行清洗。实测模型常先寒暄、或把答案包在代码块里、
    或在正文后附「自检」段落；逐行严格清洗会提前中断，导致正文整段丢失（实测素材2 变 0 条）。
    """
    warnings = []
    text = _pre_normalize(_cut_noise(_strip_think(_strip_fences(raw or ""))))
    lines = text.splitlines()

    hits = list(_BULLET_ANY.finditer(text))
    entries = []            # [名称, 说明]
    used = set()            # 已被条目占用的行号

    if hits:
        for i, m in enumerate(hits):
            name = m.group(2).strip()
            ln_start = text.count("\n", 0, m.start())
            ln_end = text.count("\n", 0, m.end())
            eol = text.find("\n", m.end())
            eol = len(text) if eol < 0 else eol
            stop = hits[i + 1].start() if i + 1 < len(hits) else eol
            rest = text[m.end():min(eol, stop)].strip()   # 同一行、且不属于下一条目的部分
            desc, hi = "", ln_end
            if rest:
                desc = rest
            else:                                          # 说明在下一行
                j = ln_end + 1
                while j < len(lines):
                    s = lines[j].strip()
                    if s:
                        # 下一条目 / prompt 回显 / 结尾引导语 → 本条没有说明（回显不能当成说明）；
                        # 引导语尤其不能吸成说明（实测「{04 面试时间} / 转告给身边的朋友」），
                        # 留着不动，由下面「从末尾往上找引导语」把它当引导语用。
                        if _is_entry_line(s) or _ECHO_RE.search(s) or _looks_like_cta(s):
                            break
                        desc, hi = s, j
                        break
                    j += 1
            entries.append([name, desc])
            used.update(range(ln_start, hi + 1))
    else:
        # 兜底：编号与名称没按「{01 名称}」写（实测两种走样：{01} 名称 / 01 名称）
        # 三种候选模式都跑一遍，取命中最多的一种（长的「1. 名称 说明」和短的会混着出现）
        hits2, kind = [], ""
        for pat, k in ((_BRACE_NUM_LINE, "brace"), (_NUM_LINE, "num"), (_NUM_LINE_LONG, "num_long")):
            h = list(pat.finditer(text))
            if len(h) > len(hits2):
                hits2, kind = h, k
        for m in hits2:
            ln = text.count("\n", 0, m.start())
            desc, hi = "", ln
            # 「1. 名称 说明」这种**同行**写法：说明就在 m.group(2) 里（后面交给 norm 按空格拆），
            # 不能再往下吸一行 —— 实测那样会把下一条目的正文、甚至结尾引导语吸成说明
            if kind != "num_long":
                j = ln + 1
                while j < len(lines):
                    s = lines[j].strip()
                    if s:
                        if _is_entry_line(s) or _ECHO_RE.search(s) or _looks_like_cta(s):
                            break
                        desc, hi = s, j
                        break
                    j += 1
            entries.append([m.group(2).strip(), desc])
            used.update(range(ln, hi + 1))
        if entries:
            warnings.append("模型未按「{编号 名称}」格式书写，已自动纠正")

    # 结尾引导语：从末尾往上找第一行非空、且不属于任何条目的文字
    cta = ""
    for i in range(len(lines) - 1, -1, -1):
        s = lines[i].strip()
        if not s or _ECHO_RE.search(s):
            continue
        if i in used:
            break
        cta = s
        break

    # 兜底解读：名称与说明都被写成了条目行 → 相邻两行重新配对
    if pair and len(entries) >= 4:
        entries = [[entries[i][0], entries[i + 1][0]] for i in range(0, len(entries) - 1, 2)]
        # 配对是按位置硬配的，可能把结尾引导语配成最后一条的说明（实测「{04 面试时间} / 转告给身边的朋友」）
        # → 把像引导语的那半还回引导语，别当说明。
        for _e in entries:
            if _e[1] and _looks_like_cta(_e[1]):
                if not _looks_like_cta(cta):
                    cta = _e[1]
                _e[1] = ""
        warnings.append("检测到「名称与说明都写成了条目行」，已按相邻两行重新配对")

    # 编号归一化 + 去空名 + 名称长度兜底 + 「名称：说明」拆开
    norm = []
    for name, desc in entries:
        nm = re.sub(r"\s+", " ", name).strip(" ：:，,。.、")
        nm = re.sub(r"[\[\]{}｛｝【】]", "", nm).strip()
        if not desc:
            # 模型把「名称：说明」写在同一行（实测「{02} 城市选择：杭州、上海…」）
            m2 = re.match(r"^([^：:]{2,12})\s*[：:]\s*(.{2,})$", nm)
            if m2 and _han(m2.group(1)) >= 2:          # 别把「00:16」这种时间码误拆
                nm, desc = m2.group(1).strip(), m2.group(2).strip()
        if _han(nm) > 12:                      # 名称应短；过长时先试「名称 说明」空格拆，再按标点取第一段
            parts = nm.split()
            cut = max((k for k in range(1, len(parts))
                       if 2 <= _han(" ".join(parts[:k])) <= 12), default=0)
            if cut:
                tail2 = " ".join(parts[cut:])
                desc = (desc + " " + tail2).strip() if desc else tail2
                nm = " ".join(parts[:cut])
                warnings.append("条目名称过长，已从「名称 说明」里拆出名称")
            else:
                nm = _fit_len(nm, 1, 12, hard_max=12)
                warnings.append("条目名称过长，已截断到 12 字")
        if _han(nm) == 0:                      # 纯数字/符号的「名称」是噪声（实测模型吐过 00:16 / 00:31）
            continue
        if re.fullmatch(r"引导语|结尾引导语|结尾语|导语|CTA", nm):
            continue                           # 模型偶尔把「结尾引导语」本身写成一条清单（实测）
        if nm:
            norm.append([nm, desc])
    # 去重（先做，避免重复条目挤占 4 条名额）
    # 小模型偶尔陷入重复循环（实测连出 40 条同名条目）
    seen, dedup = set(), []
    for nm, desc in norm:
        key = re.sub(r"\s", "", nm)
        if key in seen:
            continue
        seen.add(key)
        dedup.append([nm, desc])
    if len(dedup) < len(norm):
        warnings.append("清单有重复条目，已去重")
    norm = dedup
    if len(norm) > 4:
        warnings.append("清单 %d 条，已截取前 4 条" % len(norm))
        norm = norm[:4]
    items = [["{%02d %s}" % (k, nm), ds] for k, (nm, ds) in enumerate(norm, 1)]
    # 「有说明」的判定要严一点：模型常只给最后一条挂一句无关的话（实测把引导语「转告」当成了说明），
    # 那样会让质量门误判为合格、跳过补救流程。3 条以上时允许缺 1 条。
    n_desc = sum(1 for _, ds in items if _desc_ok(ds))
    need = len(items) - 1 if len(items) >= 3 else len(items)
    has_desc = bool(items) and n_desc >= need
    if items and not has_desc:
        warnings.append("清单缺少说明文字（每项应为「{名称} + 说明」两条）")
    # 照抄 prompt 示例 → 内容与原文无关，属于编造，按不合格处理。
    # 判定条件：某条的名称命中示例，且说明与示例说明高度重合（≥70% 汉字来自示例）。
    ex = example_items or {}
    copied = False
    for nm, ds in items:
        m = _BULLET_ANY.match(nm)               # nm 是渲染后的「{01 名称}」，取回裸名称
        key = re.sub(r"\s", "", m.group(2) if m else nm)
        if key in ex and ex[key] and _overlap(ds, ex[key]) >= 0.7:
            copied = True
            break
    if copied:
        warnings.append("疑似照抄 prompt 里的示例（内容与原文无关），已重试")

    def total():
        return len(re.sub(r"\s", "", _render_body(items, cta)))

    # 条数下限 & 全文 ≤140 字：先减条目，再对最长的说明对半压缩
    items, cta, chars = _fit_total(items, cta, warnings)
    n = len(items)
    if chars > 140:
        warnings.append("全文仍 %d 字（要求 ≤140），建议人工精修" % chars)
    if n < 3:
        warnings.append("清单只有 %d 条（要求 3~4 条）" % n)
    return {"items": _render_body(items, cta), "cta": cta, "n": n, "chars": chars,
            "rows": items, "has_desc": has_desc, "copied": copied, "warnings": warnings}


def _generate(cfg, system, user, image_data_url=None):
    """按 _chain 依次尝试通道；同一通道失败重试一次。返回 (raw, used, elapsed)。

    GGUF 不重试：它最慢，失败多为硬错误（显存/解码），重试只会让等待翻倍。
    """
    _retry = {"localapi": 2, "api": 2, "gguf": 1}
    errs = []
    for src in _chain(cfg):
        if src not in _retry:
            continue
        for attempt in range(1, _retry[src] + 1):
            t0 = time.time()
            try:
                if src == "gguf":
                    raw = _gguf_generate(cfg, system, user, image_data_url)
                    used = "本地GGUF"
                elif src == "localapi":
                    _ensure_local_server(cfg)
                    base = (cfg.get("rwLocalBase") or "").strip()
                    model = (cfg.get("rwLocalModel") or "").strip()
                    if not base or not model:
                        raise RuntimeError("未配置本地服务地址/模型")
                    raw = _openai_chat(base, "", model, system, user, cfg, image_data_url,
                                       timeout=int(cfg.get("rwTimeout") or 300))
                    used = "本地服务"
                else:
                    base = (cfg.get("rwApiBase") or "").strip()
                    model = (cfg.get("rwApiModel") or "").strip()
                    key = cfg.get("rwApiKey") or ""
                    if not base or not model:
                        raise RuntimeError("未配置 API 地址/模型")
                    raw = _openai_chat(base, key, model, system, user, cfg, image_data_url,
                                       timeout=int(cfg.get("rwTimeout") or 300), vendor_tweaks=True)
                    used = "在线API"
                if not raw or not str(raw).strip():
                    raise RuntimeError("返回空")
                return raw, used, round(time.time() - t0, 1)
            except Exception as e:
                tag = src if attempt == 1 else ("%s 重试" % src)
                errs.append("%s: %s" % (tag, e))
                log.warning("内容二创 %s 失败：%s", tag, e)
    raise ValueError("全部推理通道失败 → " + " | ".join(errs))


_VOICE_BAD_START = ("其实", "今天", "很多", "大家", "这条视频", "别划走", "一定要看", "建议收藏")
_VOICE_FORBIDDEN_RE = re.compile(r"[？?！!：:；;、…\"“”'‘’「」『』【】\[\]（）()《》〈〉]")
_VOICE_EN_RE = re.compile(r"[A-Za-z]")
_VOICE_PROMISE_RE = re.compile(r"包过|稳进|百分百|100%|保证|必上岸|包就业|包拿证")
_VOICE_RETRY = (
    "\n\n【重写】上一版不合格（%s）。必须：18~22 个汉字（数字按 1 字计）、"
    "最多两个逗号、一句话；不要问号/感叹号/冒号/分号/顿号/引号/括号/英文/emoji；"
    "开头不要用「其实/今天/很多/大家/这条视频/别划走」；避开主标题和副标题原词；"
    "只输出这一句。"
)


def _voice_chars(s):
    """配音文案字数：汉字 + 数字，标点不计。"""
    return len(re.findall(r"[\u4e00-\u9fff0-9]", s or ""))


def _clean_voice(raw):
    s = _clean_single(raw)
    s = s.replace(",", "，")
    s = re.sub(r"，{2,}", "，", s)
    s = re.sub(r"[ \t\u3000]+", "", s)
    s = s.strip("，。；;、 ")
    return s


def _voice_reason(s):
    if not s:
        return "为空"
    n = _voice_chars(s)
    if not (18 <= n <= 22):
        return "字数 %d" % n
    if s.count("，") > 2:
        return "逗号 %d 个" % s.count("，")
    if _VOICE_FORBIDDEN_RE.search(s):
        return "含不允许的标点"
    if _VOICE_EN_RE.search(s):
        return "含英文"
    if _VOICE_PROMISE_RE.search(s):
        return "含承诺词"
    if any(s.startswith(x) for x in _VOICE_BAD_START):
        return "开头是禁用语"
    if "。" in s:
        return "写成了多句"
    return "不合格"


def _voice_ok(s):
    return bool(s) and _voice_reason(s) == "不合格"


def _voice_generate(cfg, prompt, user, image=None):
    """生成 18~22 字开场配音；不合格最多重写 3 次，仍不合格则保留最接近的一版。"""
    best, best_score, warns = "", 10 ** 9, []
    for _attempt in range(3):
        try:
            raw, used, _ = _generate(cfg, prompt, user, image)
        except Exception as e:
            warns.append("开场配音文案生成失败：%s" % e)
            continue
        s = _clean_voice(raw)
        if s:
            score = abs(_voice_chars(s) - 20) * 2
            if not _voice_ok(s):
                score += 10
            score += max(0, s.count("，") - 2) * 3
            if score < best_score:
                best, best_score = s, score
        if _voice_ok(s):
            return s, used, warns
        prompt = prompt + _VOICE_RETRY % _voice_reason(s)
    if best:
        warns.append("开场配音 %d 字（要求18~22），已保留最接近的一版，请人工确认" % _voice_chars(best))
    return best, "", warns


def make_voice_script(cfg, title1="", title2="", items="", source_text=""):
    """按《配音文案prompt.txt》生成开场配音（供 /voice 端点与二创共用）。"""
    user = ("【原文】\n%s\n\n【主标题】%s\n【副标题】%s\n【正文】\n%s"
            % ((source_text or "").strip(), (title1 or "").strip(),
               (title2 or "").strip(), (items or "").strip()))
    text, used, warns = _voice_generate(cfg, load_voice_prompt(), user)
    return {"voiceText": text, "warnings": warns, "source": used, "chars": _voice_chars(text)}


_CTA_ASK = (
    "你是短视频信息流文案编辑。请根据下面的原文，写 1 句 8~14 字的结尾引导语："
    "只催观众做一个动作（转告 / 收藏 / 存下 / 转给 / 告诉 / 记下），口语化、"
    "不要提问、不要问号、不要感叹号收尾、不要 emoji、不要「点赞关注」、不要解释、不要换行。"
    "只输出这一句。"
)
_PILL_HINT = "\n\n【再试一次】只输出一个不超过 8 个字的短语：不要标点、不要感叹号、不要引号、不要两句、不要解释。"


def _fix_single(cfg, system, text, lo, hi, user, image):
    """标题/胶囊长度兜底：原样合格就返回；不合格请模型重写（最多 2 次）；仍不合格才硬裁。

    为什么要 2 次：实测 7B 对「必须 N~M 个汉字」经常第一次仍然多字/少字
    （例：标题2 给 13 字、标题1 给 3 字），而**让它自己改短/改长**远比程序硬裁自然
    （硬裁会把词切成两半）。都失败时才退回 `_fit_len`（它会先整词去尾）。
    """
    w = []
    t = _fit_len(text, lo, hi)
    if t and lo <= _han(t) <= hi and _han(text) <= hi:
        return t, w                      # 原样就合格，不额外调用

    def near(x):                         # 离 [lo, hi] 的距离：0 表示已合规
        n = _han(x)
        return 0 if lo <= n <= hi else (lo - n if n < lo else n - hi)

    cands = [text] if text else []
    for i in range(2):
        try:
            # 模型爱「写满上限」，第一版超长时第二版把上限压 2 个字再要 ——
            # 让它自己给出**完整**的短标题，好过我们从词中间硬裁（实测「…百人工程”招聘」→「…招」）
            hi_ask = hi
            if i and cands and _han(cands[-1]) > hi:
                hi_ask = max(lo, hi - 2)
            ask = (system + "\n\n【只输出一条】必须 %d~%d 个汉字，不要引号、不要标点、不要解释、不要换行。"
                   % (lo, hi_ask))
            if i:
                ask += "\n上一次的字数不对：请先数清汉字个数，再输出。"
            raw, _, _ = _generate(cfg, ask, user, image)
            c = _clean_single(raw)
            if not c:
                continue
            cands.append(c)
            f = _fit_len(c, lo, hi)
            if f and lo <= _han(f) <= hi and _han(c) <= hi:
                if i:
                    w.append("长度为 %d 汉字（要求 %d~%d），重写后已修正" % (_han(c), lo, hi))
                elif _han(c) > hi:
                    w.append("模型超长，已重写")
                return f, w
        except Exception as e:
            log.warning("标题重写失败：%s", e)
    best = min(cands, key=near)
    t = _fit_len(best, lo, hi)
    w.append("长度 %d 汉字（要求 %d~%d），已裁剪" % (_han(t), lo, hi))
    return t, w


# 强格式指令：本地小模型偶尔整段无视 {编号 名称} 格式（实测 3 条素材里 2 条），
# 此时不再指望原 prompt，改用「描述式」指令重写正文。
# 实验对比过三种写法（见 temp\prompt_exp3.txt）：
#   V1 字面模板（给出 {01 名称}/说明 两行）  → 模型**照抄模板**，输出「{01 名称} / 说明」，废
#   V2 描述式 + 明令禁占位词                → 4 条全部合规，采用
#   V3 分行定义式                          → 模型把编号单独放进花括号、名称丢到下一行，废
_FORMAT_FIX = (
    "你是短视频信息流文案编辑。请把用户给出的素材，提炼成 3 到 4 条清单，每条占两行：\n"
    "第一行：一个半角花括号包起来的标签，里面是「两位数字编号 + 一个空格 + 一个短名称」，"
    "例如左花括号、01、空格、某某集团、右花括号。\n"
    "第二行：这一条的说明，15 字以内，写它最关键的信息"
    "（薪资/年龄/学历/专业/编制等数字必须原样保留，不得编造）。\n"
    "清单之后，再单独写一行结尾引导语：8~14 个字，催观众做一个动作"
    "（转告/收藏/存下/转给/告诉），不要提问、不要问号、不要「点赞关注」。\n"
    "注意：名称必须从素材里提炼，绝对不许出现「名称」「说明」「某某集团」这类占位词或示例词；"
    "宁可写不出也不要照抄示例。全文合计不超过 130 字。\n"
    "只输出清单和引导语，不要任何解释、标题或其它文字。"
)


def _strip_example(section):
    """去掉 内容prompt 里的 few-shot 示例条目（含其说明行）。

    仅用于「补救重试」，绝不修改用户的 prompt 文件。实测发现：当原文与示例
    完全不同（如「国考时间安排」）时，留着示例反而会让模型照抄示例或干脆弃用格式；
    去掉示例后模型更容易老老实实照原文写（实测救回「1. 公告发布：10月14日」这种清单）。
    """
    src = (section or "").splitlines()
    drop = set()
    for i, ln in enumerate(src):
        s = ln.strip()
        m = _BULLET_ANY.match(s) if s else None
        if not m:
            continue
        drop.add(i)
        if m.end() >= len(s):                   # 示例说明在下一行
            for j in range(i + 1, len(src)):
                t = src[j].strip()
                if t and not _BULLET_ANY.match(t):
                    drop.add(j)
                    break
    return "\n".join(ln for i, ln in enumerate(src) if i not in drop)


def _items_from_source(user, limit=4):
    """零编造兜底：原文自带「短名称：说明」式要点（≥3 条）时，直接整理成清单。

    实测本地 7B 对「入党流程」这类原文偶尔彻底走样（吐时间码、把说明全丢、或 0 条）。
    与其导出废片，不如把原文自带的要点**原样**搬成清单 —— 文字全部来自原文，不编造，
    只做长度/条数/格式的程序化收敛（交给 normalize_body）。
    """
    rows = []
    for ln in (user or "").splitlines():
        s = re.sub(r"^[\s\-–—•·*●\d.、)）]+", "", (ln or "").strip()).strip()
        m = re.match(r"^([^\s：:，,。.；;、]{2,12})\s*[：:]\s*(.{4,})$", s)
        if m and _han(m.group(1)) >= 2:
            rows.append([m.group(1).strip(), m.group(2).strip()])
    return rows[:limit] if len(rows) >= 3 else []


_NUM_UNIT_RE = re.compile(r"\d+\s*[天日月年万亿千个条类分人步周%]")


def _pill_nums_bad(pill, ground):
    """胶囊里「数字+单位」必须能在原文/正文里找到，找不到就是编造。

    实测模型给过「国考倒计时3天」，而原文只有「公告发布：10月14日 / 面试时间：1月-3月」——
    这种数字最有钩子，但属于编造，必须挡掉。
    """
    g = re.sub(r"\s", "", ground or "")
    return [t for t in _NUM_UNIT_RE.findall(pill or "") if re.sub(r"\s", "", t) not in g]


def _pill_from_body(rows, titles, maxhan=8):
    """胶囊兜底：从正文里挑一个「带数字/时间」的短钩子（文字全部取自正文，零编造）。

    实测模型给的胶囊经常直接复述标题（overlap 1.00），重写三次还是撞词。
    与其把撞词的胶囊交出去，不如从正文里挑一个具体数字当钩子（如「10天报名」）。
    """
    hits = []
    for nm, ds in rows or []:
        head = re.sub(r"[\[\]{}｛｝【】]", " ", nm or "")
        for seg in re.split(r"[，,；;。！!？?、／/\s]+", ("%s %s" % (head, ds or ""))):
            seg = seg.strip(_TRIM_CHARS).strip()
            if not seg or not re.search(r"\d", seg):
                continue                      # 要带数字才够「钩子」
            if _han(seg) < 2:
                continue                      # 「01」这种纯编号不是钩子
            if _han(seg) > maxhan or len(re.sub(r"\s", "", seg)) > maxhan + 2:
                continue
            if _overlap(seg, titles) >= 0.6:
                continue                      # 别又和标题撞词
            hits.append(seg)
    if not hits:
        return ""
    for h in hits:                            # 优先「时间/紧迫感」型钩子（最像开头 3 秒的画面字）
        if re.search(r"天|年|月|日|截止|倒计时|期限|名额", h):
            return h
    return hits[0]                            # 否则取正文里最靠前的那个


def _body_ok(b):
    """正文是否合格：≥3 条、每条都带说明（规则 5~7），且不是照抄 prompt 示例（零编造）。"""
    return b["n"] >= 3 and b["has_desc"] and not b.get("copied")


def _paired(b):
    """这版正文是不是靠「相邻两行重新配对」硬凑出来的（配对按行位置硬配，容易错位）。"""
    return any("重新配对" in (w or "") for w in (b.get("warnings") or []))


def _all_desc(b):
    """每条都带说明（规范：每条占两行，第一行 {编号 名称}、第二行说明）。"""
    rows = b.get("rows") or []
    return bool(rows) and all(d for _n, d in rows)


def _suspect_body(b):
    """结构可疑：看着合规，但名称/说明明显错位或复读。

    实测模型会把「公告发布：10月14日」整行当**名称**，又把**下一行**当说明，于是整列错位：
      {01 公告发布：10月14日} / 2027国考时间安排参考2026国考
      {02 报名开始：10月15日} / 报名确认时间：11月1日-6日   ← 说明其实是下一条
    另外实测过「4 条共用同一句说明」的复读版（{01 学生}/{02 薪资}/{03 职位}/{04 报名} 说明一模一样），
    名称按规范应是「两位编号+分类名」（≤8 字、无冒号），说明也不是「名称：值」，
    所以这几条任一成立就判可疑。
    """
    rows = b.get("rows") or []
    descs = [re.sub(r"\s", "", str(d or "")) for _n, d in rows if d]
    if len(descs) >= 3 and len(set(descs)) * 2 < len(rows):
        return True                                    # 说明大量重复 = 模型复读
    for nm, ds in rows:
        bare = re.sub(r"^\{?\d{1,2}[ \u3000]?", "", str(nm or "")).rstrip("}").strip()
        if re.search(r"[：:]", bare):
            return True
        if ds and re.match(r"^[^：:\n]{2,12}\s*[：:]\s*\S", str(ds)):
            return True
    return False


def _cand_key(c):
    """候选正文排序键：先「完全合格」，再看没照抄、有说明、没走配对、不可疑、每条都有说明、条数。"""
    b = c[0]
    return (_body_ok(b), not b["copied"], b["has_desc"],
            not _paired(b), not _suspect_body(b), _all_desc(b), b["n"])


def _best_body(cfg, secs, user, body_user, image):
    """正文生成：优先用用户 prompt；不合格（<3 条 / 全无说明 / 照抄示例）时依次补救。

    实测本地 7B 对「非实体类」素材（如入党流程、求职 tips）经常整段无视
    「{编号 名称}」格式，甚至把强格式指令的模板原样抄回来，所以：
      ① 用户 prompt → ② 同一 prompt 重掷一次（模型有随机性）
      → ③ 用户 prompt 去掉 few-shot 示例（示例会诱导照抄）→ ④⑤ 描述式强格式指令（不含示例）
      → 相邻两行重新配对的兜底解读
    取其中最好的一版：优先「没照抄示例」，再看有没有说明，最后看条数。
    """
    ex = _example_items(secs["内容"])
    attempts = (("内容prompt", secs["内容"], body_user),
                ("内容prompt 重掷", secs["内容"], body_user),
                ("内容prompt 去示例", _strip_example(secs["内容"]), body_user),
                ("强格式指令", _FORMAT_FIX, user),
                ("强格式指令 重掷", _FORMAT_FIX, user))
    cands = []                                  # [(body, raw, tag)]

    def parse(raw):
        b = normalize_body(raw, example_items=ex)
        if not _body_ok(b) and not b["has_desc"]:
            b2 = normalize_body(raw, pair=True, example_items=ex)
            if (_body_ok(b2), not b2["copied"], b2["has_desc"], b2["n"]) > \
               (_body_ok(b), not b["copied"], b["has_desc"], b["n"]):
                return b2
        return b

    for tag, sys_p, usr in attempts:
        try:
            raw, _, _ = _generate(cfg, sys_p, usr, image)
        except Exception as e:
            log.warning("正文生成失败(%s)：%s", tag, e)
            continue
        cands.append((parse(raw), raw, tag))
        if _body_ok(cands[-1][0]):
            break
    if not cands:
        b = normalize_body("")
        b["warnings"].append("正文生成失败")
        return b, "", []
    # 所有候选都不合格（<3 条 / 缺说明），或最好的那版是「硬凑/错位」的（配对兜底、名称里带「：说明」）
    # → 再出最后一招：把原文自带要点原样整理成清单（零编造）。
    # 为什么要管「可疑版」：配对只按行位置硬配、模型也可能把整列说明错位，
    # 实测「国考时间表」会被配成 {01 预计下周国考出公告}/2027国考时间安排、{02 报名开始}/打印准考证…，
    # 而这类原文本身就是「名称：说明」形式，抓原文反而准。
    _best_now = max(cands, key=_cand_key)[0]
    if (not _body_ok(_best_now) or _paired(_best_now) or _suspect_body(_best_now)
            or not _all_desc(_best_now)):
        rows = _items_from_source(user)
        if rows:
            fake = _render_body([["{%02d %s}" % (i, nm), ds]
                                 for i, (nm, ds) in enumerate(rows, 1)], "")
            b = normalize_body(fake, example_items=ex)
            if b["n"] >= 3:
                b["warnings"].append("模型输出格式不合格，正文已改为「原文要点自动整理」"
                                     "（文字均取自原文，未编造）")
                cands.append((b, fake, "原文要点整理"))
    # 择优顺序：先要「完全合格」（≥3 条 + 有说明 + 没照抄），再看没照抄、说明、条数；
    # 同档时优先「不是配对凑出来的」那一版 —— 配对容易配错位。
    body, body_raw, tag = max(cands, key=_cand_key)
    if _suspect_body(body):                 # 连「原文要点整理」都救不回来时才可能走到这
        body["warnings"].append("正文结构可疑（名称里带「：说明」或说明是下一条），请人工核对")
    if tag != "内容prompt":
        body["warnings"].append("首次输出未按清单格式，已由「%s」补救" % tag)
    # 顺带把各候选版本的引导语带回去：正文与引导语来自同一次输出，
    # 但实测会出现「这版正文好、引导语短；那版引导语合格」——留着让上层择优。
    return body, body_raw, [c[0]["cta"] for c in cands]


def _rewrite_sections(cfg, secs, user, image_data_url=None):
    """四段式链路：内容 → 标题1 → 标题2（带标题1）→ 胶囊（带两个标题）。"""
    t0 = time.time()
    warnings = []

    # ① 内容（正文清单 + 结尾引导语）
    body_user = user
    if _recent_cta:
        body_user = user + "\n\n【结尾引导语请换一句，不要与这些重复：%s】" % "、".join(_recent_cta)
    body, body_raw, cta_alts = _best_body(cfg, secs, user, body_user, image_data_url)
    warnings.extend(body["warnings"])
    if body.get("copied"):
        warnings.append("【重要】正文疑似照抄 prompt 里的示例、与原文无关，请务必人工核对后再导出")

    # 引导语缺失/不合格（规则 3/10/12）→ 先从其它候选版本里挑一句合格的，再请模型补写
    if not _looks_like_cta(body["cta"]):
        alt = next((c for c in cta_alts if c and c != body["cta"]
                    and _looks_like_cta(c) and c not in _recent_cta), "")
        if alt:
            body["cta"] = alt
            body["items"] = _render_body(body["rows"], alt)
            warnings.append("结尾引导语取自另一候选版本（同批那版不合格），已换用合格的")
    if not _looks_like_cta(body["cta"]):
        best_cta = body["cta"]
        for attempt in range(2):
            try:
                ask = _CTA_ASK
                if attempt:
                    ask += "\n务必写够 10 个字以上，别太短。"
                if _recent_cta:
                    ask += "\n不要与这些重复：" + "、".join(_recent_cta)
                cta_raw, _, _ = _generate(cfg, ask, user, image_data_url)
                cta = _fit_len(_clean_single(cta_raw), 1, 14, hard_max=14)
                if _looks_like_cta(cta):
                    best_cta = cta
                    break
                if _han(cta) > _han(best_cta or ""):
                    best_cta = cta
            except Exception as e:
                log.warning("补写引导语失败：%s", e)
        if best_cta and best_cta != body["cta"]:
            body["items"] = _render_body(body["rows"], best_cta)
            body["cta"] = best_cta
            if _looks_like_cta(best_cta):
                warnings.append("模型漏写 / 写短了结尾引导语，已自动补写")
            elif not (8 <= _han(best_cta) <= 14):
                warnings.append("结尾引导语 %d 字（要求 8~14），已尽力补写" % _han(best_cta))
            else:
                warnings.append("结尾引导语没催动作（%r），已尽力补写" % best_cta)
    if not _looks_like_cta(body["cta"]):
        warnings.append("结尾引导语缺失或不合格，建议人工补一句")
    # 引导语可能刚被换掉（择优 / 补写），长度会变 → 再收敛一次全文（≤140 字）
    body["rows"], body["cta"], body["chars"] = _fit_total(body["rows"], body["cta"], warnings)
    body["items"] = _render_body(body["rows"], body["cta"])
    body["n"] = len(body["rows"])
    if body["cta"]:
        _recent_cta.append(body["cta"])

    # ② 标题1（4~10 汉字）
    t1_raw, used, _ = _generate(cfg, secs["标题1"], user, image_data_url)
    title1, w1 = _fix_single(cfg, secs["标题1"], _clean_single(t1_raw), 4, 10, user, image_data_url)
    warnings.extend(w1)

    # ③ 标题2（6~10 汉字，且必须与标题1 互补、不重复核心词组）
    u2 = ("原文：\n%s\n\n【上一条标题，副标题必须与它明显不同、不重复它的核心词组，"
          "并补充它没提到的条件 / 对象 / 细节】：%s" % (user, title1))
    t2_raw, _, _ = _generate(cfg, secs["标题2"], u2, image_data_url)
    title2, w2 = _fix_single(cfg, secs["标题2"], _clean_single(t2_raw), 6, 10, u2, image_data_url)
    warnings.extend(w2)
    if title2 and title1 and _overlap(title2, title1) >= 0.75:
        try:
            raw, _, _ = _generate(
                cfg,
                secs["标题2"] + "\n\n【只输出一个副标题】必须与「%s」明显不同：不得复用它的核心词组，"
                                "要补充它没提到的信息，6~10 个汉字。" % title1,
                u2, image_data_url)
            t = _fit_len(_clean_single(raw), 6, 10)
            if t and 6 <= _han(t) <= 10 and _overlap(t, title1) < 0.75:
                title2 = t
                warnings.append("标题2 与标题1 过于相似，已重写")
        except Exception as e:
            log.warning("标题2 去重失败：%s", e)

    # ④ 胶囊（≤8 字，且不能与两个标题相似）
    #    把正文也喂进去：胶囊是「开头 3~5 秒的钩子」，正文里的数字/对象就是最好的钩子素材，
    #    只给原文+标题时模型很容易直接把标题复述一遍（实测 overlap 到 1.00）。
    body_txt = "\n".join((body["items"] or "").splitlines()[:6])
    u3 = ("原文：\n%s\n\n正文清单（可从中提炼钩子）：\n%s\n\n标题1：%s\n标题2：%s\n"
          "（胶囊不能与上面两个标题相似）" % (user, body_txt, title1, title2))
    p_raw, _, _ = _generate(cfg, secs["胶囊"], u3, image_data_url)
    pill, w3 = _fix_single(cfg, secs["胶囊"], _clean_single(p_raw), 1, 8, u3, image_data_url)
    warnings.extend(w3)
    if _han(pill) > 8:
        try:
            raw, _, _ = _generate(cfg, secs["胶囊"] + _PILL_HINT, u3, image_data_url)
            t = _fit_len(_clean_single(raw), 1, 8, hard_max=8)
            if t and _han(t) <= 8:
                pill = t
        except Exception as e:
            log.warning("胶囊重写失败：%s", e)
    if _han(pill) > 8:
        warnings.append("胶囊 %d 字（要求 ≤8），已裁剪" % _han(pill))
    # 胶囊太笼统（≤3 字，实测「别错过」）→ 你的胶囊prompt 要求「有适度紧迫感或悬念、
    # 可带数字」，让模型拿正文里的具体信息重写一条（没有钩子的胶囊等于白写）。
    if pill and _han(pill) <= 3:
        _vague = pill
        try:
            raw, _, _ = _generate(
                cfg,
                secs["胶囊"] + "\n\n【重写】上一版「%s」太笼统，没有钩子。请用原文里的具体信息"
                                "（数字 / 对象 / 时间）重写一条，4~8 个字，不要评论式的话。" % _vague,
                u3, image_data_url)
            t = _fit_len(_clean_single(raw), 1, 8, hard_max=8)
            if t and 4 <= _han(t) <= 8 and _overlap(t, title1 + title2) < 0.85:
                pill = t
                warnings.append("胶囊太笼统（「%s」），已用原文里的具体信息重写" % _vague)
        except Exception as e:
            log.warning("胶囊具体化失败：%s", e)
    # 规则 3：胶囊不能与两个标题相似。每次都**换一个角度**重写（只说「不能重复」模型不会照做）。
    _pill_hints = (
        "\n\n【重写】上一版「%s」与标题重复，不能再用「%s」这些词；换一个说法，不超过 8 个字。",
        "\n\n【换角度重写】上一版在复述标题（「%s」）。请改用原文里的一个**数字或时间**当钩子"
        "（如「3类岗位」「10天报名」「年薪30万」），不要出现标题里的词，不超过 8 个字。",
        "\n\n【再换角度】上一版「%s」还是和标题撞词。请用「对象＋紧迫感」写法"
        "（如「别错过这10天」「应届生专享」），与两个标题一个字都不重叠，不超过 8 个字。",
    )
    if pill and _overlap(pill, title1 + title2) >= 0.85:
        for attempt in range(3):
            try:
                if attempt == 0:
                    hint = _pill_hints[0] % (pill, "".join(sorted(set(pill))))
                else:
                    hint = _pill_hints[attempt % len(_pill_hints)] % (pill,)
                raw, _, _ = _generate(cfg, secs["胶囊"] + hint, u3, image_data_url)
                t = _fit_len(_clean_single(raw), 1, 8, hard_max=8)
                if t and _han(t) <= 8 and _overlap(t, title1 + title2) < 0.85:
                    pill = t
                    warnings.append("胶囊与标题过于相似，已换角度重写")
                    break
                if t and _han(t) <= 8 and _overlap(t, title1 + title2) < _overlap(pill, title1 + title2):
                    pill = t                     # 至少比上一版更不像标题
            except Exception as e:
                log.warning("胶囊去重失败：%s", e)
    if pill and _overlap(pill, title1 + title2) >= 0.85:
        # 还撞词 → 从正文里挑一个带数字的短钩子（文字全部取自正文，零编造）
        alt = _pill_from_body(body["rows"], title1 + title2)
        if alt:
            pill = alt
            warnings.append("胶囊与标题过于相似，已改用正文里的数字钩子（文字取自正文，未编造）")
    if pill and _overlap(pill, title1 + title2) >= 0.85:
        warnings.append("胶囊与标题用词重复较多，建议人工确认")

    # ⑤ 胶囊数字必须是原文/正文里真有的（防编造：实测「国考倒计时3天」）
    ground = user + "\n" + (body["items"] or "")
    bad = _pill_nums_bad(pill, ground)
    if pill and bad:
        try:
            raw, _, _ = _generate(
                cfg,
                secs["胶囊"] + "\n\n【重写】上一版「%s」用了原文里没有的数字「%s」（属于编造）。"
                                "请改用原文里真实出现的数字，或干脆不带数字，不超过 8 个字。"
                                % (pill, "、".join(bad)),
                u3, image_data_url)
            t = _fit_len(_clean_single(raw), 1, 8, hard_max=8)
            if t and _han(t) <= 8 and not _pill_nums_bad(t, ground):
                pill = t
                warnings.append("胶囊里的数字原文没有（有编造风险），已换用原文里的数字重写")
        except Exception as e:
            log.warning("胶囊数字校正失败：%s", e)
    bad = _pill_nums_bad(pill, ground)
    if pill and bad:
        alt = _pill_from_body(body["rows"], title1 + title2)
        if alt and not _pill_nums_bad(alt, ground):
            pill = alt
            warnings.append("胶囊数字原文没有，已改用正文里的数字（文字取自正文，未编造）")
    bad = _pill_nums_bad(pill, ground)
    if pill and bad:
        warnings.append("胶囊里的数字「%s」在原文里找不到，请核对是否编造" % "、".join(bad))

    # ⑥ 开场配音文案（18~22 字，约 5 秒；失败留空，导出时用正文兜底）
    voice_text = ""
    try:
        v_user = ("【原文】\n%s\n\n【主标题】%s\n【副标题】%s\n【正文】\n%s"
                  % (user, title1, title2, body["items"]))
        voice_text, _v_used, v_warns = _voice_generate(
            cfg, load_voice_prompt(), v_user, image_data_url)
        warnings.extend(v_warns)
    except Exception as e:
        log.warning("开场配音文案生成失败：%s", e)
        warnings.append("开场配音文案生成失败（导出时将用正文兜底）")

    # 去重（补救链可能对同一问题报两次，比如换引导语后又收敛了一次长度）
    uniq = []
    for w in warnings:
        if w not in uniq:
            uniq.append(w)
    warnings = uniq

    el = round(time.time() - t0, 1)
    if not title1:
        warnings.append("标题1 为空，请人工确认")
    res = {
        "titleStyle": "",          # 三段式：titleStyle 留空 → 工作台按「有标题2/有胶囊」自动全显示
        "bodyStyle": "",
        "title1": title1,
        "title2": title2,
        "pill": pill,
        "items": body["items"],
        "voiceText": voice_text,
        "warnings": warnings,
        "source": used,
        "elapsed": el,
        "ok": True,
        "raw": (body_raw or "")[:2000],
        "meta": {"bodyChars": body["chars"], "n": body["n"], "cta": body["cta"]},
    }
    log.info("内容二创成功(四段式)：来源=%s 用时=%.1fs 标题1=%s 标题2=%s 胶囊=%s 正文%d字/%d条 配音=%s",
             used, el, title1, title2, pill, body["chars"], body["n"], voice_text or "—")
    return res


# ----------------------------------------------------------------------------
# 统一入口
# ----------------------------------------------------------------------------
def _chain(cfg):
    order = (cfg.get("rwOrder") or "auto").strip().lower()
    if order == "api":
        return ["api", "localapi", "gguf"]
    if order == "localapi":
        return ["localapi", "gguf", "api"]
    if order == "gguf":
        return ["gguf", "localapi", "api"]
    return ["gguf", "localapi", "api"]


def rewrite(cfg, text="", image_data_url=None):
    """原文/图片 → 四要素。返回 dict（含 source / warnings / raw 截断）。失败抛异常。

    全程持有 _gen_lock：串行化本地 GGUF / 本地服务 / 在线 API 的调用，
    避免收件箱监听线程与 HTTP 请求并发使用同一个 Llama 实例。
    """
    with _gen_lock:
        return _rewrite_locked(cfg, text=text, image_data_url=image_data_url)


def _rewrite_locked(cfg, text="", image_data_url=None):
    system = load_prompt()
    user = (text or "").strip()
    if not user and not image_data_url:
        raise ValueError("没有可处理的原文")
    if user and image_data_url:
        user = "以下是原文（文字与图片，请合并理解）：\n" + user
    elif not user:
        user = "请识别图片中的原文，并按格式产出文案。"

    # 四段式规范（内容 / 标题1 / 标题2 / 胶囊）：每段单独调用、逐级把上一段结果
    # 喂给下一段，并做硬指标兜底。
    secs = split_prompt_sections(system)
    if secs:
        return _rewrite_sections(cfg, secs, user, image_data_url)

    # 只规定正文的单段提示词（如《内容生成prompt.txt》）：正文按用户文件生成，
    # 标题1 / 标题2 / 胶囊用内置指令补齐，保证工作台四要素齐全。
    if _is_body_only_prompt(system):
        secs = {"内容": system, "标题1": _TITLE1_ASK,
                "标题2": _TITLE2_ASK, "胶囊": _PILL_ASK}
        return _rewrite_sections(cfg, secs, user, image_data_url)

    # 兼容旧的「单次合并」prompt（自身就要求输出标题，且不含 4 段标记）
    raw, used, el = _generate(cfg, system, user, image_data_url)
    res = parse_output(raw)
    res["source"] = used
    res["elapsed"] = el
    res["raw"] = (raw or "")[:4000]
    res["ok"] = True
    log.info("内容二创成功(单次)：来源=%s 用时=%.1fs 标题=%s", used, el, res["title1"])
    return res


def status(cfg):
    """给前端看的通道可用性概览（不触发模型加载，保证秒回）。"""
    gg = dict(_gguf)
    return {
        "gguf": {"ok": bool(gg.get("ok")), "path": gg.get("path") or discover_gguf()[0],
                 "loaded": bool(gg.get("tried")), "vision": bool(gg.get("vision")),
                 "err": gg.get("err") or ""},
        "localapi": {"base": (cfg.get("rwLocalBase") or "").strip(),
                     "model": (cfg.get("rwLocalModel") or "").strip()},
        "api": {"base": (cfg.get("rwApiBase") or "").strip(),
                "model": (cfg.get("rwApiModel") or "").strip(),
                "key": bool(cfg.get("rwApiKey"))},
        "promptPath": prompt_path(),
        "promptChars": len(load_prompt()),
        "voicePromptPath": _VOICE_PROMPT_FILE,
        "voicePromptChars": len(load_voice_prompt()),
    }


def warm(cfg, timeout=None):
    """预热本地服务：让 Ollama / LM Studio 提前把模型载入显存。

    打开弹窗时调用，用户粘贴文字的这段时间里模型就装好了，
    点「生成」时通常已是热态（实测热态 ~1.5 秒）。失败只记日志，不抛异常。
    """
    base = (cfg.get("rwLocalBase") or "").strip()
    model = (cfg.get("rwLocalModel") or "").strip()
    if not base or not model:
        return {"ok": False, "skipped": True, "reason": "未配置本地服务地址/模型"}
    _ensure_local_server(cfg)
    t0 = time.time()
    with _gen_lock:
        try:
            _openai_chat(base, "", model, "你是助手。", "只回复一个字：好",
                         {"rwTemp": 0, "rwMaxTokens": 1},
                         timeout=int(timeout or cfg.get("rwTimeout") or 300))
        except Exception as e:
            log.warning("内容二创预热失败：%s", e)
            return {"ok": False, "elapsed": round(time.time() - t0, 1), "error": str(e)}
    return {"ok": True, "model": model, "base": base, "elapsed": round(time.time() - t0, 1)}
