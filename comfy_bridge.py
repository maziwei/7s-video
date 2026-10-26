#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""ComfyUI 本地桥：给 ComfyUI API 补 CORS 头，供 file:// 网页直接调用。
新增 /qwen/tts 端点：Qwen3-TTS 1.7B-VoiceDesign 本地直连人声（文字描述音色 + 语气指令）。
新增 /tts 端点：Edge-TTS（微软在线语音，免费、无需显卡/ComfyUI），生成 mp3 并缓存。
新增 /voice 端点：按《配音文案prompt.txt》单独生成一句 18~22 字的开场配音。
新增 /llm 端点：OpenAI 兼容大模型 API 转发（上游失败自动回退本地 Qwen3-0.6B）。

用法:
    python comfy_bridge.py [--port 8677] [--upstream http://127.0.0.1:8188]

端点:
    GET/POST/*   -> 转发到 ComfyUI（原样代理，含 CORS 头）
    POST /qwen/tts -> body {"text":"...","instruct":"...","voice":"female|male","speaker":"可选(忽略)"}
                     （Qwen3-TTS-12Hz-1.7B-VoiceDesign，文字描述音色）
                     返回生成的人声音频（wav bytes，Content-Type: audio/wav）
    POST /tts   -> body {"text":"...","voice":"zh-CN-XiaoxiaoNeural","rate":0,"volume":100}
                     返回 {"ok":true,"id":"sha1","duration":12.3,"cached":false}
    GET  /tts/audio?id=<sha1> -> 缓存的 mp3（试听/前端解码）
    POST /llm   -> body {"base":"OpenAI兼容接口","api_key":"","model":"","system":"","user":""}
                     返回 {"text":"...","source":"upstream|local"}

只监听本机回环地址，不对外暴露。
"""
import http.server
import json
import logging
import os
import sys
import threading
import time
import traceback
import urllib.error
import urllib.parse
import urllib.request

# ---------- 日志配置 ----------
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
    datefmt="%H:%M:%S",
    handlers=[
        logging.StreamHandler(sys.stdout),
        logging.FileHandler("bridge.log", encoding="utf-8"),
    ],
)
log = logging.getLogger("bridge")

# 内容二创引擎（原文/图片 → 标题1/标题2/胶囊/内容）；缺失时桥仍可正常运行
try:
    import content_rewrite as CRE
except Exception as _cre_err:  # pragma: no cover
    CRE = None
    log.warning("内容二创引擎加载失败（不影响其它功能）：%s", _cre_err)

UPSTREAM = "http://127.0.0.1:8188"
PORT = 8677

# ---------- 路径自动探测（ComfyUI 环境 → 本地独立部署） ----------
_SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
# 本地持久化配置（飞书凭证等），避免依赖浏览器 localStorage（file:// 下易被清理）
CONFIG_PATH = os.path.join(_SCRIPT_DIR, "config.json")
_LOCALAPPDATA = os.environ.get("LOCALAPPDATA") or os.path.expanduser(r"~\AppData\Local")
_INSTALLS_ROOT = os.path.join(_LOCALAPPDATA, "Comfy-Desktop", "ComfyUI-Installs")
_COMFY_INSTALLS = []
_env_comfy = (os.environ.get("COMFYUI_PATH") or "").strip()
if _env_comfy and os.path.isdir(_env_comfy):
    _COMFY_INSTALLS.append(_env_comfy)
if os.path.isdir(_INSTALLS_ROOT):
    # 常见布局：ComfyUI-Installs\<实例名>\ComfyUI；也兼容 <实例名> 本身就是 ComfyUI 目录
    try:
        _instances = sorted(os.listdir(_INSTALLS_ROOT))
    except OSError:
        _instances = []
    for _d in _instances:
        for _cand in (os.path.join(_INSTALLS_ROOT, _d, "ComfyUI"),
                      os.path.join(_INSTALLS_ROOT, _d)):
            if os.path.isfile(os.path.join(_cand, "main.py")) and _cand not in _COMFY_INSTALLS:
                _COMFY_INSTALLS.append(_cand)
    _flat = os.path.join(_INSTALLS_ROOT, "ComfyUI")
    if os.path.isfile(os.path.join(_flat, "main.py")) and _flat not in _COMFY_INSTALLS:
        _COMFY_INSTALLS.insert(0, _flat)
_COMFY_INSTALLS.append(os.path.join(_SCRIPT_DIR, "ComfyUI"))
_COMFY_ROOT = None
for _p in _COMFY_INSTALLS:
    if os.path.isdir(_p):
        _COMFY_ROOT = _p
        break

def _resolve_path(comfy_rel, local_rel):
    """优先 ComfyUI 目录，回退到脚本同目录下的本地路径。"""
    if _COMFY_ROOT:
        full = os.path.join(_COMFY_ROOT, comfy_rel)
        if os.path.isdir(full):
            return full
    local = os.path.join(_SCRIPT_DIR, local_rel)
    if os.path.isdir(local):
        return local
    # 都不存在时返回本地路径（download_models.bat 会创建它）
    return local

# ---------- Qwen3-TTS 直连（不经 ComfyUI） ----------
_QWEN_MODEL_PATH = _resolve_path(
    r"models\qwen-tts\Qwen3-TTS-12Hz-1.7B-VoiceDesign",
    r"models\qwen-tts\Qwen3-TTS-12Hz-1.7B-VoiceDesign")
# VoiceDesign：男/女声用自然语言描述音色（不再需要参考音频/预设 speaker）
_VOICE_DESIGN_BASE = {
    "female": "温暖知性、标准普通话的年轻女声，",
    "male": "专业播音员般的标准普通话男声，字正腔圆、沉稳大气，",
}
_qwen_model = None
_qwen_lock = threading.Lock()


def _qwen_free_upstream():
    """生成 TTS 前释放 ComfyUI 缓存模型，避免 8GB 显存 OOM 杀进程。"""
    try:
        import urllib.request as _ur
        req = _ur.Request(UPSTREAM + "/free",
                          data=json.dumps({"unload_models": True, "free_memory": True}).encode("utf-8"),
                          headers={"Content-Type": "application/json"}, method="POST")
        with _ur.urlopen(req, timeout=30):
            pass
    except Exception as e:
        log.warning("释放 ComfyUI 缓存模型失败: %s", e)


def _qwen_load():
    global _qwen_model
    if _qwen_model is not None:
        return _qwen_model
    import os
    import sys as _sys
    import numpy as np
    import torch
    torch.cuda.empty_cache()
    _NODE = _resolve_path(r"custom_nodes\ComfyUI-Qwen-TTS", r"custom_nodes\ComfyUI-Qwen-TTS")
    if _NODE not in _sys.path:
        _sys.path.insert(0, _NODE)
    from qwen_tts import Qwen3TTSModel
    _qwen_model = Qwen3TTSModel.from_pretrained(
        _QWEN_MODEL_PATH, device_map="cuda", dtype=torch.bfloat16,
        attn_implementation="sdpa")
    return _qwen_model


def run_qwen_tts(text, instruct, voice="female", speaker=None):
    """Qwen3-TTS 1.7B-VoiceDesign 生成人声，返回 (wav_bytes, sr, err)。生成后卸载模型释放显存。
    voice: female|male，音色由自然语言描述（instruct + 基础描述），speaker 参数忽略。"""
    global _qwen_model
    with _qwen_lock:
        try:
            _qwen_free_upstream()  # 先释放 ComfyUI 模型，防 OOM
            import numpy as np
            import soundfile as sf
            model = _qwen_load()
            full_instruct = (_VOICE_DESIGN_BASE.get(voice, "") + (instruct or "")).strip() or None
            wavs, out_sr = model.generate_voice_design(
                text=text, language="chinese", instruct=full_instruct,
                max_new_tokens=2048, do_sample=True,
                top_p=1.0, top_k=50, temperature=0.9, repetition_penalty=1.05)
            out = wavs[0] if isinstance(wavs, (list, tuple)) else wavs
            arr = np.asarray(out, dtype=np.float32)
            import io as _io
            buf = _io.BytesIO()
            sf.write(buf, arr, out_sr, format="WAV")
            return buf.getvalue(), out_sr, None
        except Exception as e:
            log.error("Qwen-TTS 生成失败: %s", e)
            return None, 0, {"error": "qwen tts: %s" % e}
        finally:
            # 生成后卸载，避免与 ComfyUI 生图/生视频抢显存
            try:
                if _qwen_model is not None:
                    _qwen_model = None
                    import torch
                    torch.cuda.empty_cache()
            except Exception as e:
                log.warning("Qwen-TTS 模型卸载/显存清理失败: %s", e)


# ---------- Edge-TTS（微软在线语音，免费、无需显卡/ComfyUI） ----------
# 前端把每行正文发到 /tts → 这里生成 mp3 缓存到 tts_cache\<sha1>.mp3，
# 再把 id/时长返回前端；导出时 /video/compose 用 id 找到 mp3 与 BGM 混音。
TTS_CACHE_DIR = os.path.join(_SCRIPT_DIR, "tts_cache")
_TTS_VOICE_RE = None  # 延迟到首次调用时编译（避免顶部再 import re）
_TTS_MAX_CHARS = 1500
_TTS_CLEANED = {"done": False}
# 同一段文案可能被「预览」和「批量导出」同时请求；共用 .part 会触发
# Windows 的 WinError 32（文件被占用）。生成阶段串行化，并每次用独立临时文件。
_TTS_GEN_LOCK = threading.Lock()


def _tts_rel_rate(v):
    try:
        v = int(round(float(v)))
    except (TypeError, ValueError):
        v = 0
    return "%+d%%" % max(-50, min(100, v))


def _tts_rel_volume(v):
    """前端音量 0~100 → Edge 的相对音量 -100%~0%；>100 视为放大。"""
    try:
        v = int(round(float(v)))
    except (TypeError, ValueError):
        v = 100
    return "%+d%%" % max(-100, min(100, v - 100))


def _tts_duration(path):
    """用 ffprobe 读 mp3 时长（秒）；失败返回 0。"""
    ff = _find_ffmpeg()
    if not ff:
        return 0.0
    probe = os.path.join(os.path.dirname(ff), "ffprobe.exe")
    if not os.path.isfile(probe):
        import shutil
        probe = shutil.which("ffprobe") or shutil.which("ffprobe.exe") or ""
    if not probe:
        return 0.0
    try:
        import subprocess
        p = subprocess.run(
            [probe, "-v", "error", "-show_entries", "format=duration", "-of", "csv=p=0", path],
            capture_output=True, text=True, timeout=30)
        return round(float((p.stdout or "").strip() or 0), 3)
    except Exception:
        return 0.0


def _tts_cleanup_once(max_age_days=7):
    """顺手清理 7 天前的配音缓存（只在首次调用时跑一次）。"""
    if _TTS_CLEANED["done"]:
        return
    _TTS_CLEANED["done"] = True
    try:
        cutoff = time.time() - max_age_days * 86400
        for fn in os.listdir(TTS_CACHE_DIR):
            p = os.path.join(TTS_CACHE_DIR, fn)
            if os.path.isfile(p) and os.path.getmtime(p) < cutoff:
                os.remove(p)
    except Exception:
        pass


def _tts_cached_result(key, path, meta, chars):
    dur = _tts_duration(path)
    try:
        with open(meta, "r", encoding="utf-8") as f:
            dur = float(json.load(f).get("duration") or dur)
    except Exception:
        pass
    return {"id": key, "duration": round(dur, 3), "cached": True, "chars": chars}


def run_edge_tts(text, voice="zh-CN-XiaoxiaoNeural", rate=0, volume=100):
    """生成 Edge-TTS 配音并缓存。返回 {id, duration, cached, chars}；失败抛异常。"""
    global _TTS_VOICE_RE
    import asyncio
    import hashlib
    import re as _re
    if _TTS_VOICE_RE is None:
        _TTS_VOICE_RE = _re.compile(r"^[A-Za-z0-9-]{3,64}$")

    text = (text or "").strip()
    if not text:
        raise ValueError("text is required")
    if len(text) > _TTS_MAX_CHARS:
        text = text[:_TTS_MAX_CHARS]
    voice = (voice or "zh-CN-XiaoxiaoNeural").strip() or "zh-CN-XiaoxiaoNeural"
    if not _TTS_VOICE_RE.match(voice):
        raise ValueError("bad voice: %s" % voice)
    rate_s = _tts_rel_rate(rate)
    vol_s = _tts_rel_volume(volume)

    os.makedirs(TTS_CACHE_DIR, exist_ok=True)
    _tts_cleanup_once()
    key = hashlib.sha1(("%s|%s|%s|%s" % (text, voice, rate_s, vol_s)).encode("utf-8")).hexdigest()
    path = os.path.join(TTS_CACHE_DIR, key + ".mp3")
    meta = os.path.join(TTS_CACHE_DIR, key + ".json")

    if os.path.isfile(path) and os.path.getsize(path) > 0:
        return _tts_cached_result(key, path, meta, len(text))

    try:
        import edge_tts
    except Exception as e:
        raise RuntimeError(
            "未安装 edge-tts（当前桥用的解释器：%s）。修复：在命令行执行 \"%s\" -m pip install edge-tts 后重试。原始错误：%s"
            % (sys.executable, sys.executable, e)
        )

    async def _save(tmp):
        c = edge_tts.Communicate(text, voice, rate=rate_s, volume=vol_s)
        await asyncio.wait_for(c.save(tmp), timeout=25)

    with _TTS_GEN_LOCK:
        # 等锁期间别的线程可能已经生成好了，先复查缓存
        if os.path.isfile(path) and os.path.getsize(path) > 0:
            return _tts_cached_result(key, path, meta, len(text))

        # 每次生成用独立临时文件，彻底避免并发共用同一个 .part
        tmp = path + ".%d.%d.part" % (os.getpid(), threading.get_ident())
        try:
            asyncio.run(_save(tmp))
            if not os.path.isfile(tmp) or os.path.getsize(tmp) <= 0:
                raise RuntimeError("Edge-TTS 返回空音频")
            # 杀毒/搜索索引可能短暂占用刚写完的文件，rename 失败时重试几次
            for attempt in range(10):
                try:
                    os.replace(tmp, path)
                    break
                except OSError:
                    if attempt == 9:
                        raise
                    time.sleep(0.2)
        finally:
            try:
                os.remove(tmp)
            except OSError:
                pass

        dur = _tts_duration(path)
        if dur <= 0:
            # ffprobe 不可用时按中文语速估算（约 4.5 字/秒），避免前端把视频剪得过短
            dur = max(1.0, len(text) / 4.5)
        try:
            with open(meta, "w", encoding="utf-8") as f:
                json.dump({"duration": dur, "chars": len(text)}, f)
        except Exception:
            pass
    return {"id": key, "duration": round(dur, 3), "cached": False, "chars": len(text)}


# ---------- Qwen3-0.6B 生图提示词翻译（直连） ----------
_LLM_MODEL_PATH = _resolve_path(
    r"models\LLM\qwen3-0.6b",
    r"models\LLM\qwen3-0.6b")
_llm = None
_llm_lock = threading.Lock()
_LLM_STYLE = {"auto": "", "garden": "（风格方向：春日花园清新风）", "star": "（风格方向：星空静谧风）",
              "floral": "（风格方向：花影雅致风）", "snow": "（风格方向：雪山壮美风）",
              "window": "（风格方向：窗台治愈风）"}


def _llm_load():
    global _llm
    if _llm is not None:
        return _llm
    import torch
    from transformers import AutoModelForCausalLM, AutoTokenizer
    torch.cuda.empty_cache()
    tok = AutoTokenizer.from_pretrained(_LLM_MODEL_PATH)
    model = AutoModelForCausalLM.from_pretrained(
        _LLM_MODEL_PATH, device_map="cuda", torch_dtype=torch.bfloat16,
        attn_implementation="sdpa")
    _llm = (model, tok)
    return _llm


def run_translate(text, style):
    """金句 -> 英文生图提示词。返回 dict{prompt} 或 dict{error}。"""
    with _llm_lock:
        try:
            _qwen_free_upstream()
            model, tok = _llm_load()
            import torch
            sys_p = ("/no_think\n你是AI绘画提示词翻译器。把中文金句翻译成一段英文绘图提示词（image generation prompt），"
                     "描述与句子意境相符的唯美画面。要求：竖构图，画面中上部留出纯净天空或柔和留白；"
                     "包含具体场景、主体、光线、色调、构图与氛围细节；画面中严禁出现任何文字、字母、标语、水印；"
                     "只输出提示词本身，不要解释、不要引号、不要前缀。")
            style_hint = _LLM_STYLE.get(style, "")
            user = "金句：" + text + style_hint + "\n请输出英文绘图提示词："
            msgs = [{"role": "system", "content": sys_p}, {"role": "user", "content": user}]
            prompt_str = tok.apply_chat_template(msgs, add_generation_prompt=True, tokenize=False)
            t = tok(prompt_str, return_tensors="pt").to("cuda")
            pad = tok.pad_token_id if tok.pad_token_id is not None else tok.eos_token_id
            out = model.generate(t["input_ids"], max_new_tokens=300, do_sample=True, temperature=0.9,
                                 top_p=0.9, repetition_penalty=1.1, pad_token_id=pad)
            res = tok.decode(out[0][t["input_ids"].shape[1]:], skip_special_tokens=True).strip()
            import re as _re
            res = _re.sub(r"<think>.*?</think>", "", res, flags=_re.S).strip()
            return {"prompt": res}
        except Exception as e:
            log.error("提示词翻译失败: %s\n%s", e, traceback.format_exc())
            return {"error": "translate: %s" % e}
        finally:
            try:
                global _llm
                if _llm is not None:
                    _llm = None
                    import torch
                    torch.cuda.empty_cache()
            except Exception as e:
                log.warning("翻译模型卸载/显存清理失败: %s", e)


# ---------- 大模型 API（OpenAI 兼容转发 + 本地 0.6B 回退） ----------
def run_llm_upstream(base, api_key, model, system, user):
    """转发 OpenAI 兼容 /chat/completions，返回文本。"""
    url = base.rstrip("/") + "/chat/completions"
    payload = {
        "model": model,
        "messages": [{"role": "system", "content": system}, {"role": "user", "content": user}],
        "temperature": 0.8, "max_tokens": 600,
    }
    req = urllib.request.Request(url, data=json.dumps(payload).encode("utf-8"), method="POST")
    req.add_header("Content-Type", "application/json")
    if api_key:
        req.add_header("Authorization", "Bearer " + api_key)
    with urllib.request.urlopen(req, timeout=180) as r:
        d = json.loads(r.read().decode("utf-8"))
    try:
        return d["choices"][0]["message"]["content"].strip()
    except Exception as e:
        raise ValueError("unexpected llm response: %s" % json.dumps(d, ensure_ascii=False)[:300]) from e


def run_llm_local(system, user):
    """本地 Qwen3-0.6B 兜底生成解读文本。"""
    with _llm_lock:
        try:
            _qwen_free_upstream()
            model, tok = _llm_load()
            import torch
            msgs = [{"role": "system", "content": system}, {"role": "user", "content": user}]
            prompt_str = tok.apply_chat_template(msgs, add_generation_prompt=True, tokenize=False)
            t = tok(prompt_str, return_tensors="pt").to("cuda")
            pad = tok.pad_token_id if tok.pad_token_id is not None else tok.eos_token_id
            out = model.generate(t["input_ids"], max_new_tokens=300, do_sample=True, temperature=0.9,
                                 top_p=0.9, repetition_penalty=1.1, pad_token_id=pad)
            res = tok.decode(out[0][t["input_ids"].shape[1]:], skip_special_tokens=True).strip()
            import re as _re
            res = _re.sub(r"<think>.*?</think>", "", res, flags=_re.S).strip()
            return res
        except Exception as e:
            log.error("本地 LLM 生成失败: %s", e)
            raise
        finally:
            try:
                global _llm
                if _llm is not None:
                    _llm = None
                    import torch
                    torch.cuda.empty_cache()
            except Exception as e:
                log.warning("本地 LLM 模型卸载/显存清理失败: %s", e)


def run_llm(base, api_key, model, system, user):
    """优先上游 OpenAI 兼容 API；失败/未配置时回退本地 0.6B。"""
    errs = []
    if base and model:
        try:
            text = run_llm_upstream(base, api_key, model, system, user)
            if text:
                return {"text": text, "source": "upstream"}
        except Exception as e:
            errs.append("upstream: %s" % e)
            log.warning("上游 LLM 调用失败: %s", e)
    try:
        text = run_llm_local(system, user)
        if text:
            return {"text": text, "source": "local"}
    except Exception as e:
        errs.append("local: %s" % e)
        log.warning("本地 LLM 回退失败: %s", e)
    raise ValueError("llm unavailable: " + ("; ".join(errs) or "no route"))


# ---------- 抖音链接解析（yt-dlp 只取元数据，不下载视频） ----------
def run_douyin_parse(url):
    import shutil, subprocess
    exe = shutil.which("yt-dlp") or shutil.which("yt-dlp.exe")
    if not exe:
        return {"error": "未检测到 yt-dlp，请先执行: pip install yt-dlp"}
    cmd = [exe, "--no-download", "--skip-download", "--dump-json", "--no-warnings",
           "--socket-timeout", "20", "--no-playlist", url]
    flags = getattr(subprocess, "CREATE_NO_WINDOW", 0)
    try:
        p = subprocess.run(cmd, capture_output=True, timeout=90, creationflags=flags)
    except subprocess.TimeoutExpired:
        return {"error": "解析超时(90秒)：链接可能失效、需要登录或网络受限"}
    out = p.stdout.decode("utf-8", "replace").strip()
    if p.returncode != 0 or not out:
        err = p.stderr.decode("utf-8", "replace").strip()[-300:]
        return {"error": "yt-dlp 解析失败: " + (err or ("exit %s" % p.returncode))}
    try:
        d = json.loads(out.splitlines()[-1])
    except Exception as e:
        return {"error": "yt-dlp 输出无法解析: %s" % e}
    return {
        "title": d.get("title") or "",
        "description": d.get("description") or "",
        "uploader": d.get("uploader") or "",
        "url": d.get("webpage_url") or url,
    }


def run_douyin_parse_batch(urls):
    """逐条解析链接，保留每条错误，避免一条失败阻断整批导入。"""
    results = []
    for url in urls or []:
        u = (url or "").strip()
        if not u:
            continue
        try:
            item = run_douyin_parse(u)
        except Exception as exc:
            item = {"url": u, "error": str(exc)}
        item.setdefault("url", u)
        results.append(item)
    return {"results": results}


# ---------- 音频转写 / 画面 OCR（可选依赖） ----------
def _decode_media(data, name):
    import base64
    import tempfile
    suffix = os.path.splitext(name or "")[1] or ".bin"
    fd, path = tempfile.mkstemp(prefix="qvs-media-", suffix=suffix)
    try:
        with os.fdopen(fd, "wb") as f:
            f.write(base64.b64decode(data or ""))
    except Exception:
        try:
            os.unlink(path)
        except OSError:
            pass
        raise
    return path


def _transcribe_audio(path):
    """Use faster-whisper or openai-whisper when present; model path is configurable."""
    model_name = os.environ.get("QVS_WHISPER_MODEL", "small")
    try:
        from faster_whisper import WhisperModel
        model = WhisperModel(model_name, device="auto", compute_type="int8")
        segs, _ = model.transcribe(path, language="zh", vad_filter=True)
        return " ".join((s.text or "").strip() for s in segs).strip(), "faster-whisper"
    except ImportError:
        pass
    try:
        import whisper
        model = whisper.load_model(model_name)
        out = model.transcribe(path, language="zh")
        return (out.get("text") or "").strip(), "openai-whisper"
    except ImportError:
        raise RuntimeError("未安装音频转写依赖，请安装 faster-whisper 或 openai-whisper")


def _extract_audio_track(path):
    import shutil
    import subprocess
    import tempfile
    ffmpeg = shutil.which("ffmpeg") or shutil.which("ffmpeg.exe")
    if not ffmpeg:
        raise RuntimeError("未检测到 ffmpeg，无法从视频提取音频")
    fd, wav = tempfile.mkstemp(prefix="qvs-audio-", suffix=".wav")
    os.close(fd)
    cmd = [ffmpeg, "-y", "-i", path, "-vn", "-ac", "1", "-ar", "16000", wav]
    p = subprocess.run(cmd, capture_output=True, timeout=180)
    if p.returncode != 0:
        try:
            os.unlink(wav)
        except OSError:
            pass
        raise RuntimeError("视频音频提取失败: " + p.stderr.decode("utf-8", "replace")[-300:])
    return wav


def _ocr_image(path):
    try:
        from PIL import Image
        import pytesseract
    except ImportError:
        raise RuntimeError("未安装画面 OCR 依赖，请安装 pillow、pytesseract，并配置 Tesseract")
    text = pytesseract.image_to_string(Image.open(path), lang="chi_sim+eng")
    return text.strip()


def run_media_extract(name, mime, mode, data):
    """Extract speech and/or visible text. Optional dependencies fail with actionable errors."""
    import os as _os
    import shutil
    import subprocess
    import tempfile
    path = _decode_media(data, name)
    extra = []
    errors = []
    wav = None
    try:
        is_image = (mime or "").startswith("image/") or (_os.path.splitext(name or "")[1].lower() in (".png", ".jpg", ".jpeg", ".webp"))
        is_video = (mime or "").startswith("video/") or (_os.path.splitext(name or "")[1].lower() in (".mp4", ".mov", ".mkv", ".webm"))
        if mode in ("auto", "visual") and (is_image or is_video):
            ocr_path = path
            if is_video:
                ffmpeg = shutil.which("ffmpeg") or shutil.which("ffmpeg.exe")
                if ffmpeg:
                    fd, frame = tempfile.mkstemp(prefix="qvs-frame-", suffix=".jpg")
                    os.close(fd)
                    p = subprocess.run([ffmpeg, "-y", "-ss", "00:00:01", "-i", path, "-frames:v", "1", frame],
                                       capture_output=True, timeout=120)
                    if p.returncode == 0:
                        ocr_path = frame
            try:
                text = _ocr_image(ocr_path)
                if text:
                    extra.append(("画面 OCR", text))
            except Exception as exc:
                errors.append("画面 OCR: %s" % exc)
            finally:
                if ocr_path != path:
                    try:
                        os.unlink(ocr_path)
                    except OSError:
                        pass
        if mode in ("auto", "audio") and (is_video or (mime or "").startswith("audio/") or
                                           (_os.path.splitext(name or "")[1].lower() in (".mp3", ".wav", ".m4a", ".aac", ".flac"))):
            try:
                wav = _extract_audio_track(path) if is_video else path
                text, source = _transcribe_audio(wav)
                if text:
                    extra.append((source, text))
            except Exception as exc:
                errors.append("音频转写: %s" % exc)
        if not extra:
            raise RuntimeError("；".join(errors) or "没有可用的提取结果")
        result = {"text": "\n".join("[%s] %s" % x for x in extra), "source": "+".join(x[0] for x in extra)}
        if errors:
            result["warnings"] = errors
        return result
    finally:
        for p in (path, wav):
            if p and p != path:
                try:
                    os.unlink(p)
                except OSError:
                    pass
        try:
            os.unlink(path)
        except OSError:
            pass


# ---------- 媒体库：images/bgm 目录列表与文件读取（供前端随机选背景/BGM） ----------
MEDIA_LIB = {
    "images": ("images", (".png", ".jpg", ".jpeg", ".webp", ".bmp")),
    "bgm": ("bgm", (".mp3", ".wav", ".m4a", ".aac", ".ogg", ".flac")),
    # 字体库：标题大字随机字体用；字体随项目目录一起拷贝即可换电脑使用
    "font": ("font", (".ttf", ".otf", ".woff", ".woff2", ".ttc")),
}


def run_media_lib():
    """列出 images / bgm 目录下的素材文件名。"""
    out = {}
    for kind, (rel, exts) in MEDIA_LIB.items():
        d = os.path.join(_SCRIPT_DIR, rel)
        names = []
        if os.path.isdir(d):
            for fn in sorted(os.listdir(d)):
                full = os.path.join(d, fn)
                if os.path.isfile(full) and os.path.splitext(fn)[1].lower() in exts:
                    names.append(fn)
        out[kind] = names
    return out


def run_media_file(kind, name):
    """安全读取素材文件：kind 白名单 + basename 清洗 + 扩展名白名单，防止路径穿越。"""
    if kind not in MEDIA_LIB:
        raise ValueError("bad kind: %s" % kind)
    rel, exts = MEDIA_LIB[kind]
    name = os.path.basename(name or "")
    if not name or os.path.splitext(name)[1].lower() not in exts:
        raise ValueError("bad name")
    path = os.path.join(_SCRIPT_DIR, rel, name)
    if not os.path.isfile(path):
        raise FileNotFoundError("file not found: %s" % name)
    with open(path, "rb") as f:
        return f.read(), name


# ---------- 批量导出落盘（导出\YYYY-MM-DD\ 目录；有类别则按 导出\YYYY-MM-DD\<类别>\ 分子目录） ----------
def _find_ffmpeg():
    import shutil
    local = os.path.join(_SCRIPT_DIR, "ffmpeg", "ffmpeg.exe")
    if os.path.isfile(local):
        return local
    return shutil.which("ffmpeg") or shutil.which("ffmpeg.exe")


def run_save_file(name, b64, convert=False, category=None):
    import base64, datetime, re as _re
    ext = ""
    if isinstance(name, str) and name.lower().endswith((".mp4", ".webm", ".mov", ".png", ".jpg", ".jpeg")):
        ext = os.path.splitext(name)[1].lower()
    day = datetime.date.today().isoformat()
    # 分类归档：传入类别时存 导出/<日期>/<类别>/，否则回退扁平 导出/<日期>/（向后兼容旧调用）
    root = os.path.join(_SCRIPT_DIR, "导出", day)
    if category:
        cat = _re.sub(r'[\\/:*?"<>|\r\n]+', "_", str(category).strip())[:60] or "未分类"
        root = os.path.join(root, cat)
    os.makedirs(root, exist_ok=True)
    safe = _re.sub(r'[\\/:*?"<>|\r\n]+', "_", (name or "video").strip())[:80] or "video"
    if not ext:
        ext = ".mp4"
    if not safe.lower().endswith((".mp4", ".webm", ".mov", ".png", ".jpg", ".jpeg")):
        safe += ext
    path = os.path.join(root, safe)
    n = 1
    while os.path.exists(path):
        stem, suffix = os.path.splitext(safe)
        path = os.path.join(root, "%s_%d%s" % (stem, n, suffix or ext))
        n += 1
    if convert and ext == ".mp4":
        # 浏览器只支持 WebM 录制时：先落盘 WebM 临时文件，再用 ffmpeg 转码成 MP4
        import subprocess
        tmp = path + ".tmp.webm"
        with open(tmp, "wb") as f:
            f.write(base64.b64decode(b64))
        ffmpeg = _find_ffmpeg()
        if not ffmpeg:
            try:
                os.remove(tmp)
            except OSError:
                pass
            raise RuntimeError("未检测到 ffmpeg，无法转码 MP4（请把 ffmpeg.exe 放到工具目录 ffmpeg\\ 下）")
        cmd = [ffmpeg, "-y", "-i", tmp, "-c:v", "libx264", "-preset", "medium", "-crf", "20",
               "-c:a", "aac", "-b:a", "192k", "-movflags", "+faststart", path]
        flags = getattr(subprocess, "CREATE_NO_WINDOW", 0)
        try:
            p = subprocess.run(cmd, capture_output=True, timeout=300, creationflags=flags)
        finally:
            try:
                os.remove(tmp)
            except OSError:
                pass
        if p.returncode != 0:
            err = (p.stderr or p.stdout or b"").decode("utf-8", "ignore")[-500:]
            raise RuntimeError("ffmpeg 转码失败: " + err)
        return {"path": path, "size": os.path.getsize(path), "converted": True}
    with open(path, "wb") as f:
        f.write(base64.b64decode(b64))
    return {"path": path, "size": os.path.getsize(path)}


# ---------- V2.20 离线快速导出：帧序列（JPEG）+ ffmpeg 合成视频 ----------
def run_video_compose(params, raw):
    """接收前端逐帧渲染的 JPEG 帧流（连续 [u32 BE 长度][帧数据]），用 ffmpeg 合成 MP4/WebM。
    params: frames/fps/w/h/bgm/vol/ext/name/category（来自 URL query）"""
    import struct, subprocess, datetime, tempfile, shutil, re as _re

    def _pint(k, d):
        try:
            return int(float(params.get(k)))
        except (TypeError, ValueError):
            return d

    frames = _pint("frames", 0)
    fps = _pint("fps", 30)
    w = _pint("w", 1080)
    h = _pint("h", 1920)
    if frames <= 0 or frames > 2000:
        raise ValueError("bad frames: %d" % frames)
    if fps <= 0 or fps > 120:
        raise ValueError("bad fps")
    if w <= 0 or h <= 0 or w > 4096 or h > 4096:
        raise ValueError("bad size")
    ffmpeg = _find_ffmpeg()
    if not ffmpeg:
        raise RuntimeError("未检测到 ffmpeg，无法合成（请把 ffmpeg.exe 放到工具目录 ffmpeg\\ 下）")
    bgm = os.path.basename((params.get("bgm") or "").strip())
    try:
        vol = float(params.get("vol") or "1")
    except ValueError:
        vol = 1.0
    ext = (params.get("ext") or "mp4").lower().strip(".")
    if ext not in ("mp4", "webm", "mov"):
        ext = "mp4"
    name = (params.get("name") or "video." + ext).strip()
    category = (params.get("category") or "").strip() or None
    # 解析帧流：连续 [u32 BE 长度][JPEG 数据]
    tmpdir = tempfile.mkdtemp(prefix="compose_", dir=_SCRIPT_DIR)
    try:
        pos, n = 0, 0
        while n < frames and pos + 4 <= len(raw):
            ln = struct.unpack(">I", raw[pos:pos + 4])[0]
            pos += 4
            if ln <= 0 or pos + ln > len(raw):
                raise ValueError("帧数据损坏 @%d" % n)
            fp = os.path.join(tmpdir, "frame_%04d.jpg" % (n + 1))
            with open(fp, "wb") as f:
                f.write(raw[pos:pos + ln])
            pos += ln
            n += 1
        if n < frames:
            raise ValueError("帧数不足: 收到 %d/%d" % (n, frames))
        # 输出路径（与 save_file 相同归档规则：导出\YYYY-MM-DD[\类别]\）
        day = datetime.date.today().isoformat()
        root = os.path.join(_SCRIPT_DIR, "导出", day)
        if category:
            cat = _re.sub(r'[\\/:*?"<>|\r\n]+', "_", category)[:60] or "未分类"
            root = os.path.join(root, cat)
        os.makedirs(root, exist_ok=True)
        safe = _re.sub(r'[\\/:*?"<>|\r\n]+', "_", name)[:80] or ("video." + ext)
        if not safe.lower().endswith("." + ext):
            safe += "." + ext
        path = os.path.join(root, safe)
        n2 = 1
        while os.path.exists(path):
            stem, suffix = os.path.splitext(safe)
            path = os.path.join(root, "%s_%d%s" % (stem, n2, suffix))
            n2 += 1
        cmd = [ffmpeg, "-y", "-framerate", str(fps), "-i", os.path.join(tmpdir, "frame_%04d.jpg")]
        bgm_path = None
        if bgm:
            rel = MEDIA_LIB.get("bgm", ("bgm", ()))[0]
            cand = os.path.join(_SCRIPT_DIR, rel, bgm)
            if os.path.isfile(cand):
                bgm_path = cand
        # 配音：前端传 Edge-TTS 的 id；找不到缓存文件就当没有配音
        voice_id = (params.get("voice") or "").strip()
        if not _re.fullmatch(r"[a-f0-9]{40}", voice_id):
            voice_id = ""
        voice_path = os.path.join(TTS_CACHE_DIR, voice_id + ".mp3") if voice_id else ""
        if voice_path and not os.path.isfile(voice_path):
            voice_path = ""
        try:
            voice_vol = max(0.0, min(2.0, float(params.get("voicevol") or "1")))
        except ValueError:
            voice_vol = 1.0
        try:
            voice_delay = max(0.0, min(10.0, float(params.get("voicedelay") or "0.3")))
        except ValueError:
            voice_delay = 0.3

        bgm_idx = voice_idx = None
        idx = 1                      # 0 是帧序列
        if bgm_path:
            cmd += ["-stream_loop", "-1", "-i", bgm_path]
            bgm_idx = idx
            idx += 1
        if voice_path:
            cmd += ["-i", voice_path]
            voice_idx = idx
            idx += 1

        audio_parts, labels = [], []
        if bgm_idx is not None:
            audio_parts.append("[%d:a]volume=%.3f[bg]" % (bgm_idx, max(0.0, min(2.0, vol))))
            labels.append("[bg]")
        if voice_idx is not None:
            ms = int(round(voice_delay * 1000))
            audio_parts.append("[%d:a]adelay=%d|%d,volume=%.3f[vo]"
                               % (voice_idx, ms, ms, voice_vol))
            labels.append("[vo]")
        has_audio = bool(labels)
        if has_audio:
            if len(labels) == 2:
                # BGM 已由前端压低，这里不再做 sidechain；normalize=0 避免人声被 amix 砍半
                audio_parts.append("[bg][vo]amix=inputs=2:duration=longest:"
                                   "dropout_transition=0:normalize=0,apad[a]")
            else:
                audio_parts.append("%sapad[a]" % labels[0])
            cmd += ["-filter_complex", ";".join(audio_parts),
                    "-map", "0:v", "-map", "[a]", "-t", "%.3f" % (frames / float(fps))]
        if ext == "mp4":
            cmd += ["-c:v", "libx264", "-preset", "medium", "-crf", "20", "-pix_fmt", "yuv420p"]
            if has_audio:
                cmd += ["-c:a", "aac", "-b:a", "192k"]
            else:
                cmd += ["-an"]
            cmd += ["-movflags", "+faststart"]
        elif ext == "webm":
            cmd += ["-c:v", "libvpx-vp9", "-b:v", "0", "-crf", "34"]
            if has_audio:
                cmd += ["-c:a", "libopus"]
            else:
                cmd += ["-an"]
        else:  # mov
            cmd += ["-c:v", "libx264", "-preset", "medium", "-crf", "20", "-pix_fmt", "yuv420p"]
            if has_audio:
                cmd += ["-c:a", "aac", "-b:a", "192k"]
            else:
                cmd += ["-an"]
        cmd.append(path)
        flags = getattr(subprocess, "CREATE_NO_WINDOW", 0)
        p = subprocess.run(cmd, capture_output=True, timeout=600, creationflags=flags)
        if p.returncode != 0:
            err = (p.stderr or p.stdout or b"").decode("utf-8", "ignore")[-600:]
            raise RuntimeError("ffmpeg 合成失败: " + err)
        return {"path": path, "size": os.path.getsize(path), "frames": n, "fps": fps,
                "has_audio": has_audio, "composed": True}
    finally:
        shutil.rmtree(tmpdir, ignore_errors=True)


# ---------- 飞书多维表格 ----------
FEISHU_HOST = "https://open.feishu.cn"
# 说明：不再补建/创建任何列。前端按表内真实列名读取，缺失列对应表单项留空。
_feishu_tok_cache = {}

def _feishu_token(app_id, app_secret):
    key = (app_id or "", app_secret or "")
    if not key[0] or not key[1]:
        raise ValueError("缺少 app_id / app_secret，请先在飞书设置里填写")
    now = time.time()
    hit = _feishu_tok_cache.get(key)
    if hit and hit[1] > now + 60:
        return hit[0]
    req = urllib.request.Request(
        FEISHU_HOST + "/open-apis/auth/v3/tenant_access_token/internal",
        data=json.dumps({"app_id": key[0], "app_secret": key[1]}).encode("utf-8"), method="POST")
    req.add_header("Content-Type", "application/json")
    with urllib.request.urlopen(req, timeout=20) as r:
        d = json.loads(r.read().decode("utf-8"))
    if d.get("code") != 0:
        raise ValueError("飞书凭证错误(code=%s): %s" % (d.get("code"), d.get("msg")))
    tok = d.get("tenant_access_token")
    _feishu_tok_cache[key] = (tok, now + int(d.get("expire") or 7200))
    return tok

def _feishu_req(method, path, token, body=None, params=None):
    url = FEISHU_HOST + path
    if params:
        url += "?" + urllib.parse.urlencode(params)
    data = json.dumps(body).encode("utf-8") if body is not None else None
    req = urllib.request.Request(url, data=data, method=method)
    req.add_header("Authorization", "Bearer " + token)
    if data is not None:
        req.add_header("Content-Type", "application/json")
    try:
        with urllib.request.urlopen(req, timeout=30) as r:
            d = json.loads(r.read().decode("utf-8"))
    except urllib.error.HTTPError as e:
        raise ValueError("飞书接口 HTTP %s: %s" % (e.code, e.read().decode("utf-8", "replace")[:300]))
    if d.get("code") != 0:
        raise ValueError("飞书接口错误(code=%s): %s" % (d.get("code"), d.get("msg")))
    return d.get("data") or {}

def feishu_ensure_table(app_id, app_secret, app_token, table_id):
    tok = _feishu_token(app_id, app_secret)
    base = "/open-apis/bitable/v1/apps/%s" % app_token
    if not table_id:
        data = _feishu_req("GET", base + "/tables", tok, params={"page_size": 100})
        for t in data.get("items") or []:
            if t.get("name") == "视频清单":
                table_id = t.get("table_id")
                break
        if not table_id:
            raise ValueError("未找到表格：请粘贴多维表格链接（含 ?table= 参数）或手动填写 table_id；不会自动建表或新建列")
    # 只读取真实字段，绝不补建缺失列（表里没有的字段对应表单项留空）
    data = _feishu_req("GET", base + "/tables/%s/fields" % table_id, tok, params={"page_size": 100})
    fields = [f.get("field_name") for f in data.get("items") or []]
    return {"table_id": table_id, "fields": fields}

def feishu_write_rows(app_id, app_secret, app_token, table_id, rows):
    tok = _feishu_token(app_id, app_secret)
    recs = [{"fields": {k: ("" if v is None else str(v)) for k, v in r.items()}} for r in rows]
    created = 0
    for i in range(0, len(recs), 500):
        data = _feishu_req("POST", "/open-apis/bitable/v1/apps/%s/tables/%s/records/batch_create"
                           % (app_token, table_id), tok, body={"records": recs[i:i + 500]})
        created += len((data.get("records") or []))
    return {"created": created}

def feishu_read_rows(app_id, app_secret, app_token, table_id):
    tok = _feishu_token(app_id, app_secret)
    out, token = [], ""
    while True:
        params = {"page_size": 500}
        if token:
            params["page_token"] = token
        data = _feishu_req("GET", "/open-apis/bitable/v1/apps/%s/tables/%s/records"
                           % (app_token, table_id), tok, params=params)
        for it in data.get("items") or []:
            f = dict(it.get("fields") or {})
            f["_record_id"] = it.get("record_id")
            out.append(f)
        if not data.get("has_more"):
            break
        token = data.get("page_token") or ""
    return {"rows": out}

def feishu_update_status(app_id, app_secret, app_token, table_id, record_id, status, output_file=""):
    tok = _feishu_token(app_id, app_secret)
    # 只回传「生成状态」列，其他列（含输出文件）一律不修改
    fields = {"生成状态": status}
    _feishu_req("PUT", "/open-apis/bitable/v1/apps/%s/tables/%s/records/%s"
                % (app_token, table_id, record_id), tok, body={"fields": fields})
    return {"ok": True}


# ---------- 内容二创：收件箱监听 + 待入队结果 ----------
# 工作流：把 .txt/.md/图片 丢进 inbox\ → 本地大模型按《内容生成prompt.txt》改写 →
#         结果写入 rewrite_pending.json → 工作台轮询 /rewrite/pending 自动入队出片。
INBOX_DIR = os.path.join(_SCRIPT_DIR, "inbox")
INBOX_DONE = os.path.join(INBOX_DIR, "_done")
REWRITE_PENDING = os.path.join(_SCRIPT_DIR, "rewrite_pending.json")
INBOX_EXTS_TEXT = (".txt", ".md", ".markdown")
INBOX_EXTS_IMG = (".png", ".jpg", ".jpeg", ".webp", ".bmp")
INBOX_POLL_SEC = 3
_pending_lock = threading.Lock()
_scan_lock = threading.Lock()
_inbox_stop = threading.Event()

RW_DEFAULTS = {
    "rwOrder": "auto",
    "rwGgufPath": "", "rwGgufMmproj": "", "rwGgufCtx": "8192", "rwGgufGpuLayers": "-1",
    "rwLocalBase": "http://127.0.0.1:11434/v1", "rwLocalModel": "qwen2.5vl:7b",
    "rwApiBase": "", "rwApiKey": "", "rwApiModel": "", "rwApiThinking": "",
    "rwTemp": "0.4", "rwMaxTokens": "900", "rwTimeout": "300", "rwInboxAuto": "1",
    "rwAutoStartLocal": "1",
}


def rw_cfg():
    """读取 config.json 并补全内容二创默认值（字段随设置面板持久化）。"""
    c = {}
    try:
        if os.path.exists(CONFIG_PATH):
            with open(CONFIG_PATH, "r", encoding="utf-8") as f:
                c = json.load(f) or {}
    except Exception as e:
        log.warning("读取内容二创配置失败: %s", e)
    out = dict(RW_DEFAULTS)
    if isinstance(c, dict):
        out.update(c)
    return out


def _pending_load():
    try:
        with open(REWRITE_PENDING, "r", encoding="utf-8") as f:
            d = json.load(f)
        return d if isinstance(d, list) else []
    except Exception:
        return []


def _pending_save(items):
    try:
        tmp = REWRITE_PENDING + ".tmp"
        with open(tmp, "w", encoding="utf-8") as f:
            json.dump(items[-500:], f, ensure_ascii=False, indent=1)
        os.replace(tmp, REWRITE_PENDING)
    except Exception as e:
        log.warning("写入待入队结果失败: %s", e)


def pending_add(item):
    with _pending_lock:
        items = _pending_load()
        it = dict(item)
        it["id"] = "rw%d_%d" % (int(time.time() * 1000), len(items))
        it["created"] = time.strftime("%Y-%m-%d %H:%M:%S")
        items.append(it)
        _pending_save(items)
        return it


def pending_ack(ids):
    with _pending_lock:
        items = _pending_load()
        if ids:
            drop = set(ids)
            items = [x for x in items if x.get("id") not in drop]
        else:
            items = []
        _pending_save(items)
        return len(items)


def _read_text_file(path):
    for enc in ("utf-8", "utf-8-sig", "gbk", "utf-16"):
        try:
            with open(path, "r", encoding=enc) as f:
                return f.read()
        except (UnicodeDecodeError, UnicodeError, LookupError):
            continue
    with open(path, "rb") as f:
        return f.read().decode("utf-8", "replace")


def _img_data_url(path):
    import base64
    ext = os.path.splitext(path)[1].lower().lstrip(".")
    mime = {"png": "image/png", "jpg": "image/jpeg", "jpeg": "image/jpeg",
            "webp": "image/webp", "bmp": "image/bmp"}.get(ext, "image/png")
    with open(path, "rb") as f:
        b64 = base64.b64encode(f.read()).decode("ascii")
    return "data:%s;base64,%s" % (mime, b64)


def _inbox_scan():
    """扫描收件箱：逐文件跑内容二创并写入待入队；处理完移入 inbox\\_done\\。
    返回处理条数；若已有扫描在进行则立即返回 -1（不排队，避免重复处理）。"""
    if not _scan_lock.acquire(blocking=False):
        return -1
    try:
        return _inbox_scan_locked()
    finally:
        _scan_lock.release()


def _inbox_scan_locked():
    cfg = rw_cfg()
    if str(cfg.get("rwInboxAuto", "1")) == "0":
        return 0
    os.makedirs(INBOX_DIR, exist_ok=True)
    os.makedirs(INBOX_DONE, exist_ok=True)
    n = 0
    for fn in sorted(os.listdir(INBOX_DIR)):
        p = os.path.join(INBOX_DIR, fn)
        if not os.path.isfile(p):
            continue
        ext = os.path.splitext(fn)[1].lower()
        if ext not in INBOX_EXTS_TEXT and ext not in INBOX_EXTS_IMG:
            continue
        t0 = time.time()
        try:
            if ext in INBOX_EXTS_TEXT:
                res = CRE.rewrite(cfg, text=_read_text_file(p))
            else:
                res = CRE.rewrite(cfg, text="", image_data_url=_img_data_url(p))
            res["file"] = fn
            res["elapsed"] = round(time.time() - t0, 1)
            pending_add(res)
            log.info("收件箱已生成：%s → %s", fn, res.get("title1"))
        except Exception as e:
            log.warning("收件箱处理失败 %s: %s", fn, e)
            pending_add({"ok": False, "error": str(e), "file": fn, "title1": "",
                         "title2": "", "pill": "", "items": "", "titleStyle": "",
                         "bodyStyle": "", "source": "-", "elapsed": round(time.time() - t0, 1)})
        # 无论成功失败都移走，避免死循环；失败原因已记进待入队条目
        dest = os.path.join(INBOX_DONE, fn)
        if os.path.exists(dest):
            dest = os.path.join(INBOX_DONE, "%d_%s" % (int(time.time()), fn))
        try:
            os.replace(p, dest)
        except OSError:
            try:
                os.remove(p)
            except OSError:
                pass
        n += 1
    return n


def _inbox_loop():
    while not _inbox_stop.is_set():
        try:
            _inbox_scan()
        except Exception as e:
            log.warning("收件箱轮询异常: %s", e)
        _inbox_stop.wait(INBOX_POLL_SEC)


def start_inbox_watcher():
    if CRE is None:
        log.warning("内容二创引擎不可用，收件箱监听未启动")
        return
    try:
        os.makedirs(INBOX_DIR, exist_ok=True)
        os.makedirs(INBOX_DONE, exist_ok=True)
    except OSError as e:
        log.warning("创建收件箱目录失败: %s", e)
    threading.Thread(target=_inbox_loop, name="inbox-watcher", daemon=True).start()
    log.info("内容二创收件箱监听已启动：%s（丢入 .txt/.md/图片 即自动生成）", INBOX_DIR)


def comfy_request(method, path, payload=None, timeout=300, raw=False):
    url = UPSTREAM + path
    data = None
    headers = {}
    if payload is not None:
        data = json.dumps(payload).encode("utf-8")
        headers["Content-Type"] = "application/json"
    req = urllib.request.Request(url, data=data, headers=headers, method=method)
    try:
        with urllib.request.urlopen(req, timeout=timeout) as r:
            body = r.read()
            ctype = r.headers.get("Content-Type", "")
            if raw:
                return r.status, body, ctype
            if not body:
                return r.status, {}, ctype
            return r.status, json.loads(body.decode("utf-8")), ctype
    except urllib.error.HTTPError as e:
        return e.code, e.read().decode("utf-8", "replace"), ""
    except urllib.error.URLError as e:
        return -1, {"error": str(e.reason)}, ""


class Handler(http.server.BaseHTTPRequestHandler):
    protocol_version = "HTTP/1.1"

    def _cors(self):
        self.send_header("Access-Control-Allow-Origin", "*")
        self.send_header("Access-Control-Allow-Methods", "GET, POST, OPTIONS")
        self.send_header("Access-Control-Allow-Headers", "Content-Type, Authorization")
        self.send_header("Access-Control-Allow-Private-Network", "true")

    def _proxy(self, method):
        try:
            body = None
            if self.headers.get("Content-Length"):
                body = self.rfile.read(int(self.headers["Content-Length"]))
            url = UPSTREAM + self.path
            req = urllib.request.Request(url, data=body, method=method)
            ct = self.headers.get("Content-Type")
            if ct:
                req.add_header("Content-Type", ct)
            with urllib.request.urlopen(req, timeout=300) as r:
                data = r.read()
                self.send_response(r.status)
                self._cors()
                ctype = r.headers.get("Content-Type")
                if ctype:
                    self.send_header("Content-Type", ctype)
                self.send_header("Content-Length", str(len(data)))
                self.end_headers()
                self.wfile.write(data)
        except urllib.error.HTTPError as e:
            data = e.read()
            self.send_response(e.code)
            self._cors()
            self.send_header("Content-Length", str(len(data)))
            self.end_headers()
            self.wfile.write(data)
        except Exception as e:
            log.error("代理请求失败 %s %s: %s", method, self.path, e)
            body = json.dumps({"error": str(e)}).encode("utf-8")
            self.send_response(502)
            self._cors()
            self.send_header("Content-Type", "application/json")
            self.send_header("Content-Length", str(len(body)))
            self.end_headers()
            self.wfile.write(body)

    def _reply_json(self, code, obj):
        body = json.dumps(obj, ensure_ascii=False).encode("utf-8")
        self.send_response(code)
        self._cors()
        self.send_header("Content-Type", "application/json")
        self.send_header("Content-Length", str(len(body)))
        self.end_headers()
        self.wfile.write(body)

    def _handle_qwen_tts(self):
        try:
            ln = int(self.headers.get("Content-Length") or 0)
            data = json.loads(self.rfile.read(ln).decode("utf-8"))
        except Exception as e:
            log.warning("TTS 请求体解析失败: %s", e)
            self._reply_json(400, {"error": "bad body: %s" % e})
            return
        text = (data.get("text") or "").strip()
        if not text:
            self._reply_json(400, {"error": "text is required"})
            return
        instruct = data.get("instruct") or ""
        voice = data.get("voice") or "female"
        speaker = data.get("speaker") or ""
        try:
            wav, sr, err = run_qwen_tts(text, instruct, voice, speaker)
        except Exception as e:
            log.error("TTS 端点异常: %s", e)
            self._reply_json(502, {"error": str(e)})
            return
        if err is not None:
            self._reply_json(502, err)
            return
        self.send_response(200)
        self._cors()
        self.send_header("Content-Type", "audio/wav")
        self.send_header("Content-Length", str(len(wav)))
        self.send_header("X-QVS-Duration-Hint", "qwen-tts")
        self.end_headers()
        self.wfile.write(wav)

    def _handle_tts(self):
        """Edge-TTS：POST {text, voice, rate, volume} → {ok, id, duration, cached}。"""
        try:
            ln = int(self.headers.get("Content-Length") or 0)
            data = json.loads(self.rfile.read(ln).decode("utf-8"))
        except Exception as e:
            log.warning("Edge-TTS 请求体解析失败: %s", e)
            self._reply_json(400, {"error": "bad body: %s" % e})
            return
        text = (data.get("text") or "").strip()
        if not text:
            self._reply_json(400, {"error": "text is required"})
            return
        try:
            res = run_edge_tts(text, data.get("voice") or "zh-CN-XiaoxiaoNeural",
                               data.get("rate") or 0, data.get("volume", 100))
            res["ok"] = True
            self._reply_json(200, res)
        except Exception as e:
            log.warning("Edge-TTS 失败: %s", e)
            self._reply_json(502, {"error": "edge tts: %s" % e})

    def _handle_tts_audio(self):
        """按 id 返回缓存的 mp3（用于试听和前端解码）。"""
        import re as _re
        qs = urllib.parse.parse_qs(urllib.parse.urlparse(self.path).query)
        tid = (qs.get("id") or [""])[0]
        if not _re.fullmatch(r"[a-f0-9]{40}", tid):
            self._reply_json(400, {"error": "bad id"})
            return
        path = os.path.join(TTS_CACHE_DIR, tid + ".mp3")
        if not os.path.isfile(path):
            self._reply_json(404, {"error": "audio not found"})
            return
        with open(path, "rb") as f:
            data = f.read()
        self.send_response(200)
        self._cors()
        self.send_header("Content-Type", "audio/mpeg")
        self.send_header("Content-Length", str(len(data)))
        self.send_header("Cache-Control", "public, max-age=86400")
        self.end_headers()
        self.wfile.write(data)

    def _handle_translate(self):
        try:
            ln = int(self.headers.get("Content-Length") or 0)
            data = json.loads(self.rfile.read(ln).decode("utf-8"))
        except Exception as e:
            log.warning("翻译请求体解析失败: %s", e)
            self._reply_json(400, {"error": "bad body: %s" % e})
            return
        text = (data.get("text") or "").strip()
        if not text:
            self._reply_json(400, {"error": "text is required"})
            return
        style = (data.get("style") or "auto")
        res = run_translate(text, style)
        if "error" in res:
            self._reply_json(502, res)
            return
        self._reply_json(200, res)

    def _handle_llm(self):
        try:
            ln = int(self.headers.get("Content-Length") or 0)
            data = json.loads(self.rfile.read(ln).decode("utf-8"))
        except Exception as e:
            log.warning("LLM 请求体解析失败: %s", e)
            self._reply_json(400, {"error": "bad body: %s" % e})
            return
        base = (data.get("base") or "").strip()
        api_key = (data.get("api_key") or "").strip()
        model = (data.get("model") or "").strip()
        system = (data.get("system") or "").strip()
        user = (data.get("user") or "").strip()
        if not user:
            self._reply_json(400, {"error": "user is required"})
            return
        try:
            res = run_llm(base, api_key, model, system, user)
        except Exception as e:
            log.error("LLM 端点异常: %s", e)
            self._reply_json(502, {"error": "llm: %s" % e})
            return
        self._reply_json(200, res)

    def _read_json(self):
        ln = int(self.headers.get("Content-Length") or 0)
        return json.loads(self.rfile.read(ln).decode("utf-8"))

    def _drain_body(self):
        """POST 处理器若不解析 body，必须把 Content-Length 指定的字节读干净，
        否则残留数据会被当成下一个请求行（keep-alive 连接下报 501）。"""
        try:
            ln = int(self.headers.get("Content-Length") or 0)
            if ln > 0:
                self.rfile.read(ln)
        except Exception:
            pass

    def _handle_douyin_parse(self):
        try:
            d = self._read_json()
        except Exception as e:
            self._reply_json(400, {"error": "bad body: %s" % e})
            return
        url = (d.get("url") or "").strip()
        if not url:
            self._reply_json(400, {"error": "缺少 url"})
            return
        log.info("抖音解析: %s", url)
        self._reply_json(200, run_douyin_parse(url))

    def _handle_douyin_parse_batch(self):
        try:
            d = self._read_json()
            urls = d.get("urls") or []
            if not isinstance(urls, list):
                raise ValueError("urls 必须是数组")
            self._reply_json(200, run_douyin_parse_batch(urls))
        except Exception as e:
            log.warning("批量抖音解析失败: %s", e)
            self._reply_json(400, {"error": str(e)})

    def _handle_media_extract(self):
        try:
            d = self._read_json()
            name = (d.get("name") or "media.bin").strip()
            mime = d.get("mime") or ""
            mode = d.get("mode") or "auto"
            data = d.get("data") or ""
            if not data:
                raise ValueError("缺少媒体数据")
            out = run_media_extract(name, mime, mode, data)
            self._reply_json(200, out)
        except Exception as e:
            log.warning("媒体文字提取失败: %s", e)
            self._reply_json(502, {"error": str(e)})

    def _handle_save_file(self):
        try:
            d = self._read_json()
            res = run_save_file(d.get("name"), d.get("data") or "", bool(d.get("convert")), d.get("category") or None)
            self._reply_json(200, res)
        except Exception as e:
            log.error("保存文件失败: %s", e)
            self._reply_json(500, {"error": str(e)})

    def _handle_video_compose(self):
        try:
            qs = urllib.parse.parse_qs(urllib.parse.urlparse(self.path).query)
            params = {k: v[0] for k, v in qs.items()}
            ln = int(self.headers.get("Content-Length") or 0)
            raw = self.rfile.read(ln)
            res = run_video_compose(params, raw)
            self._reply_json(200, res)
        except Exception as e:
            log.error("视频合成失败: %s", e)
            self._reply_json(500, {"error": str(e)})

    def _handle_media_lib(self):
        try:
            self._reply_json(200, run_media_lib())
        except Exception as e:
            log.warning("媒体库列表失败: %s", e)
            self._reply_json(500, {"error": str(e)})

    def _handle_media_file(self):
        qs = urllib.parse.parse_qs(urllib.parse.urlparse(self.path).query)
        kind = (qs.get("kind") or [""])[0]
        name = (qs.get("name") or [""])[0]
        try:
            data, name2 = run_media_file(kind, name)
        except Exception as e:
            log.warning("媒体文件读取失败 %s/%s: %s", kind, name, e)
            self._reply_json(404, {"error": str(e)})
            return
        ext = os.path.splitext(name2)[1].lstrip(".").lower()
        ctype = {"png": "image/png", "jpg": "image/jpeg", "jpeg": "image/jpeg", "webp": "image/webp",
                 "bmp": "image/bmp", "mp3": "audio/mpeg", "wav": "audio/wav", "m4a": "audio/mp4",
                 "aac": "audio/aac", "ogg": "audio/ogg", "flac": "audio/flac",
                 "ttf": "font/ttf", "otf": "font/otf", "woff": "font/woff", "woff2": "font/woff2",
                 "ttc": "font/collection"}.get(ext, "application/octet-stream")
        self.send_response(200)
        self._cors()
        self.send_header("Content-Type", ctype)
        self.send_header("Content-Length", str(len(data)))
        self.end_headers()
        self.wfile.write(data)

    def _handle_feishu(self, p):
        try:
            d = self._read_json()
        except Exception as e:
            self._reply_json(400, {"error": "bad body: %s" % e})
            return
        aid, asec = d.get("app_id"), d.get("app_secret")
        atk, tid = d.get("app_token"), d.get("table_id")
        try:
            if p == "/feishu/ensure_table":
                res = feishu_ensure_table(aid, asec, atk, tid or "")
            elif p == "/feishu/write_rows":
                res = feishu_write_rows(aid, asec, atk, tid, d.get("rows") or [])
            elif p == "/feishu/read_rows":
                res = feishu_read_rows(aid, asec, atk, tid)
            elif p == "/feishu/update_status":
                res = feishu_update_status(aid, asec, atk, tid, d.get("record_id"), d.get("status"),
                                           d.get("output_file") or "")
            else:
                self._reply_json(404, {"error": "unknown feishu endpoint"})
                return
            self._reply_json(200, res)
        except Exception as e:
            log.warning("飞书调用失败 %s: %s", p, e)
            self._reply_json(502, {"error": str(e)})

    # ---------- 本地配置持久化（飞书凭证落盘到 config.json） ----------
    def _handle_config_get(self):
        try:
            if os.path.exists(CONFIG_PATH):
                with open(CONFIG_PATH, "r", encoding="utf-8") as f:
                    data = json.load(f)
            else:
                data = {}
            self._reply_json(200, data if isinstance(data, dict) else {})
        except Exception as e:
            log.warning("读取 config.json 失败: %s", e)
            self._reply_json(200, {})

    def _handle_config_save(self):
        try:
            d = self._read_json()
        except Exception as e:
            self._reply_json(400, {"error": "bad body: %s" % e})
            return
        if not isinstance(d, dict):
            self._reply_json(400, {"error": "body must be a JSON object"})
            return
        try:
            old = {}
            if os.path.exists(CONFIG_PATH):
                with open(CONFIG_PATH, "r", encoding="utf-8") as f:
                    old = json.load(f) or {}
            if isinstance(old, dict):
                for k in ("rwApiBase", "rwApiModel"):
                    if not str(d.get(k) or "").strip() and str(old.get(k) or "").strip():
                        d[k] = old[k]
            tmp = CONFIG_PATH + ".tmp"
            with open(tmp, "w", encoding="utf-8") as f:
                json.dump(d, f, ensure_ascii=False, indent=2)
            os.replace(tmp, CONFIG_PATH)
            self._reply_json(200, {"ok": True, "path": CONFIG_PATH})
        except Exception as e:
            log.warning("写入 config.json 失败: %s", e)
            self._reply_json(502, {"error": str(e)})

    # ---------- 内容二创 ----------
    def _handle_rewrite(self):
        if CRE is None:
            self._reply_json(503, {"error": "内容二创引擎未加载（content_rewrite.py 缺失）"})
            return
        try:
            d = self._read_json()
        except Exception as e:
            self._reply_json(400, {"error": "bad body: %s" % e})
            return
        text = d.get("text") or ""
        img = (d.get("image") or "").strip()
        if img and not img.startswith("data:"):
            img = "data:image/png;base64," + img
        if not text.strip() and not img:
            self._reply_json(400, {"error": "请提供原文文字或图片"})
            return
        try:
            res = CRE.rewrite(rw_cfg(), text=text, image_data_url=img or None)
            self._reply_json(200, res)
        except Exception as e:
            log.warning("内容二创失败: %s", e)
            self._reply_json(502, {"error": str(e)})

    def _handle_voice_script(self):
        """按《配音文案prompt.txt》单独生成一句开场配音（飞书行没有配音文案时用）。"""
        if CRE is None or not hasattr(CRE, "make_voice_script"):
            self._reply_json(503, {"error": "内容二创引擎未加载（content_rewrite.py 缺失）"})
            return
        try:
            d = self._read_json()
        except Exception as e:
            self._reply_json(400, {"error": "bad body: %s" % e})
            return
        title1 = (d.get("title1") or "").strip()
        title2 = (d.get("title2") or "").strip()
        items = (d.get("items") or "").strip()
        source = (d.get("text") or d.get("source_text") or "").strip()
        if not (title1 or items or source):
            self._reply_json(400, {"error": "title1 / items / text 至少提供一个"})
            return
        try:
            res = CRE.make_voice_script(rw_cfg(), title1=title1, title2=title2,
                                        items=items, source_text=source)
            res["ok"] = True
            self._reply_json(200, res)
        except Exception as e:
            log.warning("开场配音文案生成失败: %s", e)
            self._reply_json(502, {"error": str(e)})

    def _handle_rewrite_pending(self):
        items = _pending_load()
        self._reply_json(200, {"items": items, "count": len(items)})

    def _handle_rewrite_warm(self):
        """预热本地模型（打开弹窗时调用），让首次生成不必再等载入。"""
        self._drain_body()
        if CRE is None or not hasattr(CRE, "warm"):
            self._reply_json(503, {"error": "内容二创引擎未加载"})
            return
        try:
            self._reply_json(200, CRE.warm(rw_cfg()))
        except Exception as e:
            log.warning("内容二创预热失败: %s", e)
            self._reply_json(502, {"error": str(e)})

    def _handle_rewrite_ack(self):
        try:
            d = self._read_json()
        except Exception:
            d = {}
        self._reply_json(200, {"ok": True, "remaining": pending_ack(d.get("ids") or [])})

    def _handle_rewrite_scan(self):
        self._drain_body()
        try:
            n = _inbox_scan()
            busy = n < 0
            self._reply_json(200, {"ok": True, "scanned": max(n, 0), "busy": busy})
        except Exception as e:
            self._reply_json(500, {"error": str(e)})

    def _handle_rewrite_status(self):
        try:
            st = CRE.status(rw_cfg()) if CRE else {"error": "engine missing"}
        except Exception as e:
            st = {"error": str(e)}
        st["inbox"] = INBOX_DIR
        st["inboxAuto"] = str(rw_cfg().get("rwInboxAuto", "1")) != "0"
        st["pending"] = len(_pending_load())
        st["order"] = rw_cfg().get("rwOrder")
        self._reply_json(200, st)

    def do_GET(self):
        if self.path.startswith("/media_lib"):
            self._handle_media_lib()
        elif self.path.startswith("/media_file"):
            self._handle_media_file()
        elif self.path.rstrip("/") == "/config":
            self._handle_config_get()
        elif self.path.rstrip("/") == "/video/status":
            try:
                self._reply_json(200, {"ffmpeg": bool(_find_ffmpeg()), "ok": True})
            except Exception as e:
                log.warning("视频状态探测失败: %s", e)
                self._reply_json(500, {"error": str(e)})
        elif self.path.rstrip("/") == "/rewrite/pending":
            self._handle_rewrite_pending()
        elif self.path.rstrip("/") == "/rewrite/status":
            self._handle_rewrite_status()
        elif self.path.split("?", 1)[0].rstrip("/") == "/tts/audio":
            self._handle_tts_audio()
        else:
            self._proxy("GET")

    def do_POST(self):
        p = self.path.split("?", 1)[0].rstrip("/")
        if p == "/qwen/tts":
            self._handle_qwen_tts()
        elif p == "/tts":
            self._handle_tts()
        elif p == "/translate":
            self._handle_translate()
        elif p == "/llm":
            self._handle_llm()
        elif p == "/douyin/parse":
            self._handle_douyin_parse()
        elif p == "/douyin/parse_batch":
            self._handle_douyin_parse_batch()
        elif p == "/media/extract":
            self._handle_media_extract()
        elif p == "/save_file":
            self._handle_save_file()
        elif p == "/video/compose":
            self._handle_video_compose()
        elif p.startswith("/feishu/"):
            self._handle_feishu(p)
        elif p == "/config":
            self._handle_config_save()
        elif p == "/rewrite":
            self._handle_rewrite()
        elif p == "/voice":
            self._handle_voice_script()
        elif p == "/rewrite/ack":
            self._handle_rewrite_ack()
        elif p == "/rewrite/warm":
            self._handle_rewrite_warm()
        elif p == "/rewrite/scan":
            self._handle_rewrite_scan()
        else:
            self._proxy("POST")

    def do_OPTIONS(self):
        self.send_response(204)
        self._cors()
        self.send_header("Content-Length", "0")
        self.end_headers()

    def log_message(self, fmt, *args):
        log.info("[HTTP] %s", fmt % args)


class ExclusiveServer(http.server.ThreadingHTTPServer):
    # Windows 默认 allow_reuse_address 会让多个实例同时绑定同一端口互相抢请求；
    # 置 False 后重复启动会直接报 "端口被占用" 并退出，保证 8677 只有一个桥。
    allow_reuse_address = False


if __name__ == "__main__":
    args = sys.argv[1:]
    if "--port" in args:
        PORT = int(args[args.index("--port") + 1])
    if "--upstream" in args:
        UPSTREAM = args[args.index("--upstream") + 1]
    try:
        srv = ExclusiveServer(("127.0.0.1", PORT), Handler)
    except OSError as e:
        log.error("端口 %d 绑定失败（可能已有桥在运行，本实例退出）: %s", PORT, e)
        sys.exit(1)
    log.info("ComfyUI bridge on http://127.0.0.1:%d -> %s (endpoints: /qwen/tts /tts /voice /translate /llm /douyin/parse /save_file /media_lib /media_file /feishu/* /rewrite*)", PORT, UPSTREAM)
    # 启动即自检 Edge-TTS，避免等用户点“生成配音”才发现缺依赖
    try:
        import edge_tts as _edge_tts
        log.info("Edge-TTS 可用：edge-tts %s（解释器 %s）",
                 getattr(_edge_tts, "__version__", "?"), sys.executable)
    except Exception as _e:
        log.warning("Edge-TTS 不可用（配音会失败）：%s —— 修复：\"%s\" -m pip install edge-tts",
                    _e, sys.executable)
    start_inbox_watcher()
    srv.serve_forever()
