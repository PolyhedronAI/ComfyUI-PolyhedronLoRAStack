# -*- coding: ascii -*-
"""Cine storyboard -- a script page -> a Cine Timeline (C7, v1025). Pure:
no ComfyUI, no torch; stdlib only (urllib), so it runs on any machine.

A local LLM writes the board; this module decides nothing about WHICH LLM.
Three ways in, one format out:

  1. ComfyUI itself: a text encoder that can generate (Core master's
     "Generate Text" machinery -- e.g. Qwen3.5 safetensors from
     models/text_encoders) wired as CLIP. ComfyUI loads and offloads it
     like any model. (The node does that call; this module only builds the
     request and reads the answer.)
  2. Any OpenAI-compatible server: llama.cpp llama-server, Ollama, LM
     Studio, vLLM -- on this machine or another one. call_endpoint().
  3. Any other LLM node: `system` + `request` out, its answer into the
     Timeline's `storyboard` input -- parse() reads it there.

THE FORMAT the LLM must answer with is the Timeline's own model
(cine_timeline.py):

  {"seconds": 8, "style": "...",
   "shots": [{"at": 0, "cut": "the camera cuts to", "text": "..."}, ...],
   "anchors": [{"src": "@tag", "at": 0}, ...],
   "soundscape": "...", "music": "N/A"}

parse() is strict the house way: it REPAIRS form (a time written as
"00:02.500" or "2.5s", a cut phrase in other case, shots out of order,
shot 1 not at 0, a tag without @, fences / <think> around the JSON) and
says so in the notes; it REFUSES content by name (no JSON, no shot, a shot
without text, a shot after the clip ends, two shots at one time, an anchor
source that is not on the board, an anchor outside the clip).
"""

import hashlib
import json
import re

from . import cine_timeline as T

# ------------------------------------------------------------------ the ask
CAMERA = ("Zoom In, Zoom Out, Push In, Pull Out, Pan Left, Pan Right, Truck Left, "
          "Truck Right, Tilt Up, Tilt Down, Pedestal Up, Pedestal Down, Arc Shot, "
          "Tracking Shot, Static Shot, Shake Slightly, Shake Strongly, POV, "
          "Roll Clockwise, Roll Counterclockwise; amplitude: with small amplitude / "
          "with large amplitude; speed: at slow speed / at fast speed")

SYSTEM = """You are a storyboard writer for the MiniMax H3 video model. You turn a script page into ONE clip: a few shots on a time axis, the anchor pictures, and the sound. You answer with ONE JSON object and nothing else -- no preamble, no markdown fence, no comment.

The JSON object:
{"seconds": <number>, "style": "<style words>", "shots": [{"at": <seconds>, "cut": "<cut phrase>", "text": "<what happens>"}], "anchors": [{"src": "@<tag>", "at": <seconds>}], "soundscape": "<ambient and action sounds>", "music": "N/A"}

Rules:
- "seconds" is the clip length you are given. Do not change it.
- Shot 1 has "at": 0. Every further shot starts strictly later, and before the clip ends. A cut must bring NEW information (new place, new subject, new view); a small change of distance or angle is a camera movement inside the shot, not a cut. Few shots: one per 2-4 seconds at most.
- "cut" is exactly one of: CUTS. Shot 1's cut is ignored.
- "text" is concrete, observable English prose: who and what is seen, where, the light, the camera, then the ONE main action written as a chain (preparation, trigger, main movement, reaction, end state). Background figures and the environment keep moving ("continue walking throughout the shot").
- The camera uses only this vocabulary, as a natural phrase: CAMERA. Never swoop, whip-pan, dive, orbit, handheld, barrel roll.
- Never negate picture content (write "shadowless soft light", not "no sun"). No weight brackets like (x:1.3). No filler words instead of an instruction (cinematic, epic, 8K).
- Dialogue: <d>[English] the words</d>, the speaker named outside the brackets. Double quotes only for text that is visible in the picture.
- Name the references by their @tag in the text where they appear (for example "@fox runs across the field"). Use only the tags you are given.
- "anchors": a reference picture pinned to a moment. At 0 = the first frame, at the clip's end = the last frame, anything between = a guide. Only picture tags. Leave the list empty when the script does not ask for a pinned picture.
- "soundscape": 1-4 sentences of ambient and action sounds, no dialogue, no music the characters hear.
- "music": only music the characters do NOT hear, as instruments, tempo and dynamics; "N/A" when none is wanted.
- "style": short style words, ending with a comma, for example "Live-action, cinematic,".
""".replace("CUTS", ", ".join('"%s"' % c for c in T.CUTS)).replace("CAMERA", CAMERA)


def refs_catalog(refs):
    """The board's references as the LLM sees them: [(tag, kind, role,
    desc)] -- tags without @, in board order. Pictures first."""
    out = []
    for kind, key in (("picture", "images"), ("video", "videos"), ("audio", "audios")):
        for it in (refs or {}).get(key, []) or []:
            if it.get("tag"):
                out.append((str(it["tag"]), kind, str(it.get("role") or ""),
                            " ".join(str(it.get("desc") or "").split())))
    return out


def build_request(script, seconds, style, catalog):
    """-> the user message: length, style, the references, the script."""
    frames = T.frames_for(seconds)
    dur = frames / float(T.FPS)
    lines = ["Clip length: %.2f seconds (the clip ends at %.3f s; the last frame is at %.3f s)."
             % (dur, dur, (frames - 1) / float(T.FPS))]
    if str(style or "").strip():
        lines.append("Style: %s" % str(style).strip())
    if catalog:
        lines.append("References (use these @tags; only pictures can be anchors):")
        for tag, kind, role, desc in catalog:
            lines.append("  @%s -- %s%s%s" % (tag, kind, (", " + role) if role else "",
                                             (": " + desc) if desc else ""))
    else:
        lines.append("References: none (write no @tags, leave anchors empty).")
    lines.append("")
    lines.append("Script page:")
    lines.append(str(script or "").strip())
    lines.append("")
    lines.append("Answer with the JSON object only.")
    return "\n".join(lines)


# ------------------------------------------------------------------ the answer
def extract_json(text):
    """The first JSON object in an LLM answer (reasoning, fences and talk
    around it are dropped). Raises ValueError by name."""
    s = str(text or "")
    if "</think>" in s.lower():                  # reasoning first (opened here or in the template)
        s = s[s.lower().rindex("</think>") + len("</think>"):]
    dec = json.JSONDecoder()
    i = s.find("{")
    while i != -1:
        try:
            obj, _ = dec.raw_decode(s, i)
            if isinstance(obj, dict):
                return obj
        except ValueError:
            pass
        i = s.find("{", i + 1)
    head = " ".join(s.split())[:160]
    raise ValueError("the LLM answer holds no JSON object (it begins: %r)" % head)


_MMSS = re.compile(r"^\s*(?:(\d+):)?(\d+(?:\.\d+)?)\s*(?:s|sec|seconds)?\s*$", re.I)


def to_seconds(v, end=None):
    """0 / 2.5 / "2.5" / "2.5s" / "00:02.500" / "first" / "last" -> seconds.
    None when it is none of these."""
    if isinstance(v, bool):
        return None
    if isinstance(v, (int, float)):
        return float(v)
    s = str(v or "").strip().lower()
    if s in ("first", "start", "begin"):
        return 0.0
    if s in ("last", "end") and end is not None:
        return float(end)
    m = _MMSS.match(s)
    if not m:
        return None
    return int(m.group(1) or 0) * 60 + float(m.group(2))


def _snap(sec):
    return round(float(sec) * T.FPS) / float(T.FPS)


def _cut(phrase):
    p = " ".join(str(phrase or "").split()).lower().rstrip(",")
    for c in T.CUTS:
        if p == c:
            return c
    return None


def parse(text, sources=None, seconds=None, style=None, tags=None, fallback_seconds=None):
    """An LLM answer (or a Timeline model as JSON) -> (model, notes).
    sources: the anchor sources that exist (with @), None = do not check.
    seconds / style: what was asked -- they win over the answer's.
    fallback_seconds: the length when neither the ask nor the answer has one.
    tags: every @tag the board has (pictures, videos, audios), for the note
    on unknown tags in the text; None = the picture sources.
    Raises ValueError listing every refusal by name."""
    d = extract_json(text)
    notes, errs = [], []
    m = json.loads(json.dumps(T.DEFAULT))
    # length: the asked one wins
    got = to_seconds(d.get("seconds")) if "seconds" in d else None
    if seconds is not None:
        m["seconds"] = float(seconds)
        if got is not None and abs(got - float(seconds)) > 1e-6:
            notes.append("the answer said %.2f s; the asked %.2f s is kept" % (got, float(seconds)))
    elif got is not None and got > 0:
        m["seconds"] = got
    elif fallback_seconds is not None:
        m["seconds"] = float(fallback_seconds)
        notes.append("the answer has no length -- the timeline's %.2f s is kept" % m["seconds"])
    else:
        errs.append("the answer has no length ('seconds') and none was asked")
        raise ValueError("; ".join(errs))
    frames = T.frames_for(m["seconds"])
    dur = frames / float(T.FPS)
    last_t = (frames - 1) / float(T.FPS)
    # style
    if style is not None and str(style).strip():
        m["style"] = str(style).strip()
    elif "style" in d:                           # the answer's, even an empty one
        m["style"] = " ".join(str(d["style"] or "").split())
    # shots
    raw = d.get("shots")
    if not isinstance(raw, list) or not raw:
        errs.append("the answer has no shots")
        raw = []
    shots = []
    for i, s in enumerate(raw, start=1):
        if isinstance(s, str):
            s = {"text": s, "at": 0 if i == 1 else None}
        if not isinstance(s, dict):
            errs.append("shot %d is not an object" % i)
            continue
        text_ = " ".join(str(s.get("text", "") or "").split())
        if not text_:
            errs.append("shot %d has no text" % i)
        at = to_seconds(s.get("at", s.get("time", s.get("start"))))
        if at is None:
            errs.append("shot %d: time %r is not a time" % (i, s.get("at", s.get("time"))))
            continue
        c = _cut(s.get("cut", T.CUTS[0]))
        if c is None:
            notes.append("shot %d: cut %r is not an official phrase -> %r" % (i, s.get("cut"), T.CUTS[0]))
            c = T.CUTS[0]
        shots.append({"at": at, "cut": c, "text": text_, "_n": i})
    if shots:
        order = [x["_n"] for x in shots]
        shots.sort(key=lambda x: x["at"])
        if [x["_n"] for x in shots] != order:
            notes.append("shots were not in time order -- sorted")
        if shots[0]["at"] != 0:
            notes.append("shot 1 started at %.3f s -- moved to 0 ([Shot 1] carries no time)" % shots[0]["at"])
            shots[0]["at"] = 0.0
        for x in shots:
            y = _snap(x["at"])
            x["at"] = y
        for k in range(1, len(shots)):
            if shots[k]["at"] >= dur:
                errs.append("shot %d at %.3f s starts after the clip ends (%.3f s)"
                            % (shots[k]["_n"], shots[k]["at"], dur))
            if shots[k]["at"] <= shots[k - 1]["at"]:
                errs.append("shots %d and %d start at the same time (%.3f s)"
                            % (shots[k - 1]["_n"], shots[k]["_n"], shots[k]["at"]))
    m["shots"] = [{"at": x["at"], "cut": x["cut"], "text": x["text"]} for x in shots]
    # anchors
    anchors = []
    names = set(sources) if sources is not None else None
    for i, a in enumerate(d.get("anchors") or [], start=1):
        if not isinstance(a, dict):
            errs.append("anchor %d is not an object" % i)
            continue
        src = str(a.get("src", a.get("tag", "")) or "").strip()
        if src and not src.startswith("@") and src not in T.IMAGE_SLOTS:
            if names is None or ("@" + src) in names:
                src = "@" + src
        if not src:
            errs.append("anchor %d has no source" % i)
            continue
        if names is not None and src not in names:
            errs.append("anchor %d: %s is not a picture on the board" % (i, src))
            continue
        at = to_seconds(a.get("at", a.get("time")), end=last_t)
        if at is None:
            errs.append("anchor %d: time %r is not a time" % (i, a.get("at")))
            continue
        if at > last_t and at <= dur + 1e-9:
            notes.append("anchor %d at %.3f s = the clip's end -> the last frame (%.3f s)" % (i, at, last_t))
            at = last_t
        if at < 0 or at > dur + 1e-9:
            errs.append("anchor %d at %.3f s is outside the clip (0 .. %.3f s)" % (i, at, last_t))
            continue
        anchors.append({"src": src, "at": _snap(at)})
    m["anchors"] = anchors
    m["soundscape"] = " ".join(str(d.get("soundscape", "") or "").split())
    music = " ".join(str(d.get("music", "") or "").split())
    m["music"] = music if music else "N/A"
    if not errs:
        errs = T.validate(m, names)
    if errs:
        raise ValueError("; ".join(errs))
    # tags in the text that the board does not have: a note, not a refusal
    known = set(tags) if tags is not None else ({n for n in names if n.startswith("@")} if names is not None else None)
    if known is not None:
        for i, s in enumerate(m["shots"], start=1):
            for t in sorted(set(re.findall(r"@[A-Za-z_][A-Za-z0-9_]*", s["text"]))):
                if t not in known:
                    notes.append("shot %d names %s, which the board does not have" % (i, t))
    return m, notes


def digest(text):
    """The fingerprint the Timeline keeps of the storyboard it adopted."""
    return hashlib.sha1(str(text or "").strip().encode("utf-8")).hexdigest()[:16]


# ------------------------------------------------------------------ the server
def chat_url(endpoint):
    """http://host:8080 | .../v1 | .../v1/chat/completions -> the chat URL."""
    u = str(endpoint or "").strip().rstrip("/")
    if not u:
        return ""
    if not re.match(r"^https?://", u, re.I):
        u = "http://" + u
    if u.endswith("/chat/completions"):
        return u
    if re.search(r"/v\d+$", u):
        return u + "/chat/completions"
    return u + "/v1/chat/completions"


def request_body(model, system, user, max_tokens, temperature, seed):
    body = {
        "model": str(model or "").strip() or "local",
        "messages": [{"role": "system", "content": system}, {"role": "user", "content": user}],
        "max_tokens": int(max_tokens),
        "temperature": float(temperature),
        "seed": int(seed),
        "stream": False,
        # Qwen3.x: think off for writing (llama.cpp / vLLM honour it, others ignore it)
        "chat_template_kwargs": {"enable_thinking": False},
    }
    return body


def call_endpoint(endpoint, model, system, user, max_tokens=2048, temperature=0.4, seed=0,
                  timeout=600, opener=None):
    """POST to an OpenAI-compatible server -> the answer text. Raises
    ValueError by name (no server / HTTP error / empty answer)."""
    import urllib.error
    import urllib.request
    url = chat_url(endpoint)
    if not url:
        raise ValueError("no endpoint given")
    data = json.dumps(request_body(model, system, user, max_tokens, temperature, seed)).encode("utf-8")
    req = urllib.request.Request(url, data=data, headers={"Content-Type": "application/json"})
    op = opener or urllib.request.urlopen
    try:
        with op(req, timeout=timeout) as r:
            out = json.loads(r.read().decode("utf-8", "replace"))
    except urllib.error.HTTPError as e:
        try:
            detail = e.read().decode("utf-8", "replace")[:300]
        except Exception:
            detail = ""
        raise ValueError("the LLM server at %s answered HTTP %d: %s" % (url, e.code, " ".join(detail.split())))
    except (urllib.error.URLError, OSError) as e:
        raise ValueError("no LLM server answers at %s (%s) -- start llama-server / Ollama / "
                         "LM Studio there, or wire a text encoder into `clip`"
                         % (url, getattr(e, "reason", e)))
    except ValueError as e:
        raise ValueError("the LLM server at %s did not answer JSON (%s)" % (url, e))
    try:
        msg = out["choices"][0]["message"]
        text = msg.get("content") or ""
    except Exception:
        raise ValueError("the LLM server at %s answered without choices[0].message: %s"
                         % (url, json.dumps(out)[:200]))
    if not str(text).strip():
        why = out["choices"][0].get("finish_reason")
        raise ValueError("the LLM server at %s answered empty (finish_reason %r%s)"
                         % (url, why, "; it only reasoned -- raise max_tokens or turn thinking off"
                            if msg.get("reasoning_content") else ""))
    return str(text)
