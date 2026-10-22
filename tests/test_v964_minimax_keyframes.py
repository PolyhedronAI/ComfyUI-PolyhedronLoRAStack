"""v964 -- ULSMiniMaxKeyframes: fl2va (first / last frame) on our rails.

  A  pure helpers, lifted with ast (no torch, no comfy): keyframe_index,
     fit_modes, aspect_off -- the ONLY places the anchor rule, the fit rule
     and the report rule live.
  B  static pins: registration, both baselines, the declared limit (no
     `positive` input -- keyframes and refs cannot share one conditioning on
     Core 7a131a3a), and the three Core coupling names -- read from the
     actual CALL and the actual dict keys, then re-checked in Core's own
     source when Core can be found.
  C  build() DRIVEN against stubs: the real module executed with comfy.utils,
     node_helpers and uls_latent_math stood in (no ComfyUI needed, torch
     needed -- it fails LOUD without torch, never vacuum-green). Checks the
     payload Core will read: indices, order, keys, crops, the plain-text run,
     the honest refusal of a flat latent, the aspect report, the latent
     pass-through.
  D  mutations: each one must be caught by C.
"""
import ast
import os
import pathlib
import sys
import types

ROOT = pathlib.Path(__file__).resolve().parent.parent
SRC = ROOT / "nodes" / "ph_minimax_keyframes.py"
FAILED = []


def _fail(m):
    FAILED.append(m)
    print("FAIL: {}".format(m))


def _ok(m):
    print("ok  : {}".format(m))


def _lift(src, names, extra_assigns=()):
    tree = ast.parse(src)
    ns, got = {}, {}
    for node in tree.body:
        if isinstance(node, ast.Assign):
            tgt = node.targets[0].id if isinstance(node.targets[0], ast.Name) else ""
            if extra_assigns and tgt.startswith(extra_assigns):
                exec(compile(ast.Module([node], []), "<v964>", "exec"), ns)
        if isinstance(node, ast.FunctionDef) and node.name in names:
            got[node.name] = node
    for name in names:
        if name in got:
            exec(compile(ast.Module([got[name]], []), "<v964>", "exec"), ns)
    return ns


src = SRC.read_text(encoding="utf-8")

# =====================================================================  A
ns = _lift(src, ("keyframe_index", "fit_modes", "aspect_off"),
           ("FRAME_FIT", "ASPECT_WARN"))
ki, fm, ao = ns["keyframe_index"], ns["fit_modes"], ns["aspect_off"]
FF = ns["FRAME_FIT"]

if ki("first", 73) == 0 and ki("last", 73) == 72 and ki("last", 5) == 4:
    _ok("A keyframe_index: first -> 0, last -> frame_count - 1")
else:
    _fail("A keyframe_index rule broken: %r %r %r"
          % (ki("first", 73), ki("last", 73), ki("last", 5)))
for bad in (("last", 0), ("middle", 73)):
    try:
        ki(*bad)
        _fail("A keyframe_index accepted %r" % (bad,))
    except ValueError:
        _ok("A keyframe_index refuses %r" % (bad,))

if fm(FF[0]) == ("disabled", "center") and fm(FF[1]) == ("disabled", "disabled") \
        and fm(FF[2]) == ("center", "center") and fm("garbage") == ("disabled", "center"):
    _ok("A fit_modes: core = stretch first / cover last; both variants; unknown -> core")
else:
    _fail("A fit_modes broken: %r %r %r" % (fm(FF[0]), fm(FF[1]), fm(FF[2])))

if abs(ao(1344, 768, 1344, 768)) < 1e-9 and 0.01 < ao(1280, 720, 1344, 768) < ns["ASPECT_WARN"] \
        and 0.20 < ao(1024, 768, 1344, 768) < 0.26 \
        and ao(0, 0, 1344, 768) == 0.0 and ao(100, 50, 0, 0) == 0.0:
    _ok("A aspect_off: 0 on the canvas itself, 1.6 % for 1280x720 (under the line), "
        "~24 % for 4:3 (over it), 0 on degenerate")
else:
    _fail("A aspect_off broken: %r %r" % (ao(1280, 720, 1344, 768), ao(1024, 768, 1344, 768)))
if 0.0 < ns["ASPECT_WARN"] < 0.1:
    _ok("A ASPECT_WARN is a small fraction (%.3f)" % ns["ASPECT_WARN"])
else:
    _fail("A ASPECT_WARN out of range: %r" % ns["ASPECT_WARN"])

# =====================================================================  B
init = (ROOT / "__init__.py").read_text(encoding="utf-8")
if "from .nodes.ph_minimax_keyframes import ULSMiniMaxKeyframes" in init \
        and 'NODE_CLASS_MAPPINGS["ULSMiniMaxKeyframes"] = ULSMiniMaxKeyframes' in init \
        and 'NODE_DISPLAY_NAME_MAPPINGS["ULSMiniMaxKeyframes"]' in init:
    _ok("B registered in __init__ (guarded import + both mappings)")
else:
    _fail("B registration missing in __init__")

import glob
for pat in ("NODE_IDS_baseline_v*.txt", "WIDGET_ORDER_baseline_v*.txt"):
    hits = sorted(glob.glob(str(ROOT / pat)))
    if len(hits) != 1:
        _fail("B %d baselines for %s (must be exactly one)" % (len(hits), pat))
        continue
    text = pathlib.Path(hits[-1]).read_text(encoding="utf-8")
    if pat.startswith("NODE_IDS"):
        good = "\nULSMiniMaxKeyframes\n" in text
    else:
        # RE-GROUNDED v1017: the promise is that the node's widget order
        # STARTS with prompt,frame_fit (the #577 prefix law) -- v1017 appended
        # `tags` behind it. Pinning the whole line pinned "nothing may ever be
        # appended", which is the opposite of the law.
        good = "\nULSMiniMaxKeyframes\tprompt,frame_fit\n" in text or \
            "\nULSMiniMaxKeyframes\tprompt,frame_fit," in text
    (_ok if good else _fail)("B %s carries the node (%s)" % (os.path.basename(hits[-1]),
                                                             "yes" if good else "NO"))

tree = ast.parse(src)
cls = [n for n in tree.body if isinstance(n, ast.ClassDef) and n.name == "ULSMiniMaxKeyframes"]
if not cls:
    _fail("B class ULSMiniMaxKeyframes not found")
else:
    csrc = ast.get_source_segment(src, cls[0])
    if '"positive": ("CONDITIONING"' in csrc:
        _fail("B the node grew a `positive` input -- on Core 7a131a3a keyframes and "
              "refs cannot share one conditioning (extra_conds overwrites "
              "cond_video_latents); read the module docstring before appending")
    else:
        _ok("B no `positive` input -- the declared limit holds")
    if 'RETURN_NAMES = ("positive", "latent", "info", "prompt_text")' in csrc \
            and 'RETURN_TYPES = ("CONDITIONING", "LATENT", "STRING", "STRING")' in csrc:
        _ok("B v965 prompt_text is the fourth output -- appended, slots 0-2 unchanged")
    else:
        _fail("B output slots drifted (prompt_text must be APPENDED as slot 3)")
    if csrc.count('"IMAGE"') == 2 and '"first_frame"' in csrc and '"last_frame"' in csrc:
        _ok("B exactly two IMAGE sockets: first_frame, last_frame")
    else:
        _fail("B image sockets drifted")

# the coupling, read from the CALL, not from a comment
tok_kw, enc_calls, set_keys, idx_key = set(), set(), set(), False
for node in ast.walk(tree):
    if isinstance(node, ast.Call):
        f = node.func
        name = f.attr if isinstance(f, ast.Attribute) else getattr(f, "id", "")
        if name == "tokenize":
            tok_kw |= {k.arg for k in node.keywords}
        if name == "encode_from_tokens_scheduled":
            enc_calls.add(name)
        if name == "conditioning_set_values" and len(node.args) == 2 \
                and isinstance(node.args[1], ast.Dict):
            for k in node.args[1].keys:
                if isinstance(k, ast.Constant):
                    set_keys.add(k.value)
    if isinstance(node, ast.Constant) and node.value == "resolved_frame_index":
        idx_key = True
if tok_kw == {"images"}:
    _ok("B tokenize(prompt, images=...) -- Core's fl2va presentation")
else:
    _fail("B tokenize keywords drifted: %r (Core's fl2va uses images=)" % sorted(tok_kw))
if enc_calls:
    _ok("B encode_from_tokens_scheduled is the encode call")
else:
    _fail("B encode_from_tokens_scheduled missing")
if set_keys == {"minimax_keyframes", "minimax_frame_count"}:
    _ok("B conditioning_set_values carries exactly minimax_keyframes + minimax_frame_count")
else:
    _fail("B conditioning keys drifted: %r" % sorted(set_keys))
if idx_key:
    _ok("B resolved_frame_index is the anchor key")
else:
    _fail("B resolved_frame_index missing")

core = None
for up in ROOT.parents:
    cand = up / "comfy_extras" / "nodes_minimax_h3.py"
    if cand.exists():
        core = cand
        break
if core is None:
    print("note: Core's nodes_minimax_h3.py not found from here -- the upstream "
          "half of the coupling pin is SKIPPED (it fires on Frank's machine).")
else:
    core_src = core.read_text(encoding="utf-8", errors="replace")
    drifted = [n for n in ("minimax_keyframes", "resolved_frame_index", "images=images")
               if n not in core_src]
    if drifted:
        _fail("Core no longer uses %r -- our keyframes would ride into a run with NO "
              "anchor and say nothing. Re-read comfy_extras/nodes_minimax_h3.py." % drifted)
    else:
        _ok("B the three load-bearing coupling names still exist in Core's own source")
    # minimax_frame_count: read by Core 7a131a3a (extra_conds), DROPPED by Core
    # master (>= e01fb4c5), which derives the count from the latent. Our key is
    # then an unused cond entry -- harmless, measured 19.09.2026 -- so this is a
    # note, not a verdict.
    if "minimax_frame_count" in core_src:
        _ok("B Core reads minimax_frame_count (7a131a3a rule)")
    else:
        print("note: this Core derives the frame count from the latent and ignores "
              "minimax_frame_count -- our key rides along unused (Core master rule).")
    model = core.parent.parent / "comfy" / "ldm" / "minimax" / "model.py"
    if model.exists():
        msrc = model.read_text(encoding="utf-8", errors="replace")
        if "only first/last keyframe anchors are supported" in msrc:
            print("note: this Core pins keyframes to first/last (7a131a3a rule) -- "
                  "exactly what this node offers.")
        else:
            print("note: this Core no longer carries the first/last-only raise -- "
                  "interior anchors may be available (MiniMaxH3AddGuide); the "
                  "guide node is a separate cut.")

# =====================================================================  C
try:
    import torch
except Exception as exc:  # loud, never vacuum-green
    _fail("C needs torch to drive build() (%r) -- pip install torch" % (exc,))
    torch = None


class _Clip:
    def __init__(self):
        self.calls = []

    def tokenize(self, prompt, **kw):
        self.calls.append(("tokenize", prompt, kw))
        return {"tok": prompt, "n_images": len(kw.get("images", []))}

    def encode_from_tokens_scheduled(self, tokens):
        self.calls.append(("encode", tokens))
        return [[torch.zeros(1, 4, 8), {"tok": tokens["tok"]}]]


class _Vae:
    def __init__(self):
        self.seen = []

    def encode(self, img):
        self.seen.append(tuple(img.shape))
        return torch.full((1, 24, 1, img.shape[1] // 16, img.shape[2] // 16),
                          float(img.shape[1] * 1000 + img.shape[2]))


class _Nested:
    is_nested = True

    def __init__(self, parts):
        self._p = parts

    def unbind(self):
        return list(self._p)


def _latent(w=1344, h=768, latent_t=22):
    v = torch.zeros(1, 24, latent_t, h // 16, w // 16)
    a = torch.zeros(1, 32, 2, 122)
    return {"samples": _Nested((v, a))}


def _stub_modules():
    """comfy.utils / node_helpers stand-ins with the semantics the node relies on."""
    comfy = types.ModuleType("comfy")
    cu = types.ModuleType("comfy.utils")

    def common_upscale(samples, width, height, method, crop):
        # [B, C, H, W] -> [B, C, height, width]; the crop keyword is recorded on
        # the tensor so the drive can read which fit was used.
        out = torch.nn.functional.interpolate(samples, size=(height, width), mode="bilinear",
                                              align_corners=False)
        return out
    cu.common_upscale = common_upscale
    comfy.utils = cu
    nh = types.ModuleType("node_helpers")

    def conditioning_set_values(cond, values):
        out = []
        for t, d in cond:
            n = dict(d)
            n.update(values)
            out.append([t, n])
        return out
    nh.conditioning_set_values = conditioning_set_values
    return {"comfy": comfy, "comfy.utils": cu, "node_helpers": nh}


def _load(module_src):
    """Execute the node module with stubs; returns the class."""
    saved = {k: sys.modules.get(k) for k in ("comfy", "comfy.utils", "node_helpers")}
    sys.modules.update(_stub_modules())
    if str(ROOT / "nodes") not in sys.path:
        sys.path.insert(0, str(ROOT / "nodes"))
    import uls_latent_math as M  # the real one -- it is torch-free by design
    ns = {"__name__": "pls_v964_drive"}
    body = module_src.replace("from . import uls_latent_math as M", "M = __M")
    ns["__M"] = M
    # v1017: the node imports the prompt compiler too -- pure, torch-free,
    # handed in the same way (a second relative import the exec cannot follow).
    import h3_prompt as HP
    body = body.replace("from . import h3_prompt as HP", "HP = __HP")
    ns["__HP"] = HP
    try:
        exec(compile(body, str(SRC), "exec"), ns)
    finally:
        for k, v in saved.items():
            if v is None:
                sys.modules.pop(k, None)
            else:
                sys.modules[k] = v
    return ns["ULSMiniMaxKeyframes"]


def _img(h, w):
    return torch.rand(1, h, w, 3)


def _res(out):
    """v965: build() returns {"ui": {"pls_kf": [...]}, "result": (...)}."""
    return out["result"] if isinstance(out, dict) else out


def drive(module_src):
    """Every observable promise, as a dict of booleans."""
    r = {}
    K = _load(module_src)()
    FFm = FF
    # 1 core mode, both frames
    clip, vae = _Clip(), _Vae()
    out = K.build(clip, vae, _latent(), "p", FFm[0],
                  first_frame=_img(720, 1280), last_frame=_img(768, 1024))
    cond, lat, info, ptext = _res(out)
    # v965: prompt_text is the 4th output (APPENDED) and the ui payload carries
    # the prompt that ran plus the report, for the in-node field
    r["prompt_text"] = ptext == "p" and isinstance(out, dict) \
        and out["ui"]["pls_kf"][0]["prompt"] == "p" and out["ui"]["pls_kf"][0]["info"] == info \
        and out["ui"]["pls_kf"][0]["keyframes"] == 2
    kfs = cond[0][1].get("minimax_keyframes", [])
    r["idx"] = [k["resolved_frame_index"] for k in kfs] == [0, 72]
    r["count"] = cond[0][1].get("minimax_frame_count") == 73
    r["order"] = clip.calls[0][2].get("images") is not None and len(clip.calls[0][2]["images"]) == 2 \
        and clip.calls[0][0] == "tokenize" and clip.calls[1][0] == "encode"
    r["latents"] = all(k["latent"] is not None and "image" not in k for k in kfs)
    r["shape"] = vae.seen == [(1, 768, 1344, 3), (1, 768, 1344, 3)]
    r["aspect_line"] = ("ASPECT OFF" in info.splitlines()[2]) and ("ASPECT OFF" not in info.splitlines()[1])
    r["pass"] = lat is not None and lat["samples"] is not None
    # 2 plain text run
    clip, vae = _Clip(), _Vae()
    cond, _, info, _ = _res(K.build(clip, vae, _latent(), "p", FFm[0]))
    r["plain"] = "minimax_keyframes" not in cond[0][1] and "minimax_frame_count" not in cond[0][1] \
        and clip.calls[0][2].get("images") == [] and "0 keyframe" in info
    # 3 last only -> <Picture 1> is the last frame, pinned at 72
    clip, vae = _Clip(), _Vae()
    cond, _, info, _ = _res(K.build(clip, vae, _latent(), "p", FFm[0], last_frame=_img(768, 1344)))
    kfs = cond[0][1].get("minimax_keyframes", [])
    r["last_only"] = [k["resolved_frame_index"] for k in kfs] == [72] and "<Picture 1>  last" in info
    # 4 the fit rule reaches the upscale call: record crops through a spy
    seen_crops = []
    K2 = _load(module_src)()

    def spy_upscale(samples, width, height, method, crop):
        seen_crops.append(crop)
        return torch.nn.functional.interpolate(samples, size=(height, width), mode="bilinear",
                                               align_corners=False)
    # the module captured its own comfy stub at exec time -- patch THAT one
    cu = K2.build.__globals__["comfy"].utils
    cu.common_upscale = spy_upscale
    K2.build(_Clip(), _Vae(), _latent(), "p", FFm[0], first_frame=_img(768, 1344), last_frame=_img(768, 1344))
    K2.build(_Clip(), _Vae(), _latent(), "p", FFm[2], first_frame=_img(768, 1344), last_frame=_img(768, 1344))
    r["crops"] = seen_crops == ["disabled", "center", "center", "center"]
    # 5 a flat latent is refused with our prose
    try:
        K.build(_Clip(), _Vae(), {"samples": torch.zeros(1, 16, 5, 48, 84)}, "p", FFm[0])
        r["refuse"] = False
    except ValueError as exc:
        r["refuse"] = "MiniMax H3 AV" in str(exc)
    # 6 a shorter latent moves the last anchor with it
    clip = _Clip()
    cond, _, _, _ = _res(K.build(clip, _Vae(), _latent(latent_t=7), "p", FFm[0], last_frame=_img(768, 1344)))
    r["short"] = cond[0][1]["minimax_keyframes"][0]["resolved_frame_index"] == 21 \
        and cond[0][1]["minimax_frame_count"] == 22
    return r


if torch is not None:
    res = drive(src)
    for k, v in res.items():
        (_ok if v else _fail)("C " + k)

    # =================================================================  D
    MUTATIONS = [
        ("M1 the last frame is pinned one past the end",
         "        return fc - 1\n", "        return fc\n", "idx"),
        ("M2 core mode cover-crops the first frame too",
         '    return ("disabled", "center")\n', '    return ("center", "center")\n', "crops"),
        ("M3 the frame count leaves the payload",
         '                "minimax_frame_count": frame_count,\n', "", "count"),
        ("M4 the last-only run keeps the first slot",
         "        for which, img, crop in ((\"first\", first_frame, first_crop),\n"
         "                                 (\"last\", last_frame, last_crop)):\n"
         "            if img is None:\n                continue\n",
         "        for which, img, crop in ((\"first\", first_frame, first_crop),\n"
         "                                 (\"last\", last_frame, last_crop)):\n"
         "            if img is None:\n                img = first_frame if first_frame is not None else last_frame\n",
         "last_only"),
        ("M5 the aspect report never fires",
         "            if off > ASPECT_WARN:\n", "            if off > 10.0:\n", "aspect_line"),
    ]
    caught = 0
    for name, a, b, key in MUTATIONS:
        if src.count(a) != 1:
            _fail("D mutation anchor broken: " + name)
            continue
        try:
            r = drive(src.replace(a, b, 1))
            hit = not r.get(key, False)
        except Exception:
            hit = True
        caught += hit
        print("  %s %s" % ("caught:" if hit else "MISSED:", name))
        if not hit:
            _fail("D mutation survived: " + name)
    print("  mutations caught %d/%d" % (caught, len(MUTATIONS)))

if FAILED:
    print("\n[test_v964_minimax_keyframes] FAIL -- %d" % len(FAILED))
    sys.exit(1)
print("\n[test_v964_minimax_keyframes] PASS -- A pure rules, B pins + Core coupling, "
      "C driven build, D mutations")
