# -*- coding: ascii -*-
"""Guard v1049 -- scene continuity ("a connection from clip 1 to 2", Frank
28.09.; "build everything sensible", 01.10.) and the second pass.

  C1  the model: shots[i].carry on a scene start, carry_mode pin|keyframe
      (validated, in DEFAULT on both sides); carry_of only for scenes >= 2.
  C2  the node, with a REAL clip (ffmpeg): a carrying scene k hands out the
      LAST frame of scene k-1's newest clip as first_frame (the frame's own
      colour, not the first one's); "pin" leaves the prompt alone, "keyframe"
      puts the official alignment line in and makes it <Picture 1>; an own
      first-frame anchor wins; no clip of k-1 = a free run, said by name;
      IS_CHANGED follows the newest clip of k-1 (a new take re-runs).
  C3  MiniMax Reference: an appended first_frame SOCKET (the widget order is
      untouched); next to references it is a keyframe marked for the merge.
  C4  the merge (h3_kf_refs): only marked keyframes, cond_video_latents =
      keyframe latents + reference latents (PackedLayout's order), frame
      count kept; unmarked = untouched; installed once, never stacked on the
      Director's patch (then its marker key is used). Core's overwrite -- the
      reason for the merge -- is pinned in Core's source when Core is found.
  C5  H3 Latent Upscale: the optional positive comes out WITHOUT keyframes
      (a first-pass keyframe does not fit the new size; the frame is in the
      latent already); references stay.
  C6  the timeline UI: the carry switch and mode in the scene row and the
      shot editor; the scene bar marks a carrying scene.
  S   ASCII.

Script-style: exit 0 = pass.
"""
import importlib
import importlib.util
import json
import os
import shutil
import subprocess
import sys
import tempfile
import types
from pathlib import Path

import torch

ROOT = Path(__file__).resolve().parents[1]
FAILS = []


def _need(c, m):
    if not c:
        FAILS.append(m)
    print(("  ok    " if c else "  FAIL  ") + m)


def _load(name, rel):
    spec = importlib.util.spec_from_file_location(name, str(ROOT / rel))
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


T = _load("ct1049", "nodes/cine_timeline.py")
js = (ROOT / "web" / "js" / "ph_cine_timeline.js").read_text(encoding="utf-8")


def model(**kw):
    m = json.loads(json.dumps(T.DEFAULT))
    m.update({"seconds": 9.0, "project": "carryp", "shots": [
        {"at": 0.0, "cut": T.CUTS[0], "text": "a"},
        {"at": 3.0, "cut": T.CUTS[0], "text": "b", "scene": True, "carry": True},
        {"at": 6.0, "cut": T.CUTS[0], "text": "c", "scene": True}]})
    m.update(kw)
    return m


# ------------------------------------------------------------------ C1
_need(T.DEFAULT.get("carry_mode") == "pin" and 'carry_mode: "pin"' in js, "C1 carry_mode in DEFAULT, both sides")
m = model()
_need([T.carry_of(m, k) for k in (0, 1, 2, 3)] == [False, False, True, False], "C1 carry_of: scene 2 only")
_need(any("carry_mode" in e for e in T.validate(model(carry_mode="x"))) and not T.validate(m), "C1 carry_mode validated")

# ------------------------------------------------------------------ C2
ff = shutil.which("ffmpeg")
if not ff:
    print("  note  ffmpeg not found -- C2/C5 SKIPPED")
else:
    TMP = Path(tempfile.mkdtemp(prefix="v1049_"))
    for d in ("output/carryp", "input", "temp"):
        (TMP / d).mkdir(parents=True, exist_ok=True)

    class _NT:
        is_nested = True

        def __init__(self, parts):
            self.parts = tuple(parts)

        def unbind(self):
            return list(self.parts)

    comfy = types.ModuleType("comfy")
    nt = types.ModuleType("comfy.nested_tensor")
    nt.NestedTensor = _NT
    comfy.nested_tensor = nt
    mm = types.ModuleType("comfy.model_management")
    mm.unload_all_models = lambda: None
    mm.soft_empty_cache = lambda force=False: None
    comfy.model_management = mm
    capi = types.ModuleType("comfy_api")
    ii = types.ModuleType("comfy_api.input_impl")
    ii.VideoFromComponents = object
    ut = types.ModuleType("comfy_api.util")
    ut.VideoComponents = object
    fp = types.ModuleType("folder_paths")
    fp.get_output_directory = lambda: str(TMP / "output")
    fp.get_input_directory = lambda: str(TMP / "input")
    fp.get_temp_directory = lambda: str(TMP / "temp")
    fp.models_dir = str(TMP)
    fp.folder_names_and_paths = {}
    fp.add_model_folder_path = lambda n, p: None
    fp.get_filename_list = lambda n: []
    sys.modules.update({"comfy": comfy, "comfy.nested_tensor": nt, "comfy.model_management": mm, "comfy_api": capi,
                        "comfy_api.input_impl": ii, "comfy_api.util": ut, "folder_paths": fp})
    pkg = types.ModuleType("plsn1049")
    pkg.__path__ = [str(ROOT / "nodes")]
    sys.modules["plsn1049"] = pkg
    TL = importlib.import_module("plsn1049.ph_cine_timeline")
    T2 = sys.modules["plsn1049.cine_timeline"]

    def clip(name, colour_last, n=48):
        # n frames: red first, `colour_last` in the last quarter
        f = TMP / "output" / "carryp" / name
        q = n // 4
        subprocess.run([ff, "-v", "error", "-y", "-f", "lavfi", "-i", "color=c=red:s=64x36:r=24:d=%f" % ((n - q) / 24.0),
                        "-f", "lavfi", "-i", "color=c=%s:s=64x36:r=24:d=%f" % (colour_last, q / 24.0),
                        "-filter_complex", "[0][1]concat=n=2:v=1:a=0", "-c:v", "libx264", "-pix_fmt", "yuv420p", str(f)],
                       capture_output=True)
        return f

    node = TL.ULSCineTimeline()
    m2 = model(render="scene", active=2)
    out = node.build(json.dumps(m2))
    r = out["result"]
    # AMENDED v1152: the stack is no longer a folder under output/ -- the reason names the project's stack
    _need(r[1] is None and "has no clip in the stack of carryp yet" in r[6] and "-- this run starts free" in r[6], "C2 (v1084 DECLARED: the reason sits between) no clip of scene 1 yet: a free run, said by name")
    _need(TL.ULSCineTimeline.IS_CHANGED(json.dumps(m2)) == "none", "C2 IS_CHANGED: 'none' while there is no clip")
    c1 = clip("scene_1_00001_.mp4", "blue")
    import time
    os.utime(c1, (time.time() - 50, time.time() - 50))
    out = node.build(json.dumps(m2))
    r = out["result"]
    img = r[1]
    ok_img = img is not None and tuple(img.shape[1:]) == (36, 64, 3)
    px = img[0, 18, 32].tolist() if ok_img else []
    _need(ok_img and px[2] > 0.6 and px[0] < 0.3, "C2 pin: first_frame = the LAST frame of scene 1 (blue, not the red start) %s" % px)
    _need("<Picture 1>" not in r[0] and "pinned only" in r[6], "C2 pin: the prompt is unchanged")
    ic1 = TL.ULSCineTimeline.IS_CHANGED(json.dumps(m2))
    c1b = clip("scene_1_00002_.mp4", "lime")
    ic2 = TL.ULSCineTimeline.IS_CHANGED(json.dumps(m2))
    _need(ic1 != ic2 and "scene_1_00002_" in ic2, "C2 IS_CHANGED follows the newest clip of scene 1")
    r = node.build(json.dumps(m2))["result"]
    px = r[1][0, 18, 32].tolist()
    _need(px[1] > 0.6 and px[2] < 0.3, "C2 the NEWEST take is the one carried (lime) %s" % px)
    mk = model(render="scene", active=2, carry_mode="keyframe")
    r = node.build(json.dumps(mk))["result"]
    _need(r[1] is not None and "<Picture 1> (from [Shot 1]) is fully referenced" in r[0], "C2 keyframe: the alignment line goes into the prompt")
    mo = model(render="scene", active=2, carry_mode="keyframe")
    mo["anchors"] = [{"src": "image_1", "at": 3.0}]
    red = torch.zeros(1, 36, 64, 3)
    red[..., 0] = 1.0
    r = node.build(json.dumps(mo), image_1=red)["result"]
    _need(r[1][0, 0, 0, 0] > 0.9 and "its own first-frame anchor -- that one wins" in r[6], "C2 an own first-frame anchor wins")
    m3 = model(render="scene", active=3)
    r = node.build(json.dumps(m3))["result"]
    _need(r[1] is None and "carry:" not in r[6], "C2 a scene without the switch: nothing carried")
    _need(TL.ULSCineTimeline.IS_CHANGED(json.dumps(model(render="whole"))) == "", "C2 a whole run never carries")

    # ------------------------------------------------------------------ C5
    Q = importlib.import_module("plsn1049.ph_h3_quality")
    from safetensors.torch import save_file
    torch.manual_seed(0)
    net = Q.LatentResizer3D(in_blocks=1, out_blocks=1, channels=32)
    upd = TMP / "ups"
    upd.mkdir()
    save_file({k: v.contiguous() for k, v in net.state_dict().items()}, str(upd / "h3.safetensors"))
    Q._full = lambda folder, name: str(upd / name)
    pos = [[torch.zeros(1, 4, 5120), {"minimax_keyframes": [{"resolved_frame_index": 0}], "minimax_frame_count": 22,
                                      "minimax_refs": [{"kind": "image"}], "x": 1}]]
    lat = {"samples": _NT((torch.randn(1, 24, 7, 4, 6), torch.randn(1, 32, 2, 10)))}
    o = Q.ULSH3LatentUpscale().run(lat, "h3.safetensors", "scale", 2.0, 0, 0, 1.0, 32, False, False, positive=pos)
    p2 = o[2]
    _need("minimax_keyframes" not in p2[0][1] and "minimax_frame_count" not in p2[0][1] and p2[0][1]["minimax_refs"] and p2[0][1]["x"] == 1
          and "minimax_keyframes" in pos[0][1], "C5 the positive leaves without keyframes, references kept, the input untouched")
    _need("keyframe(s) dropped" in o[1], "C5 ... and the info says so")
    shutil.rmtree(TMP, ignore_errors=True)

# ------------------------------------------------------------------ C3
ref = (ROOT / "nodes" / "ph_minimax_ref.py").read_text(encoding="utf-8")
opt = ref[ref.index('"optional": {'):ref.index("RETURN_TYPES")]
import re
keys = re.findall(r'^\s{16}"(\w+)": \(', opt, re.M)
_need(keys and keys[-2] == "first_frame" and keys[-3] == "tags" and keys[-1] == "ref_noise",   # AMENDED v1201: + ref_noise behind it
      "C3 first_frame: a socket appended after the last widget (v1201: ref_noise appended behind it)")
_need("KR.mark(kf, who)" in ref and "cond, {\"minimax_keyframes\": kf_list, \"minimax_frame_count\": frame_count})" in ref,
      "C3 next to references the keyframe is marked; frame count set")
_need('"first_frame"];   // v1019: + refs (Reference Board), v1049: + first_frame' in (ROOT / "web" / "js" / "ph_minimax_ref.js").read_text(encoding="utf-8"),
      "C3 the socket is a fixed pin in the frontend")

# ------------------------------------------------------------------ C4
KR = _load("kr1049", "nodes/h3_kf_refs.py")


class _C:
    def __init__(self, d):
        self.cond = d


class FakeH3:
    def extra_conds(self, **kw):
        p = {}
        if kw.get("minimax_keyframes"):
            p["cond_video_latents"] = [k["latent"] for k in kw["minimax_keyframes"]]
            p["frame_count"] = kw.get("minimax_frame_count")
        if kw.get("minimax_refs"):
            p["cond_video_latents"] = [r["latent"] for r in kw["minimax_refs"] if "latent" in r]    # Core's overwrite
        return {"minimax_payload": _C(p)}


_need(KR.ensure(FakeH3) == "ours" and KR.ensure(FakeH3) == "ours" and KR.owner(FakeH3) == "ours"
      and not hasattr(FakeH3.extra_conds._pls_orig, KR.MARK), "C4 installed once, never twice")
kf = KR.mark({"resolved_frame_index": 0, "latent": "K"}, "ours")
refs = [{"kind": "image", "latent": "R1"}, {"kind": "audio", "audio_latent": "A"}, {"kind": "image", "latent": "R2"}]
p = FakeH3().extra_conds(minimax_keyframes=[kf], minimax_refs=refs, minimax_frame_count=124)["minimax_payload"].cond
_need(p["cond_video_latents"] == ["K", "R1", "R2"] and p["frame_count"] == 124, "C4 marked: keyframe latents + reference latents %r" % p["cond_video_latents"])
p = FakeH3().extra_conds(minimax_keyframes=[{"resolved_frame_index": 0, "latent": "K"}], minimax_refs=refs)["minimax_payload"].cond
_need(p["cond_video_latents"] == ["R1", "R2"], "C4 unmarked: Core's behaviour untouched")


class Dir:
    def extra_conds(self, **kw):
        return {}


setattr(Dir.extra_conds, "_h3_director_continuity_payload_patch", True)
_need(KR.ensure(Dir) == "director" and not getattr(Dir.extra_conds, KR.MARK, False)
      and KR.mark({"resolved_frame_index": 0}, "director").get("director_context_index") == 0,
      "C4 the Director's patch installed: ours is not stacked, its marker key is used")
core = None
for c in (os.environ.get("PLS_CORE_ROOT"), os.path.abspath(os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "..", ".."))):
    if c and os.path.isfile(os.path.join(c, "comfy", "model_base.py")):
        core = c
        break
if core:
    mb = open(os.path.join(core, "comfy", "model_base.py"), encoding="utf-8").read()
    h3 = mb[mb.index("class MiniMaxH3(BaseModel):"):]
    old_core = ('payload["cond_video_latents"] = [kf["latent"] for kf in keyframes]' in h3
                and 'payload["cond_video_latents"] = [r["latent"] for r in refs if "latent" in r]' in h3)
    new_core = KR._NATIVE_MERGE in h3
    _need(old_core != new_core,
          "C4 Core is one of the two known kinds: overwrites (we merge) or appends (Core merges) -- "
          "old=%s new=%s" % (old_core, new_core))
    # v1056: on a Core that merges by itself nothing is wrapped
    class NativeH3:
        def extra_conds(self, **kw):
            payload = {}
            payload["cond_video_latents"] = payload.get("cond_video_latents", []) + [1]
            return payload
    _need(KR.owner(NativeH3) == "core-native" and KR.ensure(NativeH3) == "core-native"
          and not getattr(NativeH3.extra_conds, KR.MARK, False),
          "C4 a Core that merges natively is left alone (owner core-native, no wrapper)")
else:
    print("  note  Core not found -- the C4 source pin SKIPPED")

# ------------------------------------------------------------------ C6
_need("if (k > 1) row.append(..._carryCtl(node, m, k));" in js and "if (kk > 1) head.append(..._carryCtl(node, m, kk));" in js,
      "C6 the carry switch in the scene row and in the shot editor")
_need('(carryOf(m, k + 1) ? "\\u21e4 " : "")' in js and 'cb.dataset.pls = "carry";' in js and 'sel.dataset.pls = "carrymode";' in js,
      "C6 the scene bar marks a carrying scene")
_need('lab.append(cb, _el("span", "", "\\u21e4 begins on the last frame of scene " + (k - 1)));' in js, "C6 the switch says what it does")

bad = [f for f in ("nodes/h3_kf_refs.py", "nodes/ph_minimax_ref.py", "nodes/ph_cine_timeline.py", "nodes/cine_timeline.py",
                   "nodes/ph_h3_quality.py", "web/js/ph_cine_timeline.js", "tests/test_v1049_carry.py")
       if any(ord(ch) > 127 for ch in (ROOT / f).read_text(encoding="utf-8"))]
_need(not bad, "S ASCII only %r" % bad)
if FAILS:
    print("test_v1049_carry: %d FAILURE(S)" % len(FAILS))
    sys.exit(1)
print("test_v1049_carry: PASS (C1-C6, S)")
