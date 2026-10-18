# -*- coding: ascii -*-
"""Guard v1051 -- the Cine Timeline's image flow (Frank 02.10.: "what comes in,
what goes out, what comes back as a result and how it is used for further
runs ... especially image templates for further runs").

  I1  the model: nine image slots; @clip:K:first|last|S parsed; an @clip
      anchor without a rendered clip is refused BY NAME (scene / whole, the
      project folder), a malformed @clip too; `target` fl2va|ref2va in
      DEFAULT on both sides and validated.
  I2  the target: fl2va keeps the official alignment line; ref2va writes none
      (the first frame is a pin, not a picture of the prompt), says where the
      first frame goes and WARNS about a last-frame anchor (ref2va has none).
  I3  the node, with a REAL clip (ffmpeg): image_7 is a source; @clip:0:last /
      :first / :1.0 hand out the clip's own frames (their colours), the used
      frame is kept as output/<project>/frames/<scene>_<which>.png, the info
      names it, IS_CHANGED follows the clip; guides between first and last
      frame are announced with the Core verdict (no AddGuide -> WARN).
  I4  MiniMax Continue: nothing to add = pass-through on ANY Core (one
      workflow serves 2 and 6 images); guides on a Core without AddGuide are
      still refused by name.
  I5  the frontend: nine slots, the pins grow (wired + one spare, never a
      wired one dropped, display order kept), clip frames offered in the
      source list but never sent as wired, the model selector.
  S   ASCII.

Script-style: exit 0 = pass.
"""
import importlib
import importlib.util
import json
import shutil
import subprocess
import sys
import tempfile
import types
from pathlib import Path

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


T = _load("ct1051", "nodes/cine_timeline.py")
JS = (ROOT / "web" / "js" / "ph_cine_timeline.js").read_text(encoding="utf-8")
PJS = (ROOT / "web" / "js" / "ph_cine_timeline_pins.js").read_text(encoding="utf-8")


def model(**kw):
    m = json.loads(json.dumps(T.DEFAULT))
    m.update({"seconds": 6.0, "project": "imgp", "shots": [{"at": 0.0, "cut": T.CUTS[0], "text": "a"}]})
    m.update(kw)
    return m


# ------------------------------------------------------------------ I1
print("I1 the model")
_need(T.IMAGE_SLOTS == tuple("image_%d" % n for n in range(1, 10)), "I1 nine image slots")
_need(T.clip_src("@clip:2:last") == (2, "last") and T.clip_src("@clip:0:first") == (0, "first")
      and T.clip_src("@clip:1:2.5") == (1, 2.5) and T.clip_src("@clip:x:last") is None
      and T.clip_src("image_1") is None, "I1 @clip:K:first|last|S parsed, others not")
e = T.validate(model(anchors=[{"src": "@clip:0:last", "at": 0.0}]), set())
# AMENDED v1152: the stack is not a folder under output/ any more -- the refusal names the project's stack
_need(any("the whole timeline has no rendered clip in the stack of imgp yet" in x for x in e),
      "I1 @clip without a clip is refused by name (%s)" % (e[:1],))
e = T.validate(model(anchors=[{"src": "@clip:2:last", "at": 0.0}]), set())
_need(any("scene 2 has no rendered clip" in x for x in e), "I1 ... and names the scene")
e = T.validate(model(anchors=[{"src": "@clip:last", "at": 0.0}]), set())
_need(any("is not a clip frame" in x for x in e), "I1 a malformed @clip is named as such")
_need(not T.validate(model(anchors=[{"src": "image_9", "at": 0.0}]), {"image_9"}), "I1 image_9 is a valid source")
_need(T.DEFAULT.get("target") == "fl2va" and 'target: "fl2va" }' in JS, "I1 target in DEFAULT, both sides")
_need(any("target" in x for x in T.validate(model(target="x"))) and not T.validate(model(target="ref2va")),
      "I1 target validated")

# ------------------------------------------------------------------ I2
print("I2 fl2va vs ref2va")
fl = T.compile_prompt(model(anchors=[{"src": "image_1", "at": 0.0}, {"src": "image_2", "at": 157 / 24.0}]))
_need(fl["prompt"].startswith("How the reference pictures align"), "I2 fl2va: the official alignment line")
rf = T.compile_prompt(model(target="ref2va", anchors=[{"src": "image_1", "at": 0.0}, {"src": "image_2", "at": 157 / 24.0}]))
_need(rf["prompt"].startswith("integrated_multimodal_description:"), "I2 ref2va: no alignment line")
_need(any("MiniMax Reference first_frame" in l for l in rf["lines"]), "I2 ref2va: says where the first frame goes")
_need(any(l.strip().startswith("WARN: last frame") for l in rf["lines"]), "I2 ref2va: a last-frame anchor is warned about")

# ------------------------------------------------------------------ I3
print("I3 the node with a real clip")
ff = shutil.which("ffmpeg")
if not ff:
    print("  note  ffmpeg not found -- I3 SKIPPED")
else:
    import torch
    TMP = Path(tempfile.mkdtemp(prefix="v1051_"))
    for d in ("output/imgp", "input", "temp"):
        (TMP / d).mkdir(parents=True, exist_ok=True)
    fp = types.ModuleType("folder_paths")
    fp.get_output_directory = lambda: str(TMP / "output")
    fp.get_input_directory = lambda: str(TMP / "input")
    fp.get_temp_directory = lambda: str(TMP / "temp")
    fp.models_dir = str(TMP)
    fp.folder_names_and_paths = {}
    fp.add_model_folder_path = lambda n, p: None
    fp.get_filename_list = lambda n: []
    comfy = types.ModuleType("comfy")
    mm = types.ModuleType("comfy.model_management")
    mm.unload_all_models = lambda: None
    mm.soft_empty_cache = lambda force=False: None
    comfy.model_management = mm
    nt = types.ModuleType("comfy.nested_tensor")
    nt.NestedTensor = object
    comfy.nested_tensor = nt
    capi = types.ModuleType("comfy_api")
    ii = types.ModuleType("comfy_api.input_impl")
    ii.VideoFromComponents = object
    ut = types.ModuleType("comfy_api.util")
    ut.VideoComponents = object
    sys.modules.update({"folder_paths": fp, "comfy": comfy, "comfy.model_management": mm,
                        "comfy.nested_tensor": nt, "comfy_api": capi, "comfy_api.input_impl": ii,
                        "comfy_api.util": ut})
    nh = types.ModuleType("node_helpers")       # v1061: Continue is imported for real now
    nh.conditioning_set_values = lambda cond, v: [[c[0], dict(c[1], **v)] for c in cond]
    sys.modules.setdefault("node_helpers", nh)
    pkg = types.ModuleType("plsn1051")
    pkg.__path__ = [str(ROOT / "nodes")]
    sys.modules["plsn1051"] = pkg
    TL = importlib.import_module("plsn1051.ph_cine_timeline")
    # v1061: the guide verdict reads the H3 MODEL code only (no comfy_extras guess); the stub world sets it.
    CO51 = importlib.import_module("plsn1051.ph_minimax_continue")
    CO51.model_has_guides = lambda: False         # the pin
    A = sys.modules["plsn1051.cine_assemble"]

    # 48 frames at 24 fps: red for 1.5 s, then blue for 0.5 s
    clip = TMP / "output" / "imgp" / "whole_00001_.mp4"
    subprocess.run([ff, "-v", "error", "-y", "-f", "lavfi", "-i", "color=c=red:s=64x36:r=24:d=1.5",
                    "-f", "lavfi", "-i", "color=c=blue:s=64x36:r=24:d=0.5",
                    "-filter_complex", "[0][1]concat=n=2:v=1:a=0", "-c:v", "libx264", "-pix_fmt", "yuv420p",
                    str(clip)], capture_output=True)
    A.register(str(TMP / "output" / "imgp"), 0, str(clip))

    def rgb(img):
        return [round(float(img[0, :, :, c].mean()), 1) for c in range(3)]

    node = TL.ULSCineTimeline()
    out = node.build(json.dumps(model(anchors=[{"src": "@clip:0:last", "at": 0.0}])))["result"]
    c = rgb(out[1])
    _need(c[2] > 0.5 and c[0] < 0.3, "I3 @clip:0:last -> first_frame is the clip's LAST frame (blue %s)" % c)
    # v1083 (DECLARED): the line names the frame number the film shows last
    # AMENDED v1152: the kept frame lies in input/cine_media/<project>/stack/frames/ (output/ holds only day folders)
    # AMENDED v1168: the line names the film frame AND the clip's own frame (a clip at another rate is said)
    _need("clip frame @clip:0:last: last (film frame " in out[6] and " of whole_00001_.mp4) -> input/cine_media/imgp/stack/frames/whole_last.png" in out[6],
          "I3 the info names the clip and the kept PNG")
    _need((TMP / "input" / "cine_media" / "imgp" / "stack" / "frames" / "whole_last.png").is_file()
          and not (TMP / "output" / "imgp" / "frames").exists(), "I3 the frame is kept as PNG in frames/ -- in input/, not output/")
    out = node.build(json.dumps(model(anchors=[{"src": "@clip:0:first", "at": 0.0}])))["result"]
    c = rgb(out[1])
    _need(c[0] > 0.5 and c[2] < 0.3, "I3 @clip:0:first -> the clip's first frame (red %s)" % c)
    out = node.build(json.dumps(model(anchors=[{"src": "@clip:0:1.8", "at": 0.0}])))["result"]
    c = rgb(out[1])
    _need(c[2] > 0.5, "I3 @clip:0:1.8 -> the frame at 1.8 s (blue %s)" % c)
    ic = TL.ULSCineTimeline.IS_CHANGED(json.dumps(model(anchors=[{"src": "@clip:0:last", "at": 0.0}])))
    _need("whole_00001_.mp4" in ic, "I3 IS_CHANGED follows the clip (%r)" % ic[-60:])
    img7 = torch.full((1, 36, 64, 3), 0.25)
    out = node.build(json.dumps(model(anchors=[{"src": "image_7", "at": 0.0}])), image_7=img7)["result"]
    _need(out[1] is not None and abs(float(out[1].mean()) - 0.25) < 1e-6, "I3 image_7 is a source")
    g = model(anchors=[{"src": "image_1", "at": 0.0}, {"src": "image_2", "at": 3.0}])
    out = node.build(json.dumps(g), image_1=img7, image_2=img7)["result"]
    _need(len(out[3]) == 1 and "WARN: this Core has no MiniMaxH3AddGuide" in out[6],
          "I3 a guide in between goes out as `anchors`, with the Core verdict")
    ps = TL.preview_sources(model(anchors=[{"src": "@clip:0:last", "at": 0.0}, {"src": "@clip:3:last", "at": 1.0}]), ["image_1"])
    _need(set(ps) == {"image_1", "@clip:0:last"}, "I3 the preview knows which clip frames exist (%s)" % sorted(ps))
    _need(TL.preview_sources(model(), None) is None, "I3 ... and stays unknowable when the editor cannot tell")
    # the other verdict: a Core WITH AddGuide
    CO51.model_has_guides = lambda: True          # master
    out = node.build(json.dumps(g), image_1=img7, image_2=img7)["result"]
    _need("(this Core has MiniMaxH3AddGuide)" in out[6], "I3 a Core with AddGuide: the guides go to Continue, said")
    shutil.rmtree(TMP, ignore_errors=True)

# ------------------------------------------------------------------ I4
print("I4 MiniMax Continue")
CT = (ROOT / "nodes" / "ph_minimax_continue.py").read_text(encoding="utf-8")
i_pass = CT.find("nothing to add")
i_core = CT.find("if not core_has_guides():", CT.find("def build"))
_need(0 < i_pass < i_core, "I4 the pass-through comes BEFORE the Core check (any Core)")
_need("does nothing -- wire one" not in CT, "I4 'anchor only' without anything is no longer an error")

# ------------------------------------------------------------------ I5
print("I5 the frontend")
_need("export const N_IMAGE_SLOTS = 9;" in JS and 'const SLOTS = Array.from({ length: N_IMAGE_SLOTS }' in JS,
      "I5 nine slots in the timeline frontend")
_need("clipChoices(_model(node))" in JS and "wired.length ? wired" in JS, "I5 clip frames offered in the source list")
sof = JS[JS.index("export function sourcesOf"):]
sof = sof[:sof.index("\n}\n")]
_need("@clip" not in sof, "I5 ... but never sent as wired (sourcesOf)")
_need('tgt.dataset.pls = "target"' in JS and "mm.target = tgt.value" in JS, "I5 the model selector")
node = shutil.which("node")
if node:
    h = PJS[PJS.index("const NODE"):PJS.index("const LG_INPUT")]
    h = h.replace("export ", "")
    h += """
const mk = (wired) => ({ inputs: ["timeline", "refs", "image_1", "image_2", "image_3", "storyboard"].map((n) => ({ name: n, link: wired.includes(n) ? 1 : null })),
  removeInput(i) { this.inputs.splice(i, 1); }, addInput(n) { this.inputs.push({ name: n, link: null }); } });
const out = {};
let n = mk([]); tidy(n); out.fresh = n.inputs.map((i) => i.name).join(",");
n = mk(["image_1", "image_2"]); tidy(n); out.two = n.inputs.map((i) => i.name).join(",");
n = mk(["image_3"]); tidy(n); out.gap = n.inputs.map((i) => i.name).join(",");
n = { inputs: [{ name: "timeline", link: null }, { name: "image_9", link: 5 }], removeInput(i) { this.inputs.splice(i, 1); },
      addInput(nm) { this.inputs.push({ name: nm, link: null }); } };
tidy(n); out.nine = n.inputs.map((i) => i.name).join(",");
console.log(JSON.stringify(out));
"""
    r = subprocess.run([node, "-e", h], capture_output=True, text=True, timeout=60)
    try:
        o = json.loads(r.stdout.strip().splitlines()[-1])
    except Exception:
        o = {}
        print(r.stdout[-300:], r.stderr[-600:])
    _need(o.get("fresh") == "timeline,refs,image_1,storyboard", "I5 fresh node: image_1 only (%s)" % o.get("fresh"))
    _need(o.get("two") == "timeline,refs,image_1,image_2,image_3,storyboard", "I5 two wired + one spare (%s)" % o.get("two"))
    _need(o.get("gap") == "timeline,refs,image_1,image_2,image_3,image_4,storyboard", "I5 a wired image_3 keeps 1..3 + spare (%s)" % o.get("gap"))
    _need(o.get("nine") == "timeline," + ",".join("image_%d" % k for k in range(1, 10)), "I5 image_9 wired: all nine, in order (%s)" % o.get("nine"))
else:
    print("  skip  node not found")

# ------------------------------------------------------------------ S
print("S ASCII")
for rel in ("nodes/cine_timeline.py", "nodes/ph_cine_timeline.py", "nodes/ph_minimax_continue.py",
            "web/js/ph_cine_timeline.js", "web/js/ph_cine_timeline_pins.js", "tests/test_v1051_timeline_images.py"):
    _need((ROOT / rel).read_bytes().isascii(), "S %s is pure ASCII" % rel)

print("FAILS %d" % len(FAILS))
sys.exit(1 if FAILS else 0)
