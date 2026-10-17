# -*- coding: ascii -*-
"""Guard v1064 -- the render lane's film strip of a clip that is not 24 fps,
and the Cine Timeline's head fields that grow.

Frank, 08.10. (field, run 1 of the two-tracks test): the clip on V1 is the
48 fps interpolation; its strip showed the first 2.4 s stretched over the
whole lane -- shot 3 never appeared. Cause: the strip took every n-th frame
of the FILE with n counted in FILM frames (24 fps). And: "let the triggers
and style fields grow down as soon as the text is wider than the line".

  H1  the server, REAL ffmpeg over edit_plan(): a 48 fps clip of three colours
      (red 0-2 s, green 2-4 s, blue 4-5.875 s) in output/<project>/ -- the
      strip's tile for film second 1 / 3 / 5, picked the way the lane picks it
      (web/js tilePlan: tile = film frame // strip_step), is red / green / blue.
      A 24 fps clip: the same answer and NO resample (the cache key differs).
  H2  the browser's numbers (node.js): headFieldH clamps one line .. HEAD_MAX_H;
      fieldH = the old sum + headExtra (0 without a grown head).
  S   statics: the render lane passes film_fps, the layer rows keep the file's
      frames; the head fields are one-row textareas, Enter ends the edit, a
      line break becomes a space, they are fitted on render and on the preview
      fit; ASCII.

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


def _fail(m):
    FAILS.append(m)
    print("  FAIL  " + m)


def _ok(m):
    print("  ok    " + m)


def _need(c, m):
    (_ok if c else _fail)(m)


def _load(name, rel):
    spec = importlib.util.spec_from_file_location(name, str(ROOT / rel))
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


T = _load("ct1064", "nodes/cine_timeline.py")
tl = (ROOT / "nodes" / "ph_cine_timeline.py").read_text(encoding="utf-8")
js = (ROOT / "web" / "js" / "ph_cine_timeline.js").read_text(encoding="utf-8")

# ------------------------------------------------------------------ H1 the server
ff = shutil.which("ffmpeg")
if ff is None:
    print("  note  ffmpeg not found -- H1 SKIPPED")
else:
    try:
        from PIL import Image
    except Exception:
        Image = None
    TMP = Path(tempfile.mkdtemp(prefix="v1064_"))
    for d in ("output/fogbeach", "output/plain", "input", "temp"):
        (TMP / d).mkdir(parents=True, exist_ok=True)

    class _NT:
        is_nested = True

    comfy = types.ModuleType("comfy")
    nt = types.ModuleType("comfy.nested_tensor")
    nt.NestedTensor = _NT
    comfy.nested_tensor = nt
    capi = types.ModuleType("comfy_api")
    ii = types.ModuleType("comfy_api.input_impl")
    ii.VideoFromComponents = object
    ut = types.ModuleType("comfy_api.util")
    ut.VideoComponents = object
    fp = types.ModuleType("folder_paths")
    fp.get_output_directory = lambda: str(TMP / "output")
    fp.get_input_directory = lambda: str(TMP / "input")
    fp.get_temp_directory = lambda: str(TMP / "temp")
    fp.base_path = str(TMP)
    sys.modules.update({"comfy": comfy, "comfy.nested_tensor": nt, "comfy_api": capi,
                        "comfy_api.input_impl": ii, "comfy_api.util": ut, "folder_paths": fp})
    pkg = types.ModuleType("plsn1064")
    pkg.__path__ = [str(ROOT / "nodes")]
    sys.modules["plsn1064"] = pkg
    TL = importlib.import_module("plsn1064.ph_cine_timeline")

    def colours(fps, out):
        # red 0-2 s, green 2-4 s, blue 4-5.875 s (141 film frames)
        parts = []
        for i, (c, sec) in enumerate((("0xe00000", 2.0), ("0x00c000", 2.0), ("0x0000e0", 1.875))):
            parts += ["-f", "lavfi", "-i", "color=c=%s:s=320x180:r=%d:d=%.4f" % (c, fps, sec)]
        r = subprocess.run([ff, "-v", "error", "-y"] + parts + [
            "-filter_complex", "[0:v][1:v][2:v]concat=n=3:v=1:a=0[v]", "-map", "[v]",
            "-c:v", "libx264", "-pix_fmt", "yuv420p", "-r", str(fps), str(out)], capture_output=True, text=True)
        return r.returncode == 0

    def tile_colour(item, film_frame):
        img = Image.open(str(TMP / "temp" / "pls_cine_strips" / Path(item["strip_file"]).name)).convert("RGB")
        tiles = item["strip_tiles"]
        idx = max(0, min(tiles - 1, film_frame // max(1, item["strip_step"])))
        tw = img.width / tiles
        px = img.getpixel((int(idx * tw + tw / 2), img.height // 2))
        return "rgb"[max(range(3), key=lambda k: px[k])]

    ok = (colours(48, TMP / "output/fogbeach/whole_00001_.mp4") and colours(24, TMP / "output/plain/whole_00001_.mp4"))
    _need(ok, "H1 test clips made (48 fps and 24 fps, red / green / blue)")
    if ok and Image is not None:
        made = []
        orig = TL._strip

        def spy(path, frames, *a, **k):
            out = orig(path, frames, *a, **k)
            made.append((Path(path).parent.name, k.get("film_fps"), out))
            return out
        TL._strip = spy
        res = {}
        for proj in ("fogbeach", "plain"):
            model = T.load(json.dumps({"seconds": 5.875, "project": proj, "render": "whole",
                                       "shots": [{"at": 0, "cut": "the camera cuts to", "text": "x"}]}))
            d = TL.edit_plan(model, {}, strips=True)
            item = [c for c in d["clips"] if c.get("path")][0]
            sp = [m for m in made if m[0] == proj][-1]
            item["strip_file"] = sp[2]
            res[proj] = (item, sp)
        TL._strip = orig
        it48, sp48 = res["fogbeach"]
        it24, sp24 = res["plain"]
        _need(abs(it48["fps"] - 48) < 0.01 and it48["frames"] == 141 and it48.get("src_frames") == 282,
              "H1 the 48 fps clip counts 141 film frames (282 of its own)")
        got48 = [tile_colour(it48, f) for f in (24, 72, 120)]
        _need(got48 == ["r", "g", "b"], "H1 48 fps: the lane's tile for film second 1 / 3 / 5 is red / green / blue (got %r)" % got48)
        got24 = [tile_colour(it24, f) for f in (24, 72, 120)]
        _need(got24 == ["r", "g", "b"], "H1 24 fps: red / green / blue as before (got %r)" % got24)
        _need(sp48[1] == 24 and sp24[1] is None, "H1 only a clip off 24 fps is resampled (film_fps %r / %r)" % (sp48[1], sp24[1]))
        raw = orig(str(TMP / "output/fogbeach/whole_00001_.mp4"), 282)
        _need(sp48[2] and raw and Path(sp48[2]).name != Path(raw).name,
              "H1 the same file resampled and not resampled are cached apart")
    elif ok:
        print("  note  PIL not found -- H1 colours SKIPPED")
    shutil.rmtree(TMP, ignore_errors=True)

# ------------------------------------------------------------------ H2 the browser's numbers
if shutil.which("node") is None:
    print("  note  node.js not found -- H2 SKIPPED")
else:
    tmp = Path(tempfile.mkdtemp(prefix="v1064js_"))
    body = (js.replace('import { app } from "../../scripts/app.js";', "const app = { registerExtension() {} };")
              .replace('import { api } from "../../scripts/api.js";', "const api = {};")
              .replace('import { setHidden, refit } from "./ph_widget_vis.js";', "const setHidden = () => {}, refit = () => {};"))
    (tmp / "m.mjs").write_text(body, encoding="utf-8")
    (tmp / "h.mjs").write_text(r"""
import * as T from "./m.mjs";
const o = {};
o.hf = [T.headFieldH(0), T.headFieldH(19), T.headFieldH(52.2), T.headFieldH(1e6), T.HEAD_LINE_H, T.HEAD_MAX_H];
// AMENDED v1150: the head is a row of its own, shown while it holds text; it grows by headX
const H = { triggers: "a, b," };
const old = T.fieldH({ _plsTl: { m: H } });
o.fh = [T.fieldH({ _plsTl: { m: H } }), T.fieldH({ _plsTl: { headX: 34, m: H } }), T.fieldH({ _plsTl: { headX: -5, m: H } }), old, T.fieldH({ _plsTl: { headX: 34, m: {} } }), T.fieldH({})];
o.hx = [T.headExtra({}), T.headExtra(null), T.headExtra({ _plsTl: { headX: 34 } })];
console.log(JSON.stringify(o));
""", encoding="utf-8")
    r = subprocess.run(["node", str(tmp / "h.mjs")], capture_output=True, text=True, timeout=60)
    shutil.rmtree(tmp, ignore_errors=True)
    line = [l for l in r.stdout.splitlines() if l.startswith("{")]
    if r.returncode != 0 or not line:
        _fail("H2 harness failed: %s" % r.stderr.strip()[-400:])
    else:
        d = json.loads(line[-1])
        L, M = d["hf"][4], d["hf"][5]
        _need(d["hf"][:4] == [L, L, 53, M] and L == 19 and M == 400, "H2 headFieldH: one line .. HEAD_MAX_H, rounded up (%r)" % d["hf"])
        _need(d["fh"][0] == d["fh"][3] and d["fh"][1] == d["fh"][3] + 34 and d["fh"][2] == d["fh"][3] and d["fh"][4] == d["fh"][5],
              "H2 fieldH = the old sum, + what the head grew (never less) (%r)" % d["fh"])
        _need(d["hx"] == [0, 0, 34], "H2 headExtra: 0 without a grown head")

# ------------------------------------------------------------------ S statics
_need('sp = _strip(path, ci["frames"], film_fps=E.FPS if "src_frames" in ci else None)' in tl,
      "S the render lane strips in film frames (resampled when the clip is not 24 fps)")
_need('sp = _strip(path, ci.get("src_frames", ci["frames"]))' in tl,
      "S the layer rows keep the file's own frames (web/js nat maps them)")
_need('("fps=%d," % int(film_fps) if film_fps else "")' in tl and "tiles, height, film_fps or \"\")" in tl,
      "S the resample is the strip's first filter and part of the cache key")
fit = js[js.index("function _fitHead("):js.index("function _fitPreview(")] if "function _fitHead(" in js else ""
_need('_el("textarea", INPUT + HEADF); style.rows = 1;' in js and '_el("textarea", INPUT + HEADF + "color:#c5e1a5;"); trig.rows = 1;' in js,
      "S triggers and style are one-row textareas that wrap")
_need('if (e.key === "Enter") { e.preventDefault(); el.blur(); }' in js and 'el.value.replace(/\\r?\\n|\\r/g, " ")' in js,
      "S Enter ends the edit, a pasted line break becomes a space")
_need("st.headX = x;" in fit and "_layout(node);" in fit,   # AMENDED v1150: _layout sets the field and refits
      "S a grown head re-sizes the field and the node")
_need('st.trig.value = m.triggers || "";\n    _fitHead(node);' in js
      and "_fitHead(node);\n        _fitTools(node);\n        _layout(node);" in js,
      "S the head is fitted when its text is written and when the width changes")
_need('["headrow", HEAD_LINE_H + headExtra(node) + 6, headShown(mm)],' in js,   # AMENDED v1150
      "S fieldH counts the grown head")
bad = [f for f in ("nodes/ph_cine_timeline.py", "web/js/ph_cine_timeline.js", "tests/test_v1064_strip_and_head.py")
       if any(ord(ch) > 127 for ch in (ROOT / f).read_text(encoding="utf-8"))]
_need(not bad, "S ASCII only %r" % bad)

if FAILS:
    print("test_v1064_strip_and_head: %d FAILURE(S)" % len(FAILS))
    sys.exit(1)
print("test_v1064_strip_and_head: PASS (H1, H2, S)")
