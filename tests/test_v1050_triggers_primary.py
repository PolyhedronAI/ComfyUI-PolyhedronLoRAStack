# -*- coding: ascii -*-
"""Guard v1050 -- consistency across separately rendered scenes, the run's
clip, an honest upscale clock and a readable CLIP Text Encode under replace
(Frank 01.10.: "two separate areas, one for the triggers, one for the style
... I want my // separators back in the CTE").

  T1  the timeline model: "triggers" in DEFAULT on both sides; head_block puts
      // TRIGGERS and // LOOK at the head of [Shot 1], each line closed with a
      comma; no triggers = the style alone (the old form, unchanged); the
      head is the SAME for every scene rendered alone (scene 1 and scene 3).
  T2  the run's clip (primary): register(primary=True) stores the pick; the
      primary leads scene_takes / scene_files even when an older-named file
      of the same run is newer by a moment (grace); a LATER file wins again;
      the route passes "primary"; the editor posts primary: true at the
      run's end.
  T3  the clock: a learned prior answers before any measurement (rung 1.5),
      a measured rate replaces it; the Power Upscale feeds its phase
      estimates as priors and opens step n with step n-1's real duration;
      past the plan the heartbeat says "+X over the plan", not "~0:00 left".
  T4  the CTE: under replace + wired the own fields fold (a bar says why),
      unfolded they are greyed; the green field shows the RAW text (// rubrics),
      grows with it (measured, or estimated when Nodes 2.0 has not mounted
      it) and is EXT_H again outside replace; the counter counts only what is
      encoded; the ext/fold widgets carry their own types so Nodes 2.0 mounts
      the real element; the backend sends the raw text.
  S   ASCII.

Script-style: exit 0 = pass.
"""
import importlib.util
import json
import os
import shutil
import subprocess
import sys
import tempfile
import time
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


T = _load("ct1050", "nodes/cine_timeline.py")
A = _load("ca1050", "nodes/cine_assemble.py")
RC = _load("rc1050", "nodes/ph_runclock.py")
TL_JS = (ROOT / "web" / "js" / "ph_cine_timeline.js").read_text(encoding="utf-8")
CTE_JS = (ROOT / "web" / "js" / "ph_clip_encode.js").read_text(encoding="utf-8")
CTE_PY = (ROOT / "nodes" / "ph_clip_encode.py").read_text(encoding="utf-8")
TL_PY = (ROOT / "nodes" / "ph_cine_timeline.py").read_text(encoding="utf-8")
PU_PY = (ROOT / "nodes" / "ph_power_upscale.py").read_text(encoding="utf-8")
PR_PY = (ROOT / "nodes" / "ph_progress.py").read_text(encoding="utf-8")

# ------------------------------------------------------------------ T1
print("T1 triggers + look at the head of [Shot 1]")
_need(T.DEFAULT.get("triggers") == "", "T1 DEFAULT has triggers (python)")
_need('triggers: ""' in TL_JS[TL_JS.index("carry_mode: \"pin\""):][:80], "T1 DEFAULT has triggers (js)")


def _model(trig, style, scenes=False):
    m = json.loads(json.dumps(T.DEFAULT))
    m["seconds"] = 6.0
    m["style"] = style
    m["triggers"] = trig
    sh = [{"at": 0.0, "cut": "the camera cuts to", "text": "// SHOT\na wide shot,"},
          {"at": 2.0, "cut": "the camera cuts to", "text": "// SHOT\na medium shot,"},
          {"at": 4.0, "cut": "the camera cuts to", "text": "// SHOT\na close-up,"}]
    if scenes:
        for s in sh[1:]:
            s["scene"] = True
    m["shots"] = sh
    return m


hb = T.head_block(_model("ma8elstrom, cu7ie", "RAW photo, muted colours"))
_need(hb.startswith("// TRIGGERS\nma8elstrom, cu7ie,\n// LOOK\nRAW photo, muted colours,"),
      "T1 head_block: // TRIGGERS + // LOOK, every line closed with a comma (%r)" % hb[:70])
_need(T.head_block(_model("", "RAW photo")) == "RAW photo", "T1 no triggers = the style alone (the old form)")
_need(T.head_block(_model("a8b", "")).startswith("// TRIGGERS\na8b,") and "// LOOK" not in T.head_block(_model("a8b", "")),
      "T1 triggers without a look: no empty // LOOK")
p = T.compile_prompt(_model("ma8elstrom", "RAW photo"))["prompt"]
_need("[Shot 1] // TRIGGERS\nma8elstrom,\n// LOOK\nRAW photo,\n// SHOT" in p, "T1 compile: the head opens [Shot 1]")
heads = []
for k in (1, 3):
    m = _model("ma8elstrom, cu7ie", "RAW photo", scenes=True)
    m["render"] = "scene"
    m["active"] = k
    tgt, _k, _n = T.resolve(m)
    pp = T.compile_prompt(tgt)["prompt"]
    i = pp.find("[Shot 1]")
    heads.append(pp[i:pp.find("// SHOT", i)])
_need(heads[0] == heads[1] and "ma8elstrom, cu7ie," in heads[0], "T1 scene 1 and scene 3 alone: the SAME head")
_need('data-pls' in TL_JS and 'trig.dataset.pls = "triggers"' in TL_JS and "mm.triggers = trig.value" in TL_JS,
      "T1 the timeline UI: its own triggers field, saved to the model")

# ------------------------------------------------------------------ T2
print("T2 the run's clip leads")
tmp = tempfile.mkdtemp(prefix="pls1050_")
try:
    out = os.path.join(tmp, "output")
    folder = os.path.join(out, "proj")   # the project folder sits right under output/
    os.makedirs(folder)
    raw = os.path.join(out, "raw_00001.mp4")
    fin = os.path.join(out, "final_00001.mp4")
    for f in (raw, fin):
        open(f, "wb").write(b"\0" * 64)
    now = time.time()
    os.utime(fin, (now - 1, now - 1))
    os.utime(raw, (now, now))           # the raw clip written a moment LATER
    A.register(folder, 1, raw)
    A.register(folder, 1, fin, primary=True)
    pr = A.read_primary(folder)
    _need(pr.get(1, {}).get("rel") == "final_00001.mp4", "T2 register(primary=True) stores the pick (%r)" % pr)
    tk = A.scene_takes(folder, 1).get(1, [])
    _need(tk and os.path.basename(tk[0]) == "final_00001.mp4", "T2 the primary leads the takes (%s)" % [os.path.basename(x) for x in tk])
    sf = A.scene_files(folder, 1).get(1, {})
    _need(os.path.basename(sf.get("path", "")) == "final_00001.mp4", "T2 scene_files: the primary is THE clip")
    later = os.path.join(out, "next_00002.mp4")
    open(later, "wb").write(b"\0" * 64)
    t2 = pr[1]["at"] + A.PRIMARY_GRACE + 30
    os.utime(later, (t2, t2))
    A.register(folder, 1, later)
    tk = A.scene_takes(folder, 1).get(1, [])
    _need(tk and os.path.basename(tk[0]) == "next_00002.mp4", "T2 a LATER run wins again by its own time")
    A.register(folder, 2, raw)
    _need(2 not in A.read_primary(folder) and 1 in A.read_primary(folder), "T2 a plain register keeps the other picks")
finally:
    shutil.rmtree(tmp, ignore_errors=True)
_need('body.get("primary")' in TL_PY, "T2 the register route passes primary")
_need("primary: true" in TL_JS and "pickTrackFile" in TL_JS, "T2 the editor posts the picked clip as primary")

# ------------------------------------------------------------------ T3
print("T3 the clock")
clk = RC._RunClock(now=lambda: 0.0)
clk.post("dec:joint", 1, 1.0)
_need(clk.eta() is None, "T3 no rate, no prior: no estimate (says so)")
clk.prior("dec:joint", 135.0)
_need(clk.eta() == 135.0, "T3 the learned prior answers before any measurement (%r)" % clk.eta())
clk.post("step:joint", 3, 1.0)
clk.prior("step:joint", 40.0)
clk.posts["step:joint"]["rate"] = 90.0
_need(abs(clk.eta("joint") - (135.0 + 270.0)) < 1e-6, "T3 a measured rate replaces the prior")
_need("clock.prior(_k, _e)" in PU_PY and '"dec:joint", est.get("dec")' in PU_PY, "T3 Power Upscale feeds its learned phase estimates")
_need('stw.get("last_dt") or est["step"]' in PU_PY and 'stw["last_dt"] = dt' in PU_PY, "T3 step n opens with step n-1's real duration")
_need("_left_tail(" in PR_PY[PR_PY.index("class _Phase"):], "T3 the phase heartbeat uses _left_tail")
PG = {"_fmt_clock": RC._fmt_clock}
src = PR_PY[PR_PY.index("def _left_tail"):]
src = src[:src.index("\n\n\n")]
exec(src, PG)
lt = PG["_left_tail"](274.0, 123.0)
_need("over the plan" in lt and "left" not in lt, "T3 past the plan: '+X over the plan' (%r)" % lt)
_need("left of" in PG["_left_tail"](60.0, 123.0), "T3 inside the plan: '~X left of ~Y'")

# ------------------------------------------------------------------ T4
print("T4 the CTE under replace")
for k in ('"pos_ext_raw"', '"neg_ext_raw"', '"external_mode"'):
    _need(k in CTE_PY, "T4 backend sends %s" % k)
_need('addDOMWidget(name, "pls_ext"' in CTE_JS and 'addDOMWidget(name, "pls_fold"' in CTE_JS,
      "T4 ext/fold widgets: own types (Nodes 2.0 mounts the real element)")
_need('"external_mode"]' in CTE_JS, "T4 external_mode switch re-runs the visibility")
av = CTE_JS[CTE_JS.index("function _applyVisibility"):]
av = av[:av.index("\n}\n")]
_need("_fitExt(node)" in av, "T4 a mode switch resets the green field")
_need("(rp ? 0 : c.words)" in CTE_JS, "T4 the counter: own words not counted under replace")


def _fn(name):
    i = CTE_JS.index("function " + name + "(")
    line = CTE_JS[i:CTE_JS.index("\n", i)]
    if line.rstrip().endswith("}"):      # a one-line function
        return line
    j = CTE_JS.index("\n}\n", i)
    return CTE_JS[i:j + 2]


harness = "\n".join([
    "globalThis.requestAnimationFrame = () => 0;",
    "const EXT_H = 96, EXT_MAX_H = 640, DIM = '0.45', MAX_SEGMENTS = 4, FOLD_H = 22;",
    "const EXT_FIELDS = [['pls_ext_pos','pos_external','pos'],['pls_ext_neg','neg_external','neg']];",
    CTE_JS[CTE_JS.index("const EXT_LINE_H"):CTE_JS.index("\n", CTE_JS.index("const EXT_LINE_H"))],
    "function _w(n, name) { return (n.widgets || []).find(w => w.name === name); }",
    "let hidden = {}; function _hide(w, h) { if (w) { w.hidden = h; hidden[w.name] = h; } }",
    "function _syncFold(node, side, on) { const w = _w(node, side === 'neg' ? 'pls_fold_neg' : 'pls_fold_pos'); if (w) w.hidden = !on; }",
    _fn("_extConnected"), _fn("_replacing"), _fn("_unfolded"), _fn("_estimateExtH"), _fn("_extHeight"),
    _fn("_fitExt"), _fn("_applyVisibility"),
    "const el = (v) => ({ value: v, style: {}, isConnected: false, scrollHeight: 0 });",
    "const mk = (mode) => ({ size: [420, 600], inputs: [{ name: 'pos_external', link: 7 }],",
    "  widgets: [{ name: 'segments', value: 2 }, { name: 'use_negative', value: true }, { name: 'external_mode', value: mode },",
    "    { name: 'pos_1', element: el('a') }, { name: 'pos_2', element: el('b') }, { name: 'neg_1', element: el('c') },",
    "    { name: 'pls_fold_pos', hidden: true }, { name: 'pls_fold_neg', hidden: true },",
    "    { name: 'pls_ext_pos', hidden: false, element: el(Array(40).fill('// SHOT\\na wide shot at eye level, her figure small,').join('\\n')) },",
    "    { name: 'pls_ext_neg', hidden: true, element: el('') }] });",
    "const out = {};",
    "const n = mk('replace'); _applyVisibility(n);",
    "out.foldPos = _w(n,'pos_1').hidden && _w(n,'pos_2').hidden && !_w(n,'pls_fold_pos').hidden;",
    "out.negKept = !_w(n,'neg_1').hidden;",
    "out.grown = _w(n,'pls_ext_pos')._plsExtH;",
    "n._plsUnfold = { pos: true }; _applyVisibility(n);",
    "out.unfoldGrey = !_w(n,'pos_1').hidden && _w(n,'pos_1').element.style.opacity === DIM;",
    "_w(n,'external_mode').value = 'append'; _applyVisibility(n);",
    "out.append = !_w(n,'pos_1').hidden && _w(n,'pos_1').element.style.opacity === '' && _w(n,'pls_fold_pos').hidden && _w(n,'pls_ext_pos')._plsExtH === EXT_H;",
    "const u = mk('replace'); u.inputs = []; _applyVisibility(u);",
    "out.unwired = !_w(u,'pos_1').hidden && _w(u,'pls_ext_pos')._plsExtH === EXT_H;",
    "console.log(JSON.stringify(out));",
])
node = shutil.which("node")
if node:
    r = subprocess.run([node, "-e", harness], capture_output=True, text=True, timeout=60)
    try:
        o = json.loads(r.stdout.strip().splitlines()[-1])
    except Exception:
        o = {}
        print(r.stdout[-400:], r.stderr[-800:])
    _need(o.get("foldPos") is True, "T4 replace + wired: the own segments fold, the bar shows")
    _need(o.get("negKept") is True, "T4 neg not wired: the own negative stays")
    _need(isinstance(o.get("grown"), int) and 300 < o["grown"] <= 640,
          "T4 not mounted (Nodes 2.0): the green field grows by its line count (%r)" % o.get("grown"))
    _need(o.get("unfoldGrey") is True, "T4 unfolded: shown, greyed")
    _need(o.get("append") is True, "T4 append: fields back, not grey, bar gone, field EXT_H again")
    _need(o.get("unwired") is True, "T4 replace without a cable: nothing folds")
else:
    print("  skip  node not found -- the CTE harness did not run")

# ------------------------------------------------------------------ S
print("S ASCII")
# (ph_progress / ph_power_upscale carry older UTF-8 box rules and docstrings -- not the scripts' rule)
for rel in ("nodes/cine_timeline.py", "nodes/cine_assemble.py", "nodes/ph_runclock.py", "nodes/ph_clip_encode.py", "nodes/ph_cine_timeline.py",
            "web/js/ph_clip_encode.js", "web/js/ph_cine_timeline.js", "tests/test_v1050_triggers_primary.py"):
    b = (ROOT / rel).read_bytes()
    _need(all(c < 128 for c in b), "S %s is pure ASCII" % rel)

print("FAILS %d" % len(FAILS))
sys.exit(1 if FAILS else 0)
