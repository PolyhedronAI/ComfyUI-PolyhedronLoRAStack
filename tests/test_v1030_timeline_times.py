# -*- coding: ascii -*-
"""Guard v1030 -- times by hand, the cut choice, the growing preview
(web/js/ph_cine_timeline.js).

Frank, 28.09.: the handles snap to 1/24 s and deliver 01.667 / 03.250 with
no way to correct them; a double-click on a bar or handle should open the
field; a double-click on the handle ALSO the transition choice; a drag
snap; "prompt the run will produce" should spread like the CTE fields.

  T1  times: moveCut / setShotLength / setLength / setSceneEdge land on the
      frame, keep one frame to the neighbours, the last shot's length is
      the clip length (the grid rounds up), shots past a shorter end go.
  T2  the cut choice: H3 cut is the default; a transition makes the cut a
      scene start by itself, "H3 cut" again takes the flag away.
  T3  transitions follow their scene when scenes come and go (v1028 left
      them on the old boundary index) -- exactly one per boundary, and
      every entry is one the backend accepts (validate() on the result).
  T4  drag snap: frame / 0.25 / 0.5, read from node.properties only;
      the timeline JSON never carries it (DEFAULT equal to the backend's).
  T5  the double-click survives a re-render: two pointer-downs on the same
      key within DBL_MS; a third starts over; another key never counts.
  T6  the preview height: at least PREVIEW_MIN, at most PREVIEW_MAX; the
      field is BASE_H + that; computeSize asks fieldH(node).
  S   statics: the CTE mechanism (scrollHeight only when laid out, no
      onResize hook, no ResizeObserver, no floating window), the snap in
      node.properties, a click without a move is no edit, the handle's
      double-click opens the transition list.

Script-style: exit 0 = pass.
"""
import importlib.util
import json
import re
import shutil
import subprocess
import sys
import tempfile
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
JS = ROOT / "web" / "js" / "ph_cine_timeline.js"
FAILS = []


def _fail(m):
    FAILS.append(m)
    print("  FAIL  " + m)


def _ok(m):
    print("  ok    " + m)


def _need(c, m):
    (_ok if c else _fail)(m)


spec = importlib.util.spec_from_file_location("ct1030", str(ROOT / "nodes" / "cine_timeline.py"))
T = importlib.util.module_from_spec(spec)
spec.loader.exec_module(T)
src = JS.read_text(encoding="utf-8")

# ------------------------------------------------------------------ S
_need(re.search(r"w\.computeSize = \(width\) => \{ _widthCheck\(node\); return \[width, fieldH\(node\)\]; \};", src) is not None,
      "S the node reserves the field through computeSize = fieldH(node)")
fit = src[src.index("function _fitPreview"):src.index("function _widthCheck")]
_need("el.offsetParent !== null && el.clientWidth > 0" in fit and "el.scrollHeight" in fit
      and 'el.style.height = "auto"' in fit and "st.fitting" in fit and "_layout(node)" in fit,   # AMENDED v1150: _layout refits
      "S the preview is measured from scrollHeight only when laid out, re-entry guarded, one refit")
# code, not prose: the header may SAY "no onResize hook" (a guard must not punish its own documentation)
_need(re.search(r"\.onResize\s*=|onResize\s*\(", src) is None and "new ResizeObserver" not in src and "position:fixed" not in src
      and "window.open" not in src, "S no onResize hook, no observer, no floating window (the v604/v610 lessons)")
_need(src.count("_fitPreview(node)") >= 2 and "_fitPreview(this)" in src,
      "S the fit follows every new preview text (route, assemble, run)")
_need("node.properties.pls_tl_snap" in src and "pls_tl_snap" not in json.dumps(T.DEFAULT)
      and "snap" not in re.search(r"export const DEFAULT = \{[^}]*\}", src, re.S).group(0),
      "S the drag snap is a node property, not a field of the timeline JSON")
drag = src[src.index("function _drag"):src.index("function _fitPreview")]
# v1038: onMove carries a third argument (Alt flips A/V linked for one drag);
# the promise is unchanged -- only a MOVED drag calls onMove with done=true
_need(re.search(r"if \(moved\) onMove\(at\(e\), true[,)]", drag) is not None and "else if (onClick) onClick();" in drag
      and "isDouble(node._plsTl, key)" in drag and "snapStep(node)" in drag,
      "S a click without a move is a selection (no save, no undo step); the drag snaps; a double-click is caught first")
# measured in the browser: a popped-open native list takes the keyboard -- the
# choice is LIT, the keyboard stays with `at`
_need("st.open = { focus: \"at\", transition: true }" in src and "showPicker?.(" not in src
      and 'st.transSel.style.outline = "2px solid #f0c87a"' in src and "_openRequest(st)" in src,
      "S a double-click on a handle focuses 'at' and lights the transition choice (no popup stealing the keys)")
# v1031: NOT on the per-node draw hook -- that made the node "hand-drawn"
# for the Nodes 2.0 notice (test_v921 red on v1030); it draws nothing
_need("onDrawForeground" not in src and "new IntersectionObserver" in src and "io.observe(prev)" in src
      and 'window.addEventListener("pointerup"' in src
      # v1157: the pointer release checks the film strips too (_stripCheck follows on the same line)
      and "if (n && n.type === NODE && n._plsTl) { try { _widthCheck(n, true);" in src
      and "(st.firstTries || 0) < 10" in src and "io.disconnect()" in src,
      "S the width check runs when the field comes on screen and when a pointer is let go (no draw hook)")
# measured in the browser: the preview's fit came ~300 ms after the edit and
# was recorded as a step of its own -- Ctrl+Z took back only the height
save = src[src.index("function _save("):src.index("function _link(")]
prv = src[src.index("function _preview("):src.index("function _secFromX(")]
_need("if (commit) _begin(node);" in save and "commitEdit()" not in save
      and "st.previewPending = false;\n        _fitPreview(node);\n        _end(node);" in prv,
      "S an edit opens the tracker bracket, the preview's fit closes it: edit + height = ONE undo step")
_need('subType: "before-change"' in src[src.index("function _begin("):src.index("function _end(")]
      and 'subType: "after-change"' in src[src.index("function _end("):src.index("function _commit(")]
      and "st.bracketTimer = setTimeout(() => _end(node), 4000)" in src,
      "S the bracket is the frontend's before-/after-change, and a lost preview cannot leave it open")
_need("if (!st.bracket && !st.previewPending) _foldLayout();" in fit and "ct.squashState()" in src,
      "S a fit that is not an edit (load, run, width) folds into the current state, no undo step")
_need('ta.addEventListener("change", () => { _commit(node); _render(node); });' in src and "() => commitEdit())" not in src,
      "S finished text edits wait for their preview's fit before they are recorded")
_need("if (typeof st.fitW !== \"number\") {" in src, "S a field that was off screen fits the first frame it is laid out")
_need("setSceneStart(mm, sel.i, cb.checked)" in src and "mm.shots[sel.i].scene = cb.checked" not in src,
      "S the scene checkbox goes through setSceneStart (transitions follow)")
_need("setLength(mm, parseFloat(len.value))" in src, "S the length field goes through setLength (transitions follow)")

if shutil.which("node") is None:
    print("  note  node.js not found -- T1-T6 SKIPPED")
else:
    tmp = Path(tempfile.mkdtemp(prefix="v1030js_"))
    body = (src.replace('import { app } from "../../scripts/app.js";', "const app = { registerExtension() {} };")
               .replace('import { api } from "../../scripts/api.js";', "const api = {};")
               .replace('import { setHidden, refit } from "./ph_widget_vis.js";', "const setHidden = () => {}, refit = () => {};"))
    (tmp / "m.mjs").write_text(body, encoding="utf-8")
    (tmp / "h.mjs").write_text(r"""
import * as T from "./m.mjs";
const base = () => T.parseTimeline(JSON.stringify({ seconds: 10,
  shots: [{ at: 0, text: "a" }, { at: 83 / 24, text: "b" }, { at: 5, text: "c", scene: true }, { at: 8, text: "d", scene: true }],
  transitions: [{ kind: "dissolve", seconds: 0.5 }, { kind: "fade to black", seconds: 1 }] }));
const o = {};
let m = base(); T.moveCut(m, 1, 3.5); o.at = m.shots[1].at;
m = base(); T.moveCut(m, 1, 99); o.atHi = m.shots[1].at;
m = base(); T.setShotLength(m, 0, 2); o.len0 = m.shots[1].at;
m = base(); T.setShotLength(m, 1, 0); o.lenMin = m.shots[2].at - m.shots[1].at;
m = base(); T.setShotLength(m, 3, 3); o.lenLast = [m.seconds, T.duration(m)];
m = base(); T.setLength(m, 6); o.cut = [m.shots.map((s) => s.at), m.transitions];
m = base(); T.setSceneEdge(m, 2, "start", 4.5); T.setSceneEdge(m, 2, "end", 7.5); o.edge = [m.shots[2].at, m.shots[3].at];
m = base(); T.setSceneEdge(m, 1, "start", 2); o.edge1 = m.shots[0].at;
m = base(); o.c0 = [T.cutChoice(m, 1).kind, T.cutChoice(m, 2).kind, T.cutChoice(m, 3).kind];
T.setCutChoice(m, 1, "wipe left", 0.4); o.c1 = [!!m.shots[1].scene, m.transitions, T.scenesOf(m).length];
T.setCutChoice(m, 1, T.H3_CUT); o.c2 = [!!m.shots[1].scene, m.transitions];
T.setCutChoice(m, 2, "slide left", 99); o.c3 = m.transitions;
m = base(); T.removeShot(m, 2); o.rm = m.transitions;
m = base(); T.setSceneStart(m, 1, true); T.removeShot(m, 0); o.rm0 = [m.shots[0].at, !!m.shots[0].scene, m.transitions];
m = base(); T.setSceneStart(m, 1, true); o.ss = m.transitions;
o.model = m;
o.snap = [T.snapTo(1.667, 0), T.snapTo(1.667, 0.25), T.snapTo(1.667, 0.5), T.snapTo(3.3, 0.25)];
o.step = [T.snapStep({ properties: { pls_tl_snap: 0.25 } }), T.snapStep({ properties: { pls_tl_snap: 0.3 } }), T.snapStep({}), T.snapStep(null)];
const st = {}; o.dbl = [T.isDouble(st, "a", 0), T.isDouble(st, "a", 100), T.isDouble(st, "a", 200), T.isDouble(st, "b", 250), T.isDouble(st, "a", 300), T.isDouble(st, "a", 1000)];
o.ph = [T.previewHeight(0), T.previewHeight(300.2), T.previewHeight(1e7), T.PREVIEW_MIN, T.PREVIEW_MAX];
// AMENDED v1150: the preview is a section that counts while it is open (its switch in the status row)
o.fh = [T.fieldH({ properties: { pls_cine_prev_open: true }, _plsTl: { prevH: 300 } }), T.fieldH({}), T.fieldH({ properties: { pls_cine_prev_open: true } }), T.PREVIEW_DEF, T.ROOT_GAP];
o.def = T.DEFAULT;
console.log(JSON.stringify(o));
""", encoding="utf-8")
    r = subprocess.run(["node", str(tmp / "h.mjs")], capture_output=True, text=True, timeout=60)
    shutil.rmtree(tmp, ignore_errors=True)
    line = [l for l in r.stdout.splitlines() if l.startswith("{")]
    if r.returncode != 0 or not line:
        _fail("T harness failed: %s" % r.stderr.strip()[-400:])
    else:
        d = json.loads(line[-1])
        F = 1 / 24.0
        _need(abs(d["at"] - 3.5) < 1e-9 and abs(d["atHi"] - (5 - F)) < 1e-9,
              "T1 a typed cut lands on the frame (3.5) and keeps a frame to the next cut %r" % d["atHi"])
        _need(abs(d["len0"] - 2) < 1e-9 and abs(d["lenMin"] - F) < 1e-9,
              "T1 a shot's length moves the next cut; the shortest shot is one frame")
        _need(d["lenLast"][0] == 11 and abs(d["lenLast"][1] - T.frames_for(11) / 24.0) < 1e-9,
              "T1 the last shot's length is the clip length, the grid rounds it up %r" % d["lenLast"])
        _need(d["cut"][0] == [0, 83 / 24.0, 5] and d["cut"][1] == [{"kind": "dissolve", "seconds": 0.5}],
              "T1 a shorter clip drops the shots past its end, and their transitions with them %r" % d["cut"])
        _need(abs(d["edge"][0] - 4.5) < 1e-9 and abs(d["edge"][1] - 7.5) < 1e-9 and d["edge1"] == 0,
              "T1 a scene's start/end move the cuts at its edges; scene 1 starts at 0")
        _need(d["c0"] == ["H3 cut (inside the clip)", "dissolve", "fade to black"],
              "T2 the choice at a cut: H3 cut inside a scene, the boundary's transition at a scene start %r" % d["c0"])
        _need(d["c1"][0] is True and d["c1"][2] == 4 and d["c1"][1] == [{"kind": "wipe left", "seconds": 0.4},
              {"kind": "dissolve", "seconds": 0.5}, {"kind": "fade to black", "seconds": 1}],
              "T2+T3 a transition at an H3 cut makes it a scene start; the later transitions keep their scenes %r" % d["c1"][1])
        _need(d["c2"][0] is False and d["c2"][1] == [{"kind": "dissolve", "seconds": 0.5}, {"kind": "fade to black", "seconds": 1}],
              "T2+T3 'H3 cut' again takes the flag away; the others stay where they were")
        _need(d["c3"][0] == {"kind": "slide left", "seconds": 5}, "T2 a transition length is clamped to 0.05 .. 5 s")
        _need(d["rm"] == [{"kind": "fade to black", "seconds": 1}],
              "T3 removing a scene's start shot removes ITS transition, not the next one %r" % d["rm"])
        # the shot moving up WAS a scene start (sharpened after the mutation probe: a case without the flag proved nothing)
        _need(d["rm0"][0] == 0 and d["rm0"][1] is False
              and d["rm0"][2] == [{"kind": "dissolve", "seconds": 0.5}, {"kind": "fade to black", "seconds": 1}],
              "T3 removing shot 1: the next (a scene start) becomes shot 1 at 0, no flag, its transition gone %r" % (d["rm0"][2],))
        _need(d["ss"] == [{"kind": "cut", "seconds": 0.5}, {"kind": "dissolve", "seconds": 0.5}, {"kind": "fade to black", "seconds": 1}],
              "T3 a new scene start gets a plain cut; the existing transitions move with their scenes %r" % d["ss"])
        _need(T.validate(T.load(json.dumps(d["model"]))) == [],
              "T3 the edited timeline passes the backend's validation")
        _need(abs(d["snap"][0] - 40 / 24.0) < 1e-9 and d["snap"][1] == 1.75 and d["snap"][2] == 1.5 and d["snap"][3] == 3.25,
              "T4 snap: frame / 0.25 s / 0.5 s %r" % d["snap"])
        _need(d["step"] == [0.25, 0, 0, 0], "T4 only 0 / 0.25 / 0.5 from node.properties; anything else is the frame")
        _need(d["def"] == T.DEFAULT, "T4 the timeline JSON's DEFAULT still equals the backend's (no view setting in it)")
        _need(d["dbl"] == [False, True, False, False, False, False],
              "T5 double = two downs on one key within the window; the pair is spent; another key breaks it %r" % d["dbl"])
        _need(d["ph"] == [d["ph"][3], 301, d["ph"][4], d["ph"][3], d["ph"][4]] and d["ph"][3] >= 64,
              "T6 preview height clamped to [%d, %d], rounded up" % (d["ph"][3], d["ph"][4]))
        _need(d["fh"][0] == d["fh"][1] + 300 + d["fh"][4] and d["fh"][2] == d["fh"][1] + d["fh"][3] + d["fh"][4],
              "T6 the field is its sections + the open preview (its fitted height, the default before the first fit)")

print()
if FAILS:
    print("test_v1030_timeline_times: %d FAILURE(S)" % len(FAILS))
    sys.exit(1)
print("test_v1030_timeline_times: PASS (T1-T6, S)")
