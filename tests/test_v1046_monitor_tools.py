# -*- coding: ascii -*-
"""Guard v1046 -- the monitor on top, the timeline's toolbar over the tracks,
no takes UI, and E4 (Frank 30.09.: "the preview and the cut buttons up, the
tracks below -- with 20 tracks the preview lies in nirvana"; "decouple the cut
buttons from the preview, two different areas"; "the green take 1/3 can go --
show me the last clip -- this dropdown too -- OR make them tracks").

  M1  the order in the field: length row, MONITOR (picture + play, time,
      exact preview, live, export), the status line, the TOOLBAR (cut, heal,
      delete, A/V linked, drag snap | reset cuts, cuts -> prompt, keep tail,
      span | new render -> new track, restore on open, takes -> tracks), the
      tracks, then shots/anchors buttons, the selection, the sound, the render
      row, the prompt. The help text is the toolbar's tooltip.
  M2  no take chip, no take list; V1 shows the NEWEST run -- the server
      ignores a take in the edit unless "new render -> new track" is on (then
      V1's take is pinned; clean_edit keeps auto_track).
  M3  E4: _setAuto pins V1's takes / lets them go; a finished run (the
      register answer) becomes a track on top when the switch is on;
      takes -> tracks adds every older take (the oldest first, so the newest
      of them is on top) at its scene's start; restore on open off = the
      node's FIRST load in a page session starts with an empty edit (an
      undo-style reload later in the session does not).
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


js = (ROOT / "web" / "js" / "ph_cine_timeline.js").read_text(encoding="utf-8")
fld = js[js.index("function _field("):]

# ------------------------------------------------------------------ M1
# v1047: + the FILM row, the lanes in a scroller
# AMENDED v1150: top bar, head row, refs, the WORK area (the selected shot left, the monitor right), then as before
# AMENDED v1153: the sound column between the shot and the monitor
_need("root.append(topBar, headRow, refsRow, work, state, tools, film, scroller, bar, status, prev, pop);" in fld and "work.append(left, sndRow, stack);" in fld, "M1 the monitor on top, the toolbar over the tracks")
_need("r1.append(play, ptime);" in fld and "r2.append(exactBtn, live, join);" in fld and 'stack.dataset.pls = "monitor";' in fld,
      "M1 the monitor: picture, play, time, exact preview, live, export -- nothing else")
_need("tools.append(splitBtn, healBtn, delBtn, linkBtn, typeBtn, snapSel, sep(), resetBtn, cutsBtn, tailLab,\n        sep(), autoLab, restLab, takesBtn, clearBtn);" in fld,   # AMENDED v1200: + x tracks (v1198)
      "M1 the toolbar: the editing tools, the E4 switches (v1047: the span moved to the FILM row; v1153: + V/A by type)")
# AMENDED v1150: the toolbar is a section, fitted to its rows (TOOLS_H at least)
_need("export const TOOLS_H = 40, TOOLS_MAX = 140, AUTO_KEY = \"auto_track\", RESTORE_KEY = \"pls_cine_restore\";" in js
      and '["tools", st && typeof st.toolsH === "number" ? st.toolsH : TOOLS_H, true],' in js, "M1 the node's height counts the toolbar")
_need("tools.title = \"drag the playhead" in fld and "the divider under a lane = its height" in fld, "M1 the help is the toolbar's tooltip")

# ------------------------------------------------------------------ M2
_need('dataset.pls = "take";' not in js and 'dataset.pls = "takesel";' not in js and "take \" + (c.take + 1)" not in js,
      "M2 no take chip, no take list")
spec = importlib.util.spec_from_file_location("ce1046", str(ROOT / "nodes" / "cine_edit.py"))
E = importlib.util.module_from_spec(spec)
spec.loader.exec_module(E)
_need(E.clean_edit({"auto_track": 1, "takes": {"2": "a/b.mp4"}}) == {"auto_track": True, "takes": {"2": "a/b.mp4"}}, "M2 clean_edit keeps auto_track")
ff = shutil.which("ffmpeg")
if ff:
    TMP = Path(tempfile.mkdtemp(prefix="v1046_"))
    for d in ("output/fx", "input", "temp"):
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
    sys.modules.update({"comfy": comfy, "comfy.nested_tensor": nt, "comfy_api": capi,
                        "comfy_api.input_impl": ii, "comfy_api.util": ut, "folder_paths": fp})
    pkg = types.ModuleType("plsn1046")
    pkg.__path__ = [str(ROOT / "nodes")]
    sys.modules["plsn1046"] = pkg
    TL = importlib.import_module("plsn1046.ph_cine_timeline")
    T = sys.modules["plsn1046.cine_timeline"]
    import os
    import time
    for i, n in enumerate((48, 72)):
        f = TMP / "output" / "fx" / ("whole_0000%d_.mp4" % (i + 1))
        subprocess.run([ff, "-v", "error", "-y", "-f", "lavfi", "-i", "testsrc2=s=64x36:r=24", "-frames:v", str(n), "-c:v", "libx264", "-pix_fmt", "yuv420p", str(f)],
                       capture_output=True)
        os.utime(f, (time.time() - 100 + i * 50, time.time() - 100 + i * 50))
    model = T.load(json.dumps({"seconds": 2.0, "project": "fx", "render": "whole", "shots": [{"at": 0, "cut": "the camera cuts to", "text": "x"}]}))
    # v1080 (DECLARED): "new render -> new track" is ON unless the edit says
    # auto_track: false -- the ignored-take case is the explicit OFF now
    old = {"takes": {"0": "fx/whole_00001_.mp4"}}
    a = TL.edit_plan(model, dict(old, auto_track=False), strips=False)["clips"][0]
    b = TL.edit_plan(model, dict(old, auto_track=True), strips=False)["clips"][0]
    _need(a["name"] == "whole_00002_.mp4" and a["take"] == 0 and a["frames"] == 72,
          "M2 with the switch OFF a pinned take is ignored: V1 shows the newest run (%s)" % a["name"])
    _need(b["name"] == "whole_00001_.mp4" and b["take"] == 1, "M2 with 'new render -> new track' on, V1 keeps its pinned take")
    shutil.rmtree(TMP, ignore_errors=True)
else:
    print("  note  ffmpeg not found -- M2 server half SKIPPED")

# ------------------------------------------------------------------ M3
# v1080 (DECLARED): off is stored as an explicit false, the gate reads autoOn()
_need("if (on) {\n        e[AUTO_KEY] = true;" in js and "} else { e[AUTO_KEY] = false; delete e.takes; }" in js, "M3 the switch pins V1's takes / lets them go")
# v1088 (DECLARED): the track goes to THIS run's timeline (run.node), still in its project only
_need("if (n && projectName(_model(n)) === run.project) {" in js and "const n = tlById(run.node);" in js
      and "if (!autoOn(e)) return;" in js, "M3 a registered run becomes a track when the switch is on (its project only)")
# AMENDED v1200 (v1198, Frank 21.10.: 47 tracks): takes -> tracks = ONE track per render, the oldest render first (the
# newest ends on top), each run's clip at its scene's start, what V1 shows or a track has skipped (test_v1198_one_track)
_need("const plan = takePlan(R.renders, skip);" in js and "const at = _sceneAt(node, r.k);" in js and "return out.reverse();" in js
      and "skip.add(c.takes[c.take || 0]);" in js,
      "M3 takes -> tracks: every older render not yet on a track, oldest first, each clip at its scene's start")
# v1082 (DECLARED): the wipe keeps the settings (wipeEdit), not {}
# v1088 (DECLARED): the memory is per workflow + node (seenKeyOf), not per node id
_need("if (this.properties && this.properties[RESTORE_KEY] === false && !_SEEN.has(seenKey)) this.properties[EDIT_KEY] = wipeEdit(this.properties[EDIT_KEY]);" in js
      and "_SEEN.add(seenKey);" in js and "const seenKey = seenKeyOf(this);" in js, "M3 restore off: only the first load in a page session starts empty")

bad = [f for f in ("web/js/ph_cine_timeline.js", "nodes/ph_cine_timeline.py", "nodes/cine_edit.py", "tests/test_v1046_monitor_tools.py")
       if any(ord(ch) > 127 for ch in (ROOT / f).read_text(encoding="utf-8"))]
_need(not bad, "S ASCII only %r" % bad)
if FAILS:
    print("test_v1046_monitor_tools: %d FAILURE(S)" % len(FAILS))
    sys.exit(1)
print("test_v1046_monitor_tools: PASS (M1-M3, S)")
